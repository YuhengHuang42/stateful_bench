"""Unit tests for OccurenceBook."""
# python -m pytest Sgenerator/test/test_occurence_book.py

import copy
import json
import os
import tempfile
import pytest

from Sgenerator.state import OccurenceBook


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def empty_book():
    return OccurenceBook()


@pytest.fixture
def seeded_book():
    """Book pre-loaded with a handful of transition pairs."""
    data = {
        ("NONE", "AddSession"): 3,
        ("AddSession", "QuerySession"): 2,
        ("QuerySession", "DeleteSession"): 1,
    }
    return OccurenceBook(data=data)


@pytest.fixture
def tmp_json(tmp_path):
    """Return a path string inside a temp directory."""
    return str(tmp_path / "occ_book.json")


# ===================================================================
# 1. Basic dict-like interface
# ===================================================================
class TestDictInterface:
    def test_empty_book(self, empty_book):
        assert len(empty_book) == 0
        assert ("X", "Y") not in empty_book
        assert empty_book.get(("X", "Y")) == 0
        assert empty_book.get(("X", "Y"), 5) == 5

    def test_setitem_getitem(self, empty_book):
        empty_book[("A", "B")] = 10
        assert ("A", "B") in empty_book
        assert empty_book[("A", "B")] == 10
        assert len(empty_book) == 1

    def test_items_keys_values(self, seeded_book):
        assert set(seeded_book.keys()) == {
            ("NONE", "AddSession"),
            ("AddSession", "QuerySession"),
            ("QuerySession", "DeleteSession"),
        }
        assert list(seeded_book.values()) == [3, 2, 1] or set(seeded_book.values()) == {3, 2, 1}
        assert len(list(seeded_book.items())) == 3

    def test_iter(self, seeded_book):
        pairs = list(seeded_book)
        assert len(pairs) == 3
        assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)

    def test_repr(self, empty_book):
        r = repr(empty_book)
        assert r.startswith("OccurenceBook(")


# ===================================================================
# 2. record / record_many
# ===================================================================
class TestRecord:
    def test_record_new_pair(self, empty_book):
        empty_book.record(("A", "B"))
        assert empty_book[("A", "B")] == 1

    def test_record_increments(self, empty_book):
        empty_book.record(("A", "B"))
        empty_book.record(("A", "B"))
        empty_book.record(("A", "B"))
        assert empty_book[("A", "B")] == 3

    def test_record_many(self, empty_book):
        pairs = [("A", "B"), ("C", "D"), ("A", "B")]
        empty_book.record_many(pairs)
        assert empty_book[("A", "B")] == 2
        assert empty_book[("C", "D")] == 1


# ===================================================================
# 3. Round tracking: begin_round / get_round_transitions / end_round
# ===================================================================
class TestRoundTracking:
    def test_no_begin_raises(self, empty_book):
        with pytest.raises(RuntimeError, match="begin_round"):
            empty_book.get_round_transitions()

    def test_empty_round(self, seeded_book):
        seeded_book.begin_round()
        diff = seeded_book.get_round_transitions()
        assert diff == {}

    def test_new_pairs_only(self, seeded_book):
        seeded_book.begin_round()
        seeded_book.record(("X", "Y"))
        seeded_book.record(("X", "Y"))
        diff = seeded_book.get_round_transitions()
        assert diff == {("X", "Y"): 2}

    def test_incremented_existing_pair(self, seeded_book):
        seeded_book.begin_round()
        seeded_book.record(("NONE", "AddSession"))
        diff = seeded_book.get_round_transitions()
        assert diff == {("NONE", "AddSession"): 1}

    def test_mixed_new_and_existing(self, seeded_book):
        seeded_book.begin_round()
        seeded_book.record(("NONE", "AddSession"))
        seeded_book.record(("NewAPI", "AnotherAPI"))
        diff = seeded_book.get_round_transitions()
        assert ("NONE", "AddSession") in diff and diff[("NONE", "AddSession")] == 1
        assert ("NewAPI", "AnotherAPI") in diff and diff[("NewAPI", "AnotherAPI")] == 1

    def test_end_round_clears_snapshot(self, seeded_book):
        seeded_book.begin_round()
        seeded_book.record(("X", "Y"))
        diff = seeded_book.end_round()
        assert diff == {("X", "Y"): 1}
        with pytest.raises(RuntimeError):
            seeded_book.get_round_transitions()

    def test_multiple_rounds(self, empty_book):
        empty_book.begin_round()
        empty_book.record(("A", "B"))
        r1 = empty_book.end_round()
        assert r1 == {("A", "B"): 1}

        empty_book.begin_round()
        empty_book.record(("A", "B"))
        empty_book.record(("C", "D"))
        r2 = empty_book.end_round()
        assert r2 == {("A", "B"): 1, ("C", "D"): 1}

    def test_get_round_transitions_idempotent_before_end(self, empty_book):
        empty_book.begin_round()
        empty_book.record(("A", "B"))
        d1 = empty_book.get_round_transitions()
        d2 = empty_book.get_round_transitions()
        assert d1 == d2


# ===================================================================
# 4. Inject transitions
# ===================================================================
class TestInject:
    def test_inject_new(self, empty_book):
        empty_book.inject_transition(("A", "B"), 5)
        assert empty_book[("A", "B")] == 5

    def test_inject_keeps_higher(self, seeded_book):
        seeded_book.inject_transition(("NONE", "AddSession"), 2)
        assert seeded_book[("NONE", "AddSession")] == 3  # was 3, inject 2 -> keep 3

    def test_inject_overwrites_lower(self, seeded_book):
        seeded_book.inject_transition(("NONE", "AddSession"), 10)
        assert seeded_book[("NONE", "AddSession")] == 10

    def test_inject_transitions_batch(self, empty_book):
        empty_book.inject_transitions({
            ("A", "B"): 3,
            ("C", "D"): 7,
        })
        assert empty_book[("A", "B")] == 3
        assert empty_book[("C", "D")] == 7

    def test_inject_appears_in_round_diff(self, empty_book):
        empty_book.begin_round()
        empty_book.inject_transition(("A", "B"), 4)
        diff = empty_book.end_round()
        assert diff == {("A", "B"): 4}


# ===================================================================
# 5. Discard workflow
# ===================================================================
class TestDiscard:
    def test_mark_and_apply_partial(self, seeded_book):
        seeded_book.mark_discarded(("NONE", "AddSession"), count=1)
        assert seeded_book.has_pending_discards()
        removed = seeded_book.apply_discards()
        assert removed == set()  # 3-1=2, still present
        assert seeded_book[("NONE", "AddSession")] == 2
        assert not seeded_book.has_pending_discards()

    def test_mark_and_apply_exact_zero(self, seeded_book):
        seeded_book.mark_discarded(("QuerySession", "DeleteSession"), count=1)
        removed = seeded_book.apply_discards()
        assert ("QuerySession", "DeleteSession") in removed
        assert ("QuerySession", "DeleteSession") not in seeded_book

    def test_mark_discard_full_removal(self, seeded_book):
        seeded_book.mark_discarded(("AddSession", "QuerySession"), count=-1)
        removed = seeded_book.apply_discards()
        assert ("AddSession", "QuerySession") in removed
        assert ("AddSession", "QuerySession") not in seeded_book

    def test_mark_discarded_many(self, seeded_book):
        pairs = [("NONE", "AddSession"), ("AddSession", "QuerySession")]
        seeded_book.mark_discarded_many(pairs, count=-1)
        removed = seeded_book.apply_discards()
        assert ("NONE", "AddSession") in removed
        assert ("AddSession", "QuerySession") in removed

    def test_discard_nonexistent_pair(self, seeded_book):
        seeded_book.mark_discarded(("Foo", "Bar"), count=1)
        removed = seeded_book.apply_discards()
        assert removed == set()

    def test_pending_discards_property(self, seeded_book):
        seeded_book.mark_discarded(("NONE", "AddSession"), count=2)
        pd = seeded_book.pending_discards
        assert pd == {("NONE", "AddSession"): 2}
        pd[("X", "Y")] = 99  # mutating the copy
        assert ("X", "Y") not in seeded_book.pending_discards


# ===================================================================
# 6. Serialization: to_dict / from_dict / save / load
# ===================================================================
class TestSerialization:
    def test_round_trip_to_from_dict(self, seeded_book):
        seeded_book.mark_discarded(("NONE", "AddSession"), count=1)
        d = seeded_book.to_dict()
        restored = OccurenceBook.from_dict(d)
        assert restored.get(("NONE", "AddSession")) == seeded_book.get(("NONE", "AddSession"))
        assert restored.pending_discards == seeded_book.pending_discards

    def test_round_trip_with_snapshot(self, seeded_book):
        seeded_book.begin_round()
        seeded_book.record(("X", "Y"))
        d = seeded_book.to_dict()
        assert "round_snapshot" in d
        restored = OccurenceBook.from_dict(d)
        diff = restored.get_round_transitions()
        assert diff == {("X", "Y"): 1}

    def test_save_and_load(self, seeded_book, tmp_json):
        seeded_book.mark_discarded(("NONE", "AddSession"), count=1)
        seeded_book.save(tmp_json)
        assert os.path.exists(tmp_json)

        loaded = OccurenceBook.load(tmp_json)
        assert loaded.get(("NONE", "AddSession")) == 3
        assert loaded.has_pending_discards()
        assert loaded.pending_discards == {("NONE", "AddSession"): 1}

    def test_load_missing_file_returns_empty(self, tmp_path):
        path = str(tmp_path / "nonexistent.json")
        book = OccurenceBook.load(path)
        assert len(book) == 0
        assert book._persist_path == path

    def test_save_without_path_raises(self, empty_book):
        with pytest.raises(ValueError, match="No persist_path"):
            empty_book.save()

    def test_json_content_validity(self, seeded_book, tmp_json):
        seeded_book.save(tmp_json)
        with open(tmp_json) as f:
            raw = json.load(f)
        assert "data" in raw
        assert "pending_discards" in raw
        assert all("||" in k for k in raw["data"])


# ===================================================================
# 7. Deepcopy
# ===================================================================
class TestDeepcopy:
    def test_deepcopy_is_independent(self, seeded_book):
        copied = copy.deepcopy(seeded_book)
        copied.record(("NEW", "PAIR"))
        assert ("NEW", "PAIR") not in seeded_book
        assert ("NEW", "PAIR") in copied

    def test_deepcopy_preserves_pending_discards(self, seeded_book):
        seeded_book.mark_discarded(("NONE", "AddSession"), count=1)
        copied = copy.deepcopy(seeded_book)
        assert copied.has_pending_discards()
        copied.apply_discards()
        assert seeded_book.has_pending_discards()  # original untouched

    def test_deepcopy_preserves_round_snapshot(self, seeded_book):
        seeded_book.begin_round()
        seeded_book.record(("X", "Y"))
        copied = copy.deepcopy(seeded_book)
        diff = copied.get_round_transitions()
        assert diff == {("X", "Y"): 1}
        seeded_book.record(("X", "Y"))
        diff_original = seeded_book.get_round_transitions()
        assert diff_original[("X", "Y")] == 2
        diff_copy = copied.get_round_transitions()
        assert diff_copy[("X", "Y")] == 1  # still 1


# ===================================================================
# 8. Migration from raw dict
# ===================================================================
class TestMigration:
    def test_from_raw_dict(self):
        raw = {("A", "B"): 5, ("C", "D"): 2}
        book = OccurenceBook.from_raw_dict(raw)
        assert book[("A", "B")] == 5
        assert len(book) == 2

    def test_from_raw_dict_independent(self):
        raw = {("A", "B"): 1}
        book = OccurenceBook.from_raw_dict(raw)
        book.record(("A", "B"))
        assert raw[("A", "B")] == 1  # original dict unchanged

    def test_from_empty_dict(self):
        book = OccurenceBook.from_raw_dict({})
        assert len(book) == 0


# ===================================================================
# 9. Cross-process persistence scenario
# ===================================================================
class TestCrossProcessScenario:
    """Simulate: process 1 generates & saves, process 2 marks discards & saves,
    process 1 reloads and applies discards before next generation round."""

    def test_full_lifecycle(self, tmp_json):
        # --- Process 1: generate ---
        book = OccurenceBook(persist_path=tmp_json)
        book.begin_round()
        book.record(("NONE", "Create"))
        book.record(("Create", "Read"))
        book.record(("Read", "Update"))
        round_diff = book.end_round()
        assert len(round_diff) == 3
        book.save()

        # --- Process 2: LLM review marks "Read -> Update" as discarded ---
        book2 = OccurenceBook.load(tmp_json)
        assert book2[("Read", "Update")] == 1
        book2.mark_discarded(("Read", "Update"), count=-1)
        # LLM also adds a new transition
        book2.inject_transition(("Create", "Delete"), 1)
        book2.save()

        # --- Process 1: next round ---
        book3 = OccurenceBook.load(tmp_json)
        assert book3.has_pending_discards()
        removed = book3.apply_discards()
        assert ("Read", "Update") in removed
        assert ("Read", "Update") not in book3
        assert book3[("Create", "Delete")] == 1
        assert book3[("NONE", "Create")] == 1
