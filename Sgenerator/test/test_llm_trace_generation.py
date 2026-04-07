"""Tests for llm_trace_generation: LLM-assisted test case generation pipeline.

Covers unit tests (pytest) *and* a verbose interactive demo (``__main__``)
that prints every step so you can visually confirm the pipeline works.

Run tests:
    python -m pytest Sgenerator/test/test_llm_trace_generation.py -v

Run interactive demo with prints:
    python -m Sgenerator.test.test_llm_trace_generation
"""
import copy
import random
import pytest

from Sgenerator.state import OccurenceBook, TraceGenerator, generate_program
from Sgenerator.tensor_state import (
    TensorVariableSchema,
    TensorRandomInitializer,
    TensorEvaluator,
)
from llm_trace_generation import (
    LLMModificationResult,
    generate_and_collect_with_llm,
    llm_modify_program_mock,
    reconcile_occurence_book,
    _compute_net_diff,
    _build_program_info,
)


TRACE_CONFIG = {
    "init_local_state_num_range": (1, 2),
}

COMMON_KWARGS = dict(
    schema_class=TensorVariableSchema,
    random_init_class=TensorRandomInitializer,
    evaluator_class=TensorEvaluator,
    trace_config=TRACE_CONFIG,
    num_of_apis=3,
    control_position_candidate=[2],
    enable_if_else=False,
    enable_coverage=False,
)

DEMO_KWARGS = dict(
    schema_class=TensorVariableSchema,
    random_init_class=TensorRandomInitializer,
    evaluator_class=TensorEvaluator,
    trace_config=TRACE_CONFIG,
    num_of_apis=5,
    control_position_candidate=[1, 2, 3, 4],
    enable_if_else=True,
    enable_coverage=False,
)


# ===================================================================
# Helpers: fake LLM functions for testing
# ===================================================================

def _fake_llm_noop(program, program_info, round_transitions, task_context=None):
    """LLM that returns the program unchanged (same as placeholder)."""
    return LLMModificationResult(
        modified_program=program,
        removed_transitions=[],
        added_transitions={},
        llm_metadata={"fake": "noop"},
    )


def _make_fake_llm_remover(n_remove=1):
    """Return an LLM function that removes the first *n_remove* round transitions."""
    def _fake_llm(program, program_info, round_transitions, task_context=None):
        to_remove = list(round_transitions.keys())[:n_remove]
        return LLMModificationResult(
            modified_program=program,
            removed_transitions=to_remove,
            added_transitions={},
            llm_metadata={"fake": "remover", "removed_count": len(to_remove)},
        )
    return _fake_llm


def _fake_llm_injector(program, program_info, round_transitions, task_context=None):
    """LLM that injects a synthetic transition pair."""
    return LLMModificationResult(
        modified_program=program,
        removed_transitions=[],
        added_transitions={("LLM_Injected_A", "LLM_Injected_B"): 5},
        llm_metadata={"fake": "injector"},
    )


def _fake_llm_remover_and_injector(program, program_info, round_transitions, task_context=None):
    """LLM that removes the first transition AND adds a new one."""
    to_remove = list(round_transitions.keys())[:1]
    return LLMModificationResult(
        modified_program=program,
        removed_transitions=to_remove,
        added_transitions={("LLM_New_X", "LLM_New_Y"): 3},
        llm_metadata={"fake": "both"},
    )


# ===================================================================
# 1. LLMModificationResult dataclass
# ===================================================================
class TestLLMModificationResult:
    def test_defaults(self):
        r = LLMModificationResult(modified_program="x = 1")
        assert r.modified_program == "x = 1"
        assert r.removed_transitions == []
        assert r.added_transitions == {}
        assert r.modified_program_info is None
        assert r.llm_metadata == {}

    def test_with_values(self):
        r = LLMModificationResult(
            modified_program="y = 2",
            removed_transitions=[("A", "B")],
            added_transitions={("C", "D"): 2},
            llm_metadata={"model": "test"},
        )
        assert ("A", "B") in r.removed_transitions
        assert r.added_transitions[("C", "D")] == 2


# ===================================================================
# 2. reconcile_occurence_book
# ===================================================================
class TestReconcileOccurenceBook:
    def test_remove_transitions(self):
        book = OccurenceBook()
        book.record(("A", "B"))
        book.record(("C", "D"))
        llm_result = LLMModificationResult(
            modified_program="",
            removed_transitions=[("A", "B")],
        )
        removed = reconcile_occurence_book(book, llm_result)
        assert ("A", "B") in removed
        assert ("A", "B") not in book
        assert ("C", "D") in book

    def test_inject_transitions(self):
        book = OccurenceBook()
        book.record(("A", "B"))
        llm_result = LLMModificationResult(
            modified_program="",
            added_transitions={("X", "Y"): 10},
        )
        reconcile_occurence_book(book, llm_result)
        assert ("X", "Y") in book
        assert book[("X", "Y")] == 10

    def test_remove_and_inject(self):
        book = OccurenceBook()
        book.record(("A", "B"))
        book.record(("C", "D"))
        llm_result = LLMModificationResult(
            modified_program="",
            removed_transitions=[("A", "B")],
            added_transitions={("NEW", "PAIR"): 7},
        )
        removed = reconcile_occurence_book(book, llm_result)
        assert ("A", "B") in removed
        assert ("NEW", "PAIR") in book
        assert book[("NEW", "PAIR")] == 7

    def test_deferred_apply(self):
        book = OccurenceBook()
        book.record(("A", "B"))
        llm_result = LLMModificationResult(
            modified_program="",
            removed_transitions=[("A", "B")],
        )
        removed = reconcile_occurence_book(book, llm_result, apply_immediately=False)
        assert removed == set()
        assert ("A", "B") in book  # still present
        assert book.has_pending_discards()
        actual_removed = book.apply_discards()
        assert ("A", "B") in actual_removed

    def test_noop_result(self):
        book = OccurenceBook()
        book.record(("A", "B"))
        llm_result = LLMModificationResult(modified_program="")
        removed = reconcile_occurence_book(book, llm_result)
        assert removed == set()
        assert len(book) == 1


# ===================================================================
# 3. _compute_net_diff
# ===================================================================
class TestComputeNetDiff:
    def test_pure_additions(self):
        old = OccurenceBook()
        new = OccurenceBook()
        new.record(("A", "B"))
        new.record(("A", "B"))
        diff = _compute_net_diff(old, new)
        assert diff == {("A", "B"): 2}

    def test_no_change(self):
        book = OccurenceBook()
        book.record(("A", "B"))
        diff = _compute_net_diff(book, copy.deepcopy(book))
        assert diff == {}

    def test_removal_not_in_diff(self):
        old = OccurenceBook()
        old.record(("A", "B"))
        old.record(("C", "D"))
        new = OccurenceBook()
        new.record(("C", "D"))
        diff = _compute_net_diff(old, new)
        assert ("A", "B") not in diff
        assert diff == {}

    def test_mixed(self):
        old = OccurenceBook()
        old.record(("A", "B"))
        new = OccurenceBook()
        new.record(("A", "B"))
        new.record(("A", "B"))
        new.record(("X", "Y"))
        diff = _compute_net_diff(old, new)
        assert diff == {("A", "B"): 1, ("X", "Y"): 1}


# ===================================================================
# 4. Default placeholder LLM function
# ===================================================================
class TestPlaceholderLLM:
    def test_returns_unchanged_program(self):
        result = llm_modify_program_mock(
            program="hello = 1",
            program_info={},
            round_transitions={("A", "B"): 1},
        )
        assert result.modified_program == "hello = 1"
        assert result.removed_transitions == []
        assert result.added_transitions == {}
        assert result.llm_metadata.get("placeholder") is True


# ===================================================================
# 5. generate_and_collect_with_llm — placeholder (no-op LLM)
# ===================================================================
class TestGenerateAndCollectWithLLM:
    def test_basic_success(self):
        evaluator, success, book, diff, llm_result = generate_and_collect_with_llm(
            **COMMON_KWARGS, occurence_book=OccurenceBook(),
        )
        assert success
        assert isinstance(book, OccurenceBook)
        assert len(book) > 0
        assert isinstance(diff, dict)
        assert llm_result is not None

    def test_returns_five_values(self):
        ret = generate_and_collect_with_llm(
            **COMMON_KWARGS, occurence_book=OccurenceBook(),
        )
        assert len(ret) == 5

    def test_accepts_raw_dict(self):
        _, success, book, _, _ = generate_and_collect_with_llm(
            **COMMON_KWARGS, occurence_book={},
        )
        assert success
        assert isinstance(book, OccurenceBook)

    def test_accepts_none_book(self):
        _, success, book, _, _ = generate_and_collect_with_llm(
            **COMMON_KWARGS, occurence_book=None,
        )
        assert success
        assert isinstance(book, OccurenceBook)

    def test_original_book_not_mutated(self):
        original = OccurenceBook()
        original.record(("Sentinel", "Pair"))
        _, success, new_book, _, _ = generate_and_collect_with_llm(
            **COMMON_KWARGS, occurence_book=original,
        )
        assert success
        assert len(original) == 1
        assert len(new_book) > 1


# ===================================================================
# 6. generate_and_collect_with_llm — custom LLM functions
# ===================================================================
class TestWithCustomLLM:
    def test_noop_llm_matches_placeholder(self):
        book = OccurenceBook()
        _, s1, b1, d1, _ = generate_and_collect_with_llm(
            **COMMON_KWARGS, occurence_book=book, llm_modify_fn=_fake_llm_noop,
        )
        assert s1
        assert len(b1) > 0

    def test_llm_injector_adds_pair(self):
        _, success, book, diff, llm_result = generate_and_collect_with_llm(
            **COMMON_KWARGS,
            occurence_book=OccurenceBook(),
            llm_modify_fn=_fake_llm_injector,
        )
        assert success
        assert ("LLM_Injected_A", "LLM_Injected_B") in book
        assert book[("LLM_Injected_A", "LLM_Injected_B")] == 5

    def test_llm_remover_drops_pair(self):
        _, success, book, diff, llm_result = generate_and_collect_with_llm(
            **COMMON_KWARGS,
            occurence_book=OccurenceBook(),
            llm_modify_fn=_make_fake_llm_remover(1),
        )
        assert success
        removed_pair = llm_result.removed_transitions[0]
        assert removed_pair not in book

    def test_llm_remover_and_injector(self):
        _, success, book, diff, llm_result = generate_and_collect_with_llm(
            **COMMON_KWARGS,
            occurence_book=OccurenceBook(),
            llm_modify_fn=_fake_llm_remover_and_injector,
        )
        assert success
        assert ("LLM_New_X", "LLM_New_Y") in book
        if llm_result.removed_transitions:
            assert llm_result.removed_transitions[0] not in book


# ===================================================================
# 7. Multi-round accumulation
# ===================================================================
class TestMultiRoundAccumulation:
    def test_three_rounds(self):
        book = OccurenceBook()
        for i in range(3):
            _, success, new_book, _, _ = generate_and_collect_with_llm(
                **COMMON_KWARGS, occurence_book=book,
            )
            if success:
                book = new_book
        assert len(book) > 0

    def test_accumulation_with_injector(self):
        book = OccurenceBook()
        injected_count = 0
        for i in range(3):
            _, success, new_book, _, _ = generate_and_collect_with_llm(
                **COMMON_KWARGS,
                occurence_book=book,
                llm_modify_fn=_fake_llm_injector,
            )
            if success:
                book = new_book
                injected_count += 1
        assert ("LLM_Injected_A", "LLM_Injected_B") in book
        assert book[("LLM_Injected_A", "LLM_Injected_B")] == 5


# ===================================================================
# 8. Full lifecycle with persistence
# ===================================================================
class TestFullLifecycleWithLLM:
    def test_generate_reconcile_persist_reload(self, tmp_path):
        persist = str(tmp_path / "book.json")
        book = OccurenceBook(persist_path=persist)

        # Round 1: generate with injector LLM
        evaluator, success, book, diff, llm_result = generate_and_collect_with_llm(
            **COMMON_KWARGS,
            occurence_book=book,
            llm_modify_fn=_fake_llm_injector,
        )
        assert success
        assert ("LLM_Injected_A", "LLM_Injected_B") in book
        book.save(persist)

        # Reload and verify
        book2 = OccurenceBook.load(persist)
        assert ("LLM_Injected_A", "LLM_Injected_B") in book2
        assert book2[("LLM_Injected_A", "LLM_Injected_B")] == 5

        # Round 2: generate with remover LLM on reloaded book
        _, success2, book3, _, llm_result2 = generate_and_collect_with_llm(
            **COMMON_KWARGS,
            occurence_book=book2,
            llm_modify_fn=_make_fake_llm_remover(1),
        )
        assert success2
        book3.save(persist)

        # Round 3: reload and run with deferred discard
        book4 = OccurenceBook.load(persist)
        _, success3, book5, _, _ = generate_and_collect_with_llm(
            **COMMON_KWARGS, occurence_book=book4,
        )
        assert success3


# ===================================================================
# 9. _build_program_info
# ===================================================================
class TestBuildProgramInfo:
    def test_extracts_fields(self):
        book = OccurenceBook()
        tg = TraceGenerator(
            TensorVariableSchema(), TensorRandomInitializer(),
            TRACE_CONFIG, book,
        )
        tg.prepare_initial_state()
        result, success = generate_program(
            tg, trace_length=3, control_position_candidate=[2],
            enable_if_else=False, enable_coverage=False,
        )
        assert success
        info = _build_program_info(result)
        assert "init_local_str" in info
        assert "init_local_info" in info
        assert "init_implicit_dict" in info
        assert "end_implict_list" in info


# ===================================================================
# Interactive demo — run with: python -m Sgenerator.test.test_llm_trace_generation
# ===================================================================

def _sep(title):
    width = 72
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def interactive_demo():
    """Step-by-step demo that prints every stage of the pipeline.

    All generation steps use if-else branching so the produced programs
    contain ``if … / else …`` control flow.
    """

    # ------------------------------------------------------------------
    _sep("STEP 1: Basic generation with placeholder LLM (no-op, if-else ON)")
    # ------------------------------------------------------------------
    book = OccurenceBook()
    print(f"  Initial OccurenceBook size: {len(book)}")

    evaluator, success, book, diff, llm_result = generate_and_collect_with_llm(
        **DEMO_KWARGS, occurence_book=book,
    )
    print(f"  Success: {success}")
    print(f"  OccurenceBook size after generation: {len(book)}")
    print(f"  Added transitions (diff): {len(diff)} pairs")
    for pair, count in list(diff.items())[:5]:
        print(f"    {pair} -> {count}")
    if len(diff) > 5:
        print(f"    ... and {len(diff) - 5} more")
    print(f"  LLM metadata: {llm_result.llm_metadata}")
    print(f"  LLM removed: {len(llm_result.removed_transitions)}")
    print(f"  LLM added:   {len(llm_result.added_transitions)}")
    print(f"\n  --- Generated program (with if-else) ---")
    print(_indent(llm_result.modified_program))

    # ------------------------------------------------------------------
    _sep("STEP 2: Generation with LLM that INJECTS new transitions")
    # ------------------------------------------------------------------
    book_before = copy.deepcopy(book)
    evaluator, success, book, diff, llm_result = generate_and_collect_with_llm(
        **DEMO_KWARGS,
        occurence_book=book,
        llm_modify_fn=_fake_llm_injector,
    )
    print(f"  Success: {success}")
    print(f"  OccurenceBook size: {len(book_before)} -> {len(book)}")
    injected = ("LLM_Injected_A", "LLM_Injected_B")
    print(f"  Injected pair {injected} present: {injected in book}")
    print(f"  Injected pair count: {book.get(injected, 0)}")
    print(f"  LLM metadata: {llm_result.llm_metadata}")
    print(f"\n  --- Generated program (with if-else) ---")
    print(_indent(llm_result.modified_program))

    # ------------------------------------------------------------------
    _sep("STEP 3: Generation with LLM that REMOVES a transition")
    # ------------------------------------------------------------------
    book_before = copy.deepcopy(book)
    pairs_before = set(book.keys())
    evaluator, success, book, diff, llm_result = generate_and_collect_with_llm(
        **DEMO_KWARGS,
        occurence_book=book,
        llm_modify_fn=_make_fake_llm_remover(1),
    )
    print(f"  Success: {success}")
    print(f"  OccurenceBook size: {len(book_before)} -> {len(book)}")
    if llm_result and llm_result.removed_transitions:
        removed_pair = llm_result.removed_transitions[0]
        print(f"  LLM removed pair: {removed_pair}")
        print(f"  Pair still in book: {removed_pair in book}")
    print(f"  LLM metadata: {llm_result.llm_metadata}")
    print(f"\n  --- Generated program (with if-else) ---")
    print(_indent(llm_result.modified_program))

    # ------------------------------------------------------------------
    _sep("STEP 4: Generation with LLM that REMOVES + INJECTS")
    # ------------------------------------------------------------------
    book_before = copy.deepcopy(book)
    evaluator, success, book, diff, llm_result = generate_and_collect_with_llm(
        **DEMO_KWARGS,
        occurence_book=book,
        llm_modify_fn=_fake_llm_remover_and_injector,
    )
    print(f"  Success: {success}")
    print(f"  OccurenceBook size: {len(book_before)} -> {len(book)}")
    new_pair = ("LLM_New_X", "LLM_New_Y")
    print(f"  Newly injected {new_pair} present: {new_pair in book}, count={book.get(new_pair, 0)}")
    if llm_result.removed_transitions:
        print(f"  Removed {llm_result.removed_transitions[0]}: gone={llm_result.removed_transitions[0] not in book}")
    print(f"  Net diff has {len(diff)} entries")
    print(f"\n  --- Generated program (with if-else) ---")
    print(_indent(llm_result.modified_program))

    # ------------------------------------------------------------------
    _sep("STEP 5: Multi-round accumulation (3 rounds, if-else ON)")
    # ------------------------------------------------------------------
    book = OccurenceBook()
    for rnd in range(3):
        prev_size = len(book)
        _, success, new_book, rnd_diff, rnd_llm = generate_and_collect_with_llm(
            **DEMO_KWARGS, occurence_book=book,
        )
        if success:
            book = new_book
        print(f"  Round {rnd + 1}: success={success}, book {prev_size} -> {len(book)}, "
              f"new pairs={len(rnd_diff) if rnd_diff else 0}")
        if success:
            has_if = "if " in rnd_llm.modified_program
            has_else = "else:" in rnd_llm.modified_program
            print(f"           program has if={has_if}, else={has_else}")
    print(f"  Final OccurenceBook size: {len(book)}")
    print(f"  Sample pairs:")
    for pair, count in list(book.items())[:8]:
        print(f"    {pair}: {count}")
    if len(book) > 8:
        print(f"    ... and {len(book) - 8} more")

    # ------------------------------------------------------------------
    _sep("STEP 6: Persistence round-trip (save / reload / continue)")
    # ------------------------------------------------------------------
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        persist_path = os.path.join(tmpdir, "demo_book.json")

        # Generate + save
        book = OccurenceBook(persist_path=persist_path)
        _, success, book, _, _ = generate_and_collect_with_llm(
            **DEMO_KWARGS,
            occurence_book=book,
            llm_modify_fn=_fake_llm_injector,
        )
        assert success
        book.save(persist_path)
        print(f"  Saved book ({len(book)} pairs) to {persist_path}")

        # Reload
        book2 = OccurenceBook.load(persist_path)
        print(f"  Reloaded book: {len(book2)} pairs")
        print(f"  Injected pair survives reload: {('LLM_Injected_A', 'LLM_Injected_B') in book2}")

        # Continue generation on reloaded book
        prev_size = len(book2)
        _, success, book3, _, _ = generate_and_collect_with_llm(
            **DEMO_KWARGS, occurence_book=book2,
        )
        print(f"  After another round: {prev_size} -> {len(book3)} pairs, success={success}")

    # ------------------------------------------------------------------
    _sep("STEP 7: Reconcile OccurenceBook directly (unit-level demo)")
    # ------------------------------------------------------------------
    book = OccurenceBook()
    book.record(("AlphaOp", "BetaOp"))
    book.record(("AlphaOp", "BetaOp"))
    book.record(("GammaOp", "DeltaOp"))
    print(f"  Book before reconciliation: {dict(book.items())}")

    llm_result = LLMModificationResult(
        modified_program="",
        removed_transitions=[("AlphaOp", "BetaOp")],
        added_transitions={("EpsilonOp", "ZetaOp"): 4},
    )
    removed = reconcile_occurence_book(book, llm_result)
    print(f"  LLM removed: [('AlphaOp', 'BetaOp')]")
    print(f"  LLM injected: {{('EpsilonOp', 'ZetaOp'): 4}}")
    print(f"  Pairs fully removed: {removed}")
    print(f"  Book after reconciliation: {dict(book.items())}")

    # ------------------------------------------------------------------
    _sep("STEP 8: _compute_net_diff demo")
    # ------------------------------------------------------------------
    old_book = OccurenceBook()
    old_book.record(("A", "B"))
    new_book = OccurenceBook()
    new_book.record(("A", "B"))
    new_book.record(("A", "B"))
    new_book.record(("C", "D"))
    diff = _compute_net_diff(old_book, new_book)
    print(f"  Old book: {dict(old_book.items())}")
    print(f"  New book: {dict(new_book.items())}")
    print(f"  Net diff (tolerates removals): {diff}")

    # ------------------------------------------------------------------
    _sep("STEP 9: Show generated program text (if-else via generate_program)")
    # ------------------------------------------------------------------
    book = OccurenceBook()
    tg = TraceGenerator(
        TensorVariableSchema(), TensorRandomInitializer(),
        TRACE_CONFIG, book,
    )
    tg.prepare_initial_state()
    result, success = generate_program(
        tg, trace_length=5, control_position_candidate=[1, 2, 3, 4],
        enable_if_else=True, enable_coverage=False,
    )
    assert success
    program_info = _build_program_info(result)
    print(f"  Program generated successfully: {success}")
    print(f"  Has if-branch: {result['if_trace'] is not None}")
    print(f"  Has else-branch: {result['else_trace'] is not None}")
    if result["condition_info"]:
        print(f"  Condition: {result['condition_info']['if_statement'].strip()}")
    print(f"  Init block:\n{_indent(program_info['init_local_str'])}")
    print(f"  Program body:\n{_indent(result['program'])}")
    print(f"  Transition pairs recorded: {len(result['occurence_book'])}")

    round_diff = (
        result["occurence_book"].end_round()
        if result["occurence_book"]._round_snapshot
        else {}
    )
    llm_result = _fake_llm_injector(
        result["program"], program_info, round_diff,
    )
    print(f"\n  After fake LLM injector:")
    print(f"    Program unchanged: {llm_result.modified_program == result['program']}")
    print(f"    Injected pairs: {llm_result.added_transitions}")

    # ------------------------------------------------------------------
    _sep("ALL STEPS COMPLETED SUCCESSFULLY")
    # ------------------------------------------------------------------
    print()


def _indent(text, prefix="    | "):
    return "\n".join(f"{prefix}{line}" for line in text.strip().split("\n"))


if __name__ == "__main__":
    interactive_demo()
