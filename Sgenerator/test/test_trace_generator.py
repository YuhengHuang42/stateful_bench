"""Integration tests: OccurenceBook with TraceGenerator (using TensorState).

TensorState is used because it has no external service dependencies.
"""
# python -m pytest Sgenerator/test/test_trace_generator.py
import copy
import pytest

from Sgenerator.state import OccurenceBook, TraceGenerator, generate_program, generate_and_collect_test_case
from Sgenerator.tensor_state import TensorVariableSchema, TensorRandomInitializer, TensorEvaluator


TRACE_CONFIG = {
    "init_local_state_num_range": (1, 2),
}


def _make_generator(occurence_book=None):
    if occurence_book is None:
        occurence_book = OccurenceBook()
    schema = TensorVariableSchema()
    random_init = TensorRandomInitializer()
    tg = TraceGenerator(schema, random_init, TRACE_CONFIG, occurence_book)
    tg.prepare_initial_state()
    return tg


# ===================================================================
# 1. TraceGenerator accepts OccurenceBook and works end-to-end
# ===================================================================
class TestTraceGeneratorWithBook:
    def test_accepts_occurence_book(self):
        book = OccurenceBook()
        tg = _make_generator(book)
        assert isinstance(tg.occurence_book, OccurenceBook)

    def test_accepts_raw_dict(self):
        tg = _make_generator({})
        assert isinstance(tg.occurence_book, OccurenceBook)

    def test_generate_trace_records_pairs(self):
        book = OccurenceBook()
        tg = _make_generator(book)
        (trace, trace_str), dup_map, success = tg.generate_trace(
            3, enable_coverage=False
        )
        assert success
        assert len(trace) == 3
        assert len(tg.occurence_book) > 0

    def test_original_book_not_mutated(self):
        book = OccurenceBook()
        book.record(("Sentinel", "Pair"))
        tg = _make_generator(book)
        tg.generate_trace(3, enable_coverage=False)
        assert len(book) == 1  # only the sentinel
        assert len(tg.occurence_book) > 1

    def test_deepcopy_of_generator_has_independent_book(self):
        tg1 = _make_generator()
        tg1.generate_trace(2, enable_coverage=False)
        tg2 = copy.deepcopy(tg1)
        tg2.occurence_book.record(("DEEP", "COPY"))
        assert ("DEEP", "COPY") not in tg1.occurence_book


# ===================================================================
# 2. generate_program with OccurenceBook
# ===================================================================
class TestGenerateProgram:
    def test_no_ifelse(self):
        tg = _make_generator()
        result, success = generate_program(
            tg, trace_length=3,
            control_position_candidate=[2],
            enable_if_else=False,
            enable_coverage=False,
        )
        assert success
        assert result["program"] is not None
        occ = result["occurence_book"]
        assert isinstance(occ, OccurenceBook)
        assert len(occ) > 0

    def test_with_ifelse(self):
        tg = _make_generator()
        result, success = generate_program(
            tg, trace_length=5,
            control_position_candidate=[1, 2, 3, 4],
            enable_if_else=True,
            enable_coverage=False,
        )
        assert success
        occ = result["occurence_book"]
        assert isinstance(occ, OccurenceBook)
        assert len(occ) > 0

    def test_failure_returns_backup_book(self):
        book = OccurenceBook()
        book.record(("Backup", "Test"))
        schema = TensorVariableSchema()
        random_init = TensorRandomInitializer()
        tg = TraceGenerator(schema, random_init, TRACE_CONFIG, book)
        # Patch prepare_initial_state to a no-op so generate_program
        # starts with an empty schema that cannot produce any transitions.
        tg.prepare_initial_state = lambda: None
        result, success = generate_program(
            tg, trace_length=5,
            control_position_candidate=[4],
            enable_if_else=False,
            enable_coverage=False,
        )
        assert not success
        returned_book = result["occurence_book"]
        assert isinstance(returned_book, OccurenceBook)
        assert ("Backup", "Test") in returned_book


# ===================================================================
# 3. Round tracking through generation
# ===================================================================
class TestRoundTrackingIntegration:
    def test_begin_end_round_captures_new_pairs(self):
        book = OccurenceBook()
        book.begin_round()

        tg = TraceGenerator(
            TensorVariableSchema(), TensorRandomInitializer(),
            TRACE_CONFIG, book
        )
        tg.prepare_initial_state()
        (trace, _), _, success = tg.generate_trace(3, enable_coverage=False)
        assert success

        round_diff = tg.occurence_book.get_round_transitions()
        assert len(round_diff) > 0
        for pair, count in round_diff.items():
            assert count >= 1
            assert isinstance(pair, tuple)

    def test_pre_existing_pairs_excluded_from_diff(self):
        book = OccurenceBook()
        book.record(("OldA", "OldB"))
        book.begin_round()

        tg = TraceGenerator(
            TensorVariableSchema(), TensorRandomInitializer(),
            TRACE_CONFIG, book
        )
        tg.prepare_initial_state()
        tg.generate_trace(3, enable_coverage=False)

        diff = tg.occurence_book.get_round_transitions()
        assert ("OldA", "OldB") not in diff


# ===================================================================
# 4. Inject + Discard through generation rounds
# ===================================================================
class TestInjectDiscardIntegration:
    def test_inject_influences_coverage(self):
        book = OccurenceBook()
        tg1 = _make_generator(book)
        tg1.generate_trace(3, enable_coverage=False)
        pairs_after_r1 = dict(tg1.occurence_book.items())

        injected_pair = ("InjectedA", "InjectedB")
        tg1.occurence_book.inject_transition(injected_pair, 100)
        assert tg1.occurence_book[injected_pair] == 100
        assert injected_pair in tg1.occurence_book

    def test_discard_then_generate(self):
        book = OccurenceBook()
        tg = _make_generator(book)
        tg.generate_trace(3, enable_coverage=False)
        initial_len = len(tg.occurence_book)
        assert initial_len > 0

        first_pair = next(iter(tg.occurence_book))
        tg.occurence_book.mark_discarded(first_pair, count=-1)
        removed = tg.occurence_book.apply_discards()
        assert first_pair in removed
        assert len(tg.occurence_book) == initial_len - 1


# ===================================================================
# 5. generate_and_collect_test_case with OccurenceBook
# ===================================================================
class TestGenerateAndCollect:
    def test_returns_occurence_book(self):
        book = OccurenceBook()
        evaluator, success, new_book, occ_diff = generate_and_collect_test_case(
            schema_class=TensorVariableSchema,
            random_init_class=TensorRandomInitializer,
            evaluator_class=TensorEvaluator,
            trace_config=TRACE_CONFIG,
            num_of_apis=3,
            control_position_candidate=[2],
            occurence_book=book,
            enable_if_else=False,
            enable_coverage=False,
        )
        assert success
        assert isinstance(new_book, OccurenceBook)
        assert len(new_book) > 0

    def test_backward_compat_with_raw_dict(self):
        evaluator, success, new_book, occ_diff = generate_and_collect_test_case(
            schema_class=TensorVariableSchema,
            random_init_class=TensorRandomInitializer,
            evaluator_class=TensorEvaluator,
            trace_config=TRACE_CONFIG,
            num_of_apis=3,
            control_position_candidate=[2],
            occurence_book={},
            enable_if_else=False,
            enable_coverage=False,
        )
        assert success
        assert isinstance(new_book, OccurenceBook)

    def test_with_none_default(self):
        evaluator, success, new_book, occ_diff = generate_and_collect_test_case(
            schema_class=TensorVariableSchema,
            random_init_class=TensorRandomInitializer,
            evaluator_class=TensorEvaluator,
            trace_config=TRACE_CONFIG,
            num_of_apis=3,
            control_position_candidate=[2],
            enable_if_else=False,
            enable_coverage=False,
        )
        assert success
        assert isinstance(new_book, OccurenceBook)

    def test_multi_round_accumulation(self):
        book = OccurenceBook()
        for _ in range(3):
            evaluator, success, new_book, occ_diff = generate_and_collect_test_case(
                schema_class=TensorVariableSchema,
                random_init_class=TensorRandomInitializer,
                evaluator_class=TensorEvaluator,
                trace_config=TRACE_CONFIG,
                num_of_apis=3,
                control_position_candidate=[2],
                occurence_book=book,
                enable_if_else=False,
                enable_coverage=False,
            )
            if success:
                book = new_book
        assert len(book) > 0

    def test_occ_diff_is_dict(self):
        evaluator, success, new_book, occ_diff = generate_and_collect_test_case(
            schema_class=TensorVariableSchema,
            random_init_class=TensorRandomInitializer,
            evaluator_class=TensorEvaluator,
            trace_config=TRACE_CONFIG,
            num_of_apis=3,
            control_position_candidate=[2],
            occurence_book=OccurenceBook(),
            enable_if_else=False,
            enable_coverage=False,
        )
        assert success
        assert isinstance(occ_diff, dict)
        assert all(v >= 1 for v in occ_diff.values())


# ===================================================================
# 6. Full lifecycle: generate -> discard -> re-generate
# ===================================================================
class TestFullLifecycle:
    def test_generate_discard_regenerate(self, tmp_path):
        persist = str(tmp_path / "book.json")
        book = OccurenceBook(persist_path=persist)

        # Round 1
        book.begin_round()
        evaluator, success, book, _ = generate_and_collect_test_case(
            schema_class=TensorVariableSchema,
            random_init_class=TensorRandomInitializer,
            evaluator_class=TensorEvaluator,
            trace_config=TRACE_CONFIG,
            num_of_apis=4,
            control_position_candidate=[2],
            occurence_book=book,
            enable_if_else=False,
            enable_coverage=False,
        )
        assert success
        #print("Round 1 diff:", book.get_round_transitions())
        round1_diff = book.end_round()
        assert len(round1_diff) > 0
        book.save(persist)

        # Simulate LLM discarding a pair
        book2 = OccurenceBook.load(persist)
        pair_to_discard = next(iter(book2))
        book2.mark_discarded(pair_to_discard, count=-1)
        book2.save()

        # Round 2
        book3 = OccurenceBook.load(persist)
        removed = book3.apply_discards()
        assert pair_to_discard in removed
        book3.begin_round()
        evaluator, success, book3, _ = generate_and_collect_test_case(
            schema_class=TensorVariableSchema,
            random_init_class=TensorRandomInitializer,
            evaluator_class=TensorEvaluator,
            trace_config=TRACE_CONFIG,
            num_of_apis=4,
            control_position_candidate=[2],
            occurence_book=book3,
            enable_if_else=False,
            enable_coverage=False,
        )
        assert success
        #print("Round 2 diff:", book3.get_round_transitions())
        round2_diff = book3.end_round()
        assert len(round2_diff) > 0
