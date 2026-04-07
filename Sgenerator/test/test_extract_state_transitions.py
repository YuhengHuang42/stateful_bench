"""Stress test: extract_state_transitions vs OccurenceBook.get_round_transitions.

Verifies that the static-analysis tool recovers the same state transitions
that the program generator records at generation time, when the generator
runs exactly once from an empty OccurenceBook.

Run:
    python -m pytest Sgenerator/test/test_extract_state_transitions.py -v
    python -m pytest Sgenerator/test/test_extract_state_transitions.py -v -k stress
"""
import copy
import random
import pytest
from collections import Counter

from Sgenerator.state import (
    OccurenceBook,
    TraceGenerator,
    generate_program,
    generate_and_collect_test_case,
)
from Sgenerator.tensor_state import (
    TensorVariableSchema,
    TensorRandomInitializer,
    TensorEvaluator,
)
from Sgenerator.utils import extract_state_transitions, APICallAnalyzer

# -----------------------------------------------------------------
# Schema-name  →  AST-level qualified function name(s)
# -----------------------------------------------------------------
TENSOR_SCHEMA_TO_AST = {
    "PermuteTransition": "torch.permute",
    "SplitTransition":   "torch.split",
    "CatTransition":     "torch.cat",
    "TransposeTransition": "torch.transpose",
    "Conv2dTransition":  "torch.nn.functional.conv2d",
    "LinearTransition":  "torch.nn.functional.linear",
}

AST_TO_TENSOR_SCHEMA = {v: k for k, v in TENSOR_SCHEMA_TO_AST.items()}


def _map_pair_schema_to_ast(pair):
    """Convert a single (schema_a, schema_b) pair to (ast_a, ast_b).

    ``"NONE"`` stays as ``"NONE"`` (initial sentinel).
    """
    a, b = pair
    a_mapped = TENSOR_SCHEMA_TO_AST.get(a, a)
    b_mapped = TENSOR_SCHEMA_TO_AST.get(b, b)
    return (a_mapped, b_mapped)


def _map_pair_ast_to_schema(pair):
    """Convert a single (ast_a, ast_b) pair to (schema_a, schema_b)."""
    a, b = pair
    a_mapped = AST_TO_TENSOR_SCHEMA.get(a, a)
    b_mapped = AST_TO_TENSOR_SCHEMA.get(b, b)
    return (a_mapped, b_mapped)


TRACE_CONFIG = {
    "init_local_state_num_range": (1, 2),
}


def _generate_once(num_apis=5, enable_if_else=False, seed=None):
    """Run the generator once and return (program_code, round_transitions).

    ``round_transitions`` comes from ``OccurenceBook.get_round_transitions()``,
    and ``program_code`` is the full program string (init block + body) that
    can be fed to ``extract_state_transitions``.
    """
    if seed is not None:
        random.seed(seed)

    book = OccurenceBook()
    book.begin_round()

    schema = TensorVariableSchema()
    rand_init = TensorRandomInitializer()
    tg = TraceGenerator(schema, rand_init, TRACE_CONFIG, book)
    tg.prepare_initial_state()

    if enable_if_else:
        control_pos = list(range(1, num_apis))
    else:
        control_pos = [num_apis - 1]

    result, success = generate_program(
        tg,
        trace_length=num_apis,
        control_position_candidate=control_pos,
        enable_if_else=enable_if_else,
        enable_coverage=False,
    )
    if not success:
        return None, None, None

    occ_book = result["occurence_book"]
    round_transitions = occ_book.get_round_transitions()

    init_block = result["init_block"][0] if result["init_block"] else ""
    program_body = result["program"] if result["program"] else ""

    load_info = result.get("init_load_info")
    load_str = ""
    if load_info is not None and load_info[0] is not None:
        load_str = load_info[0]

    full_code = "import torch\n" + load_str + init_block + program_body
    result_str_addition = ""
    if "RESULT" in program_body:
        pass
    return full_code, round_transitions, result


# =================================================================
# Helpers to compare the two transition dicts
# =================================================================

def _compare_transitions(round_transitions, static_transitions, full_code):
    """Compare OccurenceBook round transitions with static analysis transitions.

    Returns (matched_pairs, only_in_book, only_in_static, details_str).

    The comparison maps schema-level names to AST-level names so they can be
    compared on common ground.  "NONE" is kept as-is (both sides produce it).
    """
    book_as_ast = {}
    for pair, count in round_transitions.items():
        mapped = _map_pair_schema_to_ast(pair)
        book_as_ast[mapped] = book_as_ast.get(mapped, 0) + count

    static_as_schema = {}
    for pair, count in static_transitions.items():
        mapped = _map_pair_ast_to_schema(pair)
        static_as_schema[mapped] = static_as_schema.get(mapped, 0) + count

    book_pair_set = set(book_as_ast.keys())
    static_pair_set = set(static_transitions.keys())

    matched = book_pair_set & static_pair_set
    only_in_book = book_pair_set - static_pair_set
    only_in_static = static_pair_set - book_pair_set

    details = []
    details.append(f"Matched pairs ({len(matched)}): {matched}")
    details.append(f"Only in book ({len(only_in_book)}): {only_in_book}")
    details.append(f"Only in static ({len(only_in_static)}): {only_in_static}")
    details.append(f"\nBook (mapped to AST): {book_as_ast}")
    details.append(f"Static: {static_transitions}")
    details.append(f"\nFull code:\n{full_code}")
    detail_str = "\n".join(details)

    return matched, only_in_book, only_in_static, detail_str


# =================================================================
# 1. Basic: single generation, no if-else
# =================================================================
class TestBasicComparison:
    @pytest.mark.parametrize("num_apis", [3, 4, 5, 6])
    def test_single_generation(self, num_apis):
        full_code, round_trans, result = _generate_once(
            num_apis=num_apis, enable_if_else=False
        )
        if full_code is None:
            pytest.skip("Generation failed (can happen with unlucky RNG)")

        static_trans = extract_state_transitions(full_code)

        matched, only_book, only_static, details = _compare_transitions(
            round_trans, static_trans, full_code
        )
        # Every pair in the book (including NONE) should be found by static analysis
        assert len(only_book) == 0, (
            f"Static analysis missed transitions from OccurenceBook:\n{details}"
        )


# =================================================================
# 2. Stress test: many random seeds, no if-else
# =================================================================
class TestStressNoIfElse:
    NUM_ITERATIONS = 5

    def test_stress_no_ifelse(self):
        missed_total = 0
        extra_total = 0
        failures = []

        for i in range(self.NUM_ITERATIONS):
            full_code, round_trans, result = _generate_once(
                num_apis=random.randint(3, 7),
                enable_if_else=False,
                seed=i * 137,
            )
            if full_code is None:
                continue

            static_trans = extract_state_transitions(full_code)
            matched, only_book, only_static, details = _compare_transitions(
                round_trans, static_trans, full_code
            )
            if len(only_book) > 0:
                missed_total += len(only_book)
                failures.append(f"seed={i * 137}: missed {only_book}")
            extra_total += len(only_static)

        summary = (
            f"Over {self.NUM_ITERATIONS} iterations: "
            f"{missed_total} missed pairs, {extra_total} extra pairs in static. "
            f"Failures:\n" + "\n".join(failures[:10])
        )
        assert missed_total == 0, summary


# =================================================================
# 3. Stress test: with if-else branches
# =================================================================
class TestStressWithIfElse:
    NUM_ITERATIONS = 5

    def test_stress_with_ifelse(self):
        missed_total = 0
        extra_total = 0
        failures = []

        for i in range(self.NUM_ITERATIONS):
            full_code, round_trans, result = _generate_once(
                num_apis=random.randint(4, 7),
                enable_if_else=True,
                seed=i * 251,
            )
            if full_code is None:
                continue

            static_trans = extract_state_transitions(full_code)
            matched, only_book, only_static, details = _compare_transitions(
                round_trans, static_trans, full_code
            )
            if len(only_book) > 0:
                missed_total += len(only_book)
                failures.append(f"seed={i * 251}: missed {only_book}")
            extra_total += len(only_static)

        summary = (
            f"Over {self.NUM_ITERATIONS} iterations: "
            f"{missed_total} missed pairs, {extra_total} extra pairs in static. "
            f"Failures:\n" + "\n".join(failures[:10])
        )
        assert missed_total == 0, summary


# =================================================================
# 4. Verify the return format matches OccurenceBook.get_round_transitions
# =================================================================
class TestReturnFormat:
    def test_return_type_is_dict_of_tuple_str_str_to_int(self):
        code = "import torch\nx = torch.randn(3, 4)\ny = torch.permute(x, (1, 0))\nz = torch.transpose(y, 0, 1)\n"
        result = extract_state_transitions(code)
        assert isinstance(result, dict)
        for key, val in result.items():
            assert isinstance(key, tuple), f"Key {key!r} is not a tuple"
            assert len(key) == 2, f"Key {key!r} does not have 2 elements"
            assert isinstance(key[0], str), f"key[0] {key[0]!r} is not str"
            assert isinstance(key[1], str), f"key[1] {key[1]!r} is not str"
            assert isinstance(val, int), f"Value {val!r} is not int"
            assert val >= 1, f"Value {val!r} < 1"

    def test_known_chain(self):
        code = """\
import torch
x = torch.randn(3, 4)
y = torch.permute(x, (1, 0))
z = torch.transpose(y, 0, 1)
"""
        result = extract_state_transitions(code)
        assert ("torch.permute", "torch.transpose") in result

    def test_no_transitions_for_independent_calls(self):
        code = """\
import torch
a = torch.randn(2, 3)
b = torch.randn(4, 5)
"""
        result = extract_state_transitions(code)
        assert ("torch.randn", "torch.randn") not in result


# =================================================================
# 5. Verbose demo (run with: python -m Sgenerator.test.test_extract_state_transitions)
# =================================================================
def _run_demo():
    import textwrap

    print("=" * 70)
    print("DEMO: extract_state_transitions vs OccurenceBook.get_round_transitions")
    print("=" * 70)

    n_pass = 0
    n_fail = 0
    n_skip = 0

    for i in range(3):
        seed = i * 42 + 7
        full_code, round_trans, result = _generate_once(
            num_apis=random.randint(3, 6),
            enable_if_else=(i % 3 == 0),
            seed=seed,
        )
        if full_code is None:
            n_skip += 1
            continue

        static_trans = extract_state_transitions(full_code)
        matched, only_book, only_static, details = _compare_transitions(
            round_trans, static_trans, full_code
        )

        status = "PASS" if len(only_book) == 0 else "FAIL"
        if status == "PASS":
            n_pass += 1
        else:
            n_fail += 1

        print(f"\n--- Iteration {i} (seed={seed}) [{status}] ---")
        print(f"  Code lines: {len(full_code.splitlines())}")
        print(f"  Book pairs: { {k: v for k, v in round_trans.items()} }")
        print(f"  Static pairs:           {static_trans}")
        print(f"  Matched: {len(matched)}  OnlyBook: {len(only_book)}  OnlyStatic: {len(only_static)}")
        if len(only_book) > 0:
            print(f"  *** MISSED: {only_book}")
        if len(only_static) > 0:
            print(f"  (extra from static): {only_static}")
        if status == "FAIL":
            print(f"\n  Full code:\n{textwrap.indent(full_code, '    ')}")

    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {n_pass} passed, {n_fail} failed, {n_skip} skipped out of 30")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    _run_demo()
