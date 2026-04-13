"""LLM-assisted test case generation.

Mirrors the workflow in ``trace_generation.py`` but inserts a two-round
LLM review/modification step between program generation and test-case
collection.

Flow
----
1. **Fuzzing engine** generates a program (reuses the existing
   ``generate_program`` / ``TraceGenerator`` pipeline).
2. **API allowlist derivation** — ``_extract_api_names_from_doc``
   parses the API documentation to build a whitelist of valid API
   names.  Only calls matching this allowlist are recognised as APIs
   throughout the pipeline (transition extraction, API counting, and
   post-checks).  This filters out auxiliary calls such as
   ``response.json()`` or ``raise_for_status()``.
3. **Transition extraction** — ``utils.extract_transition_chains`` and
   ``utils.extract_state_transitions`` analyse the generated program
   (filtered by the allowlist) to obtain API transition chains and
   pairwise transition counts.
4. **LLM Round 1 (Analysis)** — the program, transition chains, API
   documentation, and initialisation context are sent to the LLM in a
   structured prompt.  When ``use_api_structured_output`` is enabled
   (default), the chat API uses ``response_format`` JSON schema for
   ``Round1Response``.  The LLM performs sub-intention mapping, an
   unnaturalness audit, and returns a binary modification label
   (0 = keep, 1 = modify).
5. **LLM Round 2 (Modification)** — if the label is 1, a follow-up
   message is appended to the same multi-turn conversation asking the
   LLM to refactor the code under strict constraints (immutable init
   variables, bounded API-count delta, transition-pair preservation).
   Structured output uses ``Round2Response`` schema when enabled.
   The LLM returns the modified program body in the structured fields.
6. **Post-check** — the modified program is validated: the absolute
   change in *documented* API call count (per the allowlist) must not
   exceed ``max_api_modifications`` (derived from a configurable
   ratio).  Indentation issues in the LLM output are auto-repaired
   before AST parsing.
7. **Transition diff** — ``extract_state_transitions`` (with the
   allowlist) is run on both the original and modified programs to
   compute the exact set of removed and added transition pairs.
8. **OccurrenceBook reconciliation** — removed pairs are flagged via
   ``mark_discarded`` and added pairs are injected, keeping the
   coverage book in sync with the actual modified program.
9. **Test-case collection** proceeds with the LLM-modified program.
10. **Error handling** — every stage is wrapped in try/except; on any
    failure the OccurrenceBook rolls back to its pre-round snapshot and
    the iteration is retried.  A configurable consecutive-failure cap
    prevents infinite loops.
11. **Logging** — per-call LLM logs (prompts, raw responses, token
    usage, latency) and aggregate statistics (round-2 call rate, total
    tokens, error counts) are saved to ``<trace_save_path>/llm_logs/``.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import copy
import json
import os
import pickle
import re
import time
from dotenv import load_dotenv
import typer
import yaml
from loguru import logger
from pathlib import Path
from typing import Annotated
from pydantic import BaseModel, Field
import ast as _ast
import textwrap as _textwrap
from openai import OpenAI
from tqdm import tqdm

from Sgenerator.state import (
    OccurenceBook,
    Schema,
    TraceGenerator,
    generate_program,
)
from Sgenerator import utils
from Sgenerator.agent import (
    _extract_json_from_response as _extract_json_object_from_llm_text,
    _flatten_system_messages_for_chat_api,
    openai_json_schema_response_format,
)
from Sgenerator.utils import _extract_api_names_from_doc
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")
DEFAULT_MODIFICATION_RATIO = 0.3
FIX_MAX_TOKENS = 16384
DEFAULT_MAX_CONSECUTIVE_FAILURES = 5
# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LLMModificationResult:
    """Captures what the LLM changed about a fuzzer-generated program.

    Fields
    ------
    modified_program : str
        The program text after LLM rewriting.
    removed_transitions : list of (str, str)
        Transition pairs that the LLM decided to remove from the program.
        Each pair will be fully deleted from the OccurrenceBook via
        ``mark_discarded(pair, count=-1)``.
    added_transitions : dict mapping (str, str) -> int
        New transition pairs that the LLM introduced, with occurrence
        counts.  Injected into the OccurrenceBook via
        ``inject_transitions``.
    modified_program_info : dict or None
        If the LLM also changed the initialisation block (local
        variables, implicit state, load info), provide the updated
        ``program_info`` dict here.  ``None`` means reuse the original.
    llm_metadata : dict
        Free-form metadata about the LLM call (model name, token usage,
        latency, …).
    """

    modified_program: str
    removed_transitions: List[Tuple[str, str]] = field(default_factory=list)
    added_transitions: Dict[Tuple[str, str], int] = field(default_factory=dict)
    modified_program_info: Optional[Dict[str, Any]] = None
    llm_metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# LLM 
# ---------------------------------------------------------------------------

ROUND_1_PROMPT = """Role: You are an expert Software Engineer and Static Analysis Specialist tasked with reviewing code behavior.

Inputs Provided:

{program}: The raw code being analyzed.

{extracted_transitions}: Def-use chains and API sequences bound to specific program states or variables.

{api_documentation}: Reference for the expected behavior of the APIs used.

{example_input}: The data fed into the snippet.

### Task 1: Sub-Intention Mapping
For each API Transition Trace, identify the logical sub-intent of that specific sequence (not the overall program intent).

## Task 2: Unnaturalness Audit
Systematically analyze the code snippet for structural and semantic anomalies. Specifically, flag occurrences of the following four criteria:

Dead State: Variables that are defined but never used or consumed by a terminal API.

Redundancy: Operations or API calls that duplicate effort or yield overwritten/ignored results.

Context Whiplash (Unnatural Order): Instances where the execution flow abruptly abandons a cohesive operation to perform an unrelated task before returning, lacking logical grouping.

Branching Flaws: Ideal branching conditions are missing, overly broad, or syntactically nonsensical given the state variables.

## Task 3: Modification Label
Based on your audit, assign a binary label:

0: The code is human-like, cohesive, and requires no modification.
1: The code exhibits unnaturalness and requires modification.

## Task 4: Overall Program Intent Summary (conditional)
If your Task 3 label is 0, provide a short 1-2 sentence summary of the
overall program intent by composing all sub-intents.
If your Task 3 label is 1, leave this summary empty.
"""

ROUND_2_PROMPT = """
Role: You are an expert Code Refactorer specializing in humanizing machine-generated code.

{max_api_modifications}: The absolute maximum number of API calls you are permitted to add or delete (in-place parameter modifications do not count towards this limit).

Objective: Improve the original code snippet to make it human-readable and logically cohesive, resolving critical issues identified in the Round 1 Analysis.

## Absolute Constraints (CRITICAL):
1. Immutability of Boundaries: You MUST NOT change the names, types, or initial assignments of the initialization variables.
2. Transition Pair Preservation: You should retain ALL API transition pairs identified in the {pair_transition}. Remove these pairs only when necessary.
3. **Minimal Delta**: Do not rewrite the code from scratch. Apply the minimal necessary changes to fix the reported unnaturalness.
4. Syntactic Validity & Indentation: The resulting code MUST have perfectly valid Python syntax. You must ensure rigorous and consistent indentation. Pay strict attention to colons, closed parentheses, and the alignment of if/else and try/except blocks.
5. If there is a RESULT variable as the sink variable, the generated code should also have that variable as the sink variable.
6. Please maintain similar length of the generated code as the original code.
7. Also provide a short 1-2 sentence summary of the final modified program's
   overall intent by composing all sub-intents.

## Permitted Actions (In Order of Preference):
You are restricted to the following operations, prioritizing addition over deletion:

1. Add missing connective logic: (PREFERRED) Introduce a limited number of necessary API calls (available in the document), intermediate variables, or valid parameters to logically bridge the gap.
2. Reorder code blocks: Fix context whiplash and group related operations logically without removing the underlying API calls.
3. Modify branching: Adjust if-conditions to ensure logical execution flow based on state variables.
4. Delete redundant/dead-state APIs: (LAST RESORT) You may delete an API call ONLY when necessary.
"""

ROUND_3_REPAIR_PROMPT = """The following program raised a runtime error. Fix it with minimal changes.

## Program:
{modified_program}

## Error:
{error_message}

## Failing Line (if available):
{line_content}

Return ONLY the fixed program. No explanation, no markdown fences.
"""


# ---------------------------------------------------------------------------
# Pydantic models for structured LLM output
# ---------------------------------------------------------------------------

class TransitionIntent(BaseModel):
    """One entry in the sub-intention mapping (Round 1, Task 1)."""
    transition_trace: List[List[str]] = Field(
        description=(
            "The API transition traces as a list of chains, e.g. "
            "[[API_1, API_2, API_3], [API_4, API_5]]. "
            "Each inner list is a def-use chain / API sequence bound to "
            "a specific variable."
        )
    )
    sub_intent: str = Field(
        description="Brief description of what this specific trace accomplishes."
    )


class UnnaturalnessIssue(BaseModel):
    """A single unnaturalness finding (Round 1, Task 2)."""
    category: str = Field(
        description="One of: Dead State, Redundancy, Context Whiplash, Branching Flaws."
    )
    description: str = Field(
        description="Concrete description of the issue, referencing variable/line."
    )
    affected_transitions: List[List[str]] = Field(
        default_factory=list,
        description=(
            "Transition chains affected by this issue, e.g. "
            "[[API_1, API_2], [API_3, API_4, API_5]]. "
            "Each inner list is a chain of API calls."
        ),
    )


class Round1Response(BaseModel):
    """Structured output for Round 1 (analysis + modification label)."""
    sub_intention_mapping: List[TransitionIntent] = Field(
        description="Task 1: sub-intention for each transition trace."
    )
    unnaturalness_issues: List[UnnaturalnessIssue] = Field(
        default_factory=list,
        description="Task 2: list of unnaturalness findings."
    )
    modification_label: int = Field(
        description="Task 3: 0 = no modification needed, 1 = modification needed."
    )
    reasoning: str = Field(
        default="",
        description="Brief justification for the label."
    )
    overall_program_intent_summary: str = Field(
        default="",
        description=(
            "If modification_label=0, provide a short overall intent summary. "
            "If modification_label=1, keep empty."
        ),
    )


class Round2Response(BaseModel):
    """Structured output for Round 2 (code modification)."""
    modified_program: str = Field(
        description="The modified code snippet (body only, no init block)."
    )
    change_summary: str = Field(
        default="",
        description="Brief summary of what was changed and why."
    )
    overall_program_intent_summary: str = Field(
        default="",
        description=(
            "Short 1-2 sentence summary of the final modified program intent."
        ),
    )


ROUND1_RESPONSE_JSON_KEYS = frozenset(
    {"modification_label", "sub_intention_mapping"}
)
ROUND2_RESPONSE_JSON_KEYS = frozenset({"modified_program"})

STRUCTURED_API_ROUND1_PROMPT_TAIL = (
    "\n\nThe API enforces the response shape: a JSON object with fields "
    "``sub_intention_mapping``, ``unnaturalness_issues``, ``modification_label``, "
    "``reasoning``, and ``overall_program_intent_summary`` (see schema). "
    "Do not echo the schema text in your reply.\n"
)

STRUCTURED_API_ROUND2_PROMPT_TAIL = (
    "\n\nThe API enforces the response shape: a JSON object with fields "
    "``modified_program``, ``change_summary``, and ``overall_program_intent_summary``. "
    "Do not echo the schema text in your reply.\n"
)


def _parse_round1_response_text(
    text: str,
    *,
    use_api_structured_output: bool = True,
) -> Round1Response:
    if use_api_structured_output:
        try:
            return Round1Response.model_validate_json(text.strip())
        except Exception:
            pass
    return Round1Response.model_validate(
        _extract_json_object_from_llm_text(text, ROUND1_RESPONSE_JSON_KEYS)
    )


def _parse_round2_response_text(
    text: str,
    *,
    use_api_structured_output: bool = True,
) -> Round2Response:
    if use_api_structured_output:
        try:
            return Round2Response.model_validate_json(text.strip())
        except Exception:
            pass
    return Round2Response.model_validate(
        _extract_json_object_from_llm_text(text, ROUND2_RESPONSE_JSON_KEYS)
    )


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _format_chains(chains: List[List[str]]) -> str:
    """Format transition chains (from ``utils.extract_transition_chains``)
    into a human-readable string for the LLM prompt."""
    if not chains:
        return "(no transitions)"
    lines = []
    for i, chain in enumerate(chains, 1):
        lines.append(f"  Chain {i}: {' -> '.join(chain)}")
    return "\n".join(lines)


def _summarise_value_for_prompt(val: Any) -> Any:
    """Produce a JSON-safe summary of a value for the LLM prompt.

    For tensor-like objects (anything with ``.shape``), emit only shape
    and dtype instead of the full data.  Everything else is passed
    through to ``json.dumps(..., default=str)``.
    """
    if hasattr(val, "shape"):
        summary: Dict[str, Any] = {"shape": str(val.shape)}
        if hasattr(val, "dtype"):
            summary["dtype"] = str(val.dtype)
        return summary
    if isinstance(val, dict):
        return {k: _summarise_value_for_prompt(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_summarise_value_for_prompt(v) for v in val]
    if hasattr(val, "current_value"):
        return _summarise_value_for_prompt(val.current_value)
    if hasattr(val, "initial_value"):
        return _summarise_value_for_prompt(val.initial_value)
    return val


def _format_program_info_as_example_input(program_info: Dict[str, Any]) -> str:
    """Render program_info into a concise text block for the prompt.

    Tensor data is summarised to shape/dtype only to avoid blowing up
    the prompt size.
    """
    parts = []
    if program_info.get("init_local_str"):
        parts.append(f"Initialization:\n{program_info['init_local_str']}")
    if program_info.get("init_load_str"):
        parts.append(f"Load block:\n{program_info['init_load_str']}")
    #if program_info.get("init_implicit_dict"):
    #    summarised = _summarise_value_for_prompt(
    #        program_info["init_implicit_dict"],
    #    )
    #    parts.append(
    #        f"Implicit state (server-side, do NOT change):\n"
    #        f"{json.dumps(summarised, indent=2, default=str)}"
    #    )
    return "\n\n".join(parts) if parts else "(no example input)"


def _fix_code_indentation(code: str) -> str:
    """Best-effort repair of indentation issues in LLM-generated Python code.

    Common problems:
    - Leading whitespace on every line (the LLM wraps code in a block).
    - Tabs mixed with spaces.
    - Inconsistent indent widths (e.g. 3 spaces instead of 4).
    - Spurious indent on lines that should be at column 0.

    The function tries progressively more aggressive fixes, returning
    as soon as ``ast.parse`` succeeds.
    """
    candidates: List[str] = []

    # Step 0: try as-is
    try:
        _ast.parse(code)
        return code
    except SyntaxError:
        pass

    # Step 1: dedent (common leading whitespace)
    dedented = _textwrap.dedent(code)
    try:
        _ast.parse(dedented)
        return dedented
    except SyntaxError:
        pass

    # Step 2: tabs → 4 spaces
    notab = dedented.expandtabs(4)
    try:
        _ast.parse(notab)
        return notab
    except SyntaxError:
        pass

    lines = notab.splitlines(keepends=True)

    # Collect all non-zero leading-space widths from non-blank lines.
    indent_widths: List[int] = []
    for line in lines:
        stripped = line.lstrip(" ")
        if stripped.strip():
            w = len(line) - len(stripped)
            if w > 0:
                indent_widths.append(w)

    # Step 3: try shifting all indented lines left by each candidate
    # offset.  This fixes the common LLM pattern where every line
    # after the first gets a spurious constant offset.
    offsets_to_try = sorted(set(indent_widths)) if indent_widths else []
    for offset in offsets_to_try:
        shifted = []
        for line in lines:
            stripped = line.lstrip(" ")
            if not stripped.strip():
                shifted.append(stripped)
                continue
            raw = len(line) - len(stripped)
            new_indent = max(0, raw - offset)
            shifted.append(" " * new_indent + stripped)
        candidate = "".join(shifted)
        try:
            _ast.parse(candidate)
            return candidate
        except SyntaxError:
            candidates.append(candidate)

    # Step 4: detect the indent unit and re-map to multiples of 4.
    detected_unit = min(indent_widths) if indent_widths else 4

    def _remap(lines_in: List[str], unit: int) -> str:
        out = []
        for line in lines_in:
            stripped = line.lstrip(" ")
            if not stripped.strip():
                out.append(stripped)
                continue
            raw = len(line) - len(stripped)
            level = round(raw / unit) if unit else 0
            out.append(" " * (4 * level) + stripped)
        return "".join(out)

    for unit in sorted({detected_unit, 2, 3, 4}):
        candidate = _remap(lines, unit)
        try:
            _ast.parse(candidate)
            return candidate
        except SyntaxError:
            candidates.append(candidate)

    # Step 5: strip blank-line indent artefacts and re-dedent
    cleaned = []
    for line in lines:
        cleaned.append("\n" if line.strip() == "" else line)
    final = _textwrap.dedent("".join(cleaned))
    try:
        _ast.parse(final)
        return final
    except SyntaxError:
        candidates.append(final)

    return candidates[0] if candidates else code


_JSON_OUTPUT_INSTRUCTION = (
    "\n\n## Output Format\n"
    "Respond with a single valid JSON object only (no markdown fences, no extra text). Schema:\n"
)


def _should_retry_llm_with_max_completion_tokens(error_text: str) -> bool:
    """Detect 'use max_completion_tokens instead of max_tokens' (OpenAI + compatible gateways)."""
    if not error_text:
        return False
    t = error_text.lower()
    return (
        "max_completion_tokens" in error_text
        and "max_tokens" in error_text
        and ("unsupported" in t or "invalid_request" in t or "not supported" in t)
    )


def _compact_schema(schema: dict) -> str:
    """Produce a minimal JSON schema string, stripping verbose Pydantic metadata."""
    def _strip(obj):
        if isinstance(obj, dict):
            drop = {"title", "default", "examples"}
            return {k: _strip(v) for k, v in obj.items() if k not in drop}
        if isinstance(obj, list):
            return [_strip(v) for v in obj]
        return obj
    return json.dumps(_strip(schema), indent=1)


def _call_llm(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 16384,
    temperature: float = 1.0,
    response_format: Optional[dict] = None,
) -> Tuple[str, dict]:
    """Fire a single chat-completion request and return (text, usage_dict).

    *messages* is the full conversation list (system + user + assistant + …).
    When *response_format* is set (e.g. OpenAI ``json_schema`` mode), the API
    should return a single JSON object; see ``openai_json_schema_response_format``.

    Messages are passed through ``_flatten_system_messages_for_chat_api`` for
    compatibility with gateways that require a user turn after system prompts.

    Raises on HTTP / API errors so the caller can handle retries.
    Raises ``ValueError`` when the response is truncated (``finish_reason
    == "length"``).
    """
    api_messages = _flatten_system_messages_for_chat_api(messages)
    approx_chars = sum(
        len(m.get("content", "") if isinstance(m.get("content"), str)
            else "".join(b.get("text", "") for b in m.get("content", []) if isinstance(b, dict)))
        for m in api_messages
    )
    logger.debug(
        f"LLM call: ~{approx_chars} prompt char, max_tokens={max_tokens}"
    )
    t0 = time.time()
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": api_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    token_limit_param = "max_tokens"
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as e:
        if _should_retry_llm_with_max_completion_tokens(str(e)):
            kwargs.pop("max_tokens", None)
            kwargs["max_completion_tokens"] = max_tokens
            token_limit_param = "max_completion_tokens"
            logger.debug(
                "Retrying chat.completions with max_completion_tokens=… "
                "(API rejected max_tokens for this model)."
            )
            response = client.chat.completions.create(**kwargs)
        else:
            raise
    elapsed = time.time() - t0
    choice = response.choices[0]
    text = choice.message.content or ""
    finish_reason = getattr(choice, "finish_reason", None)
    usage = {
        "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
        "completion_tokens": getattr(response.usage, "completion_tokens", None),
        "total_tokens": getattr(response.usage, "total_tokens", None),
        "latency_s": round(elapsed, 3),
        "finish_reason": finish_reason,
    }
    if finish_reason == "length":
        raise ValueError(
            f"LLM response truncated (finish_reason='length', "
            f"completion_tokens={usage['completion_tokens']}, "
            f"{token_limit_param}={max_tokens}). Increase the token limit in config."
        )
    return text, usage


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences if the model returns them."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped)
    return stripped.strip()


def _fix_program_with_small_llm(
    program: str,
    error_message: str,
    task_context: Dict[str, Any],
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Try to repair code with a small LLM; returns (fixed_program, metadata)."""
    if not task_context.get("enable_fix_llm", False):
        return None

    fix_model = task_context.get("fix_llm_model", "")
    fix_base_url = task_context.get("base_url", "")
    fix_api_key = task_context.get("fix_llm_api_key", "")
    if not fix_model or not fix_base_url:
        logger.warning(
            "enable_fix_llm=True but small LLM config is incomplete "
            "(need fix_llm_model and base_url)."
        )
        return None

    logger.info(
        f"Small LLM fixer enabled: model={fix_model}, base_url={fix_base_url}"
    )
    # Keep explicit print for visibility in long-running generation logs.
    print(
        f"[fix-llm] enabled with model={fix_model}, base_url={fix_base_url}"
    )
    client = OpenAI(base_url=fix_base_url, api_key=fix_api_key)
    fix_max_tokens = task_context.get("fix_llm_max_tokens", FIX_MAX_TOKENS)
    fix_temperature = task_context.get("fix_llm_temperature", 1.0)
    fix_messages = [
        {
            "role": "system",
            "content": (
                "Currently, the code has syntax issues, please repair it with minimal edits. "
                "Return only the fixed code, no markdown fences."
            ),
        },
        {
            "role": "user",
            "content": (
                "The following program failed static transition extraction.\n"
                f"Error: {error_message}\n\n"
                "Please fix syntax/indentation/runtime-structure issues with minimal changes.\n"
                "Program:\n"
                f"{program}"
            ),
        },
    ]
    fixed_text, usage = _call_llm(
        client,
        fix_model,
        messages=fix_messages,
        max_tokens=fix_max_tokens,
        temperature=fix_temperature,
    )
    return _strip_code_fences(fixed_text), usage


def _repair_program_with_llm(
    modified_program: str,
    error_message: str,
    line_content: Optional[str],
    task_context: Dict[str, Any],
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Round 3: ask the same LLM to fix a runtime error in the modified program.

    Returns ``(repaired_program, usage_dict)`` on success, or ``None`` when
    the required config keys are missing.
    """
    base_url = task_context.get("base_url")
    api_key = task_context.get("api_key")
    model = task_context.get("model")
    if not base_url or not model:
        logger.warning("Round 3 repair skipped: base_url or model not set in task_context.")
        return None

    client = OpenAI(base_url=base_url, api_key=api_key)
    max_tokens = task_context.get("round3_max_tokens", task_context.get("round2_max_tokens", 32768))
    temperature = task_context.get("temperature", 1)

    prompt = ROUND_3_REPAIR_PROMPT.format(
        modified_program=modified_program,
        error_message=error_message,
        line_content=line_content if line_content else "(not available)",
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert Python programmer. "
                "Fix the given program so it runs without error. "
                "Return only the fixed code, no markdown fences."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    logger.info(
        f"Round 3 repair: asking {model} to fix runtime error: {error_message!r}, "
        f"line_content={line_content!r}"
    )
    raw_text, usage = _call_llm(client, model, messages=messages, max_tokens=max_tokens, temperature=temperature)
    repaired = _strip_code_fences(raw_text)
    return repaired, usage


def _build_round1_user_content(
    program: str,
    program_info: Dict[str, Any],
    transition_chains: List[List[str]],
    api_documentation: str,
    *,
    use_api_structured_output: bool = True,
) -> str:
    """Assemble the user message for Round 1."""
    example_input = _format_program_info_as_example_input(program_info)
    extracted_transitions = _format_chains(transition_chains)

    filled = ROUND_1_PROMPT.format(
        program=program,
        extracted_transitions=extracted_transitions,
        api_documentation=api_documentation,
        example_input=example_input,
    )
    if use_api_structured_output:
        filled += STRUCTURED_API_ROUND1_PROMPT_TAIL
    else:
        schema_json = _compact_schema(Round1Response.model_json_schema())
        filled += _JSON_OUTPUT_INSTRUCTION + schema_json
    return filled


def _format_pair_transitions(
    round_transitions: Dict[Tuple[str, str], int],
) -> str:
    """Format pairwise transitions from ``get_round_transitions`` for the prompt."""
    if not round_transitions:
        return "(no pair transitions)"
    lines = []
    for (src, tgt), count in round_transitions.items():
        lines.append(f"  ({src}, {tgt})  count={count}")
    return "\n".join(lines)


def _build_round2_followup(
    max_api_modifications: int,
    pair_transition_str: str,
    *,
    use_api_structured_output: bool = True,
) -> str:
    """Build the Round 2 follow-up user message.

    Because this is appended to the same multi-turn conversation that
    already contains the Round 1 exchange, we don't need to re-inject
    the original code, API docs, or initialization — the LLM already
    has them in context.
    """
    filled = ROUND_2_PROMPT.format(
        max_api_modifications=max_api_modifications,
        pair_transition=pair_transition_str,
    )
    if use_api_structured_output:
        return filled + STRUCTURED_API_ROUND2_PROMPT_TAIL
    schema_json = _compact_schema(Round2Response.model_json_schema())
    return filled + _JSON_OUTPUT_INSTRUCTION + schema_json


# ---------------------------------------------------------------------------
# Main LLM entry point
# ---------------------------------------------------------------------------

def llm_modify_program(
    program: str,
    program_info: Dict[str, Any],
    round_transitions: Dict[Tuple[str, str], int],
    task_context: Optional[Dict[str, Any]] = None,
) -> LLMModificationResult:
    """Two-round LLM pipeline: analyse → (optionally) modify.
    
    **This is OPENAI based LLM Pipeline.**

    ``task_context`` should contain at minimum::

        {
            "base_url": "https://api.forge.tensorblock.co/v1",
            "api_key": "...",
            "model": "tensorblock/gemini-3-flash-preview",  # optional
            "api_documentation": "...",                      # optional
            "max_retries": 2,                                # optional
            "temperature": 0.3,                              # optional
            "use_api_structured_output": True,               # optional; json_schema + validate
        }

    All original inputs are deep-copied before use so the LLM logic
    can never accidentally mutate the caller's data.
    """
    # ── Guard against side-effects: deep-copy all inputs ──────────────
    program = copy.deepcopy(program)
    program_info = copy.deepcopy(program_info)
    round_transitions = copy.deepcopy(round_transitions)
    task_context = copy.deepcopy(task_context) if task_context else {}

    # ── Read config ───────────────────────────────────────────────────
    base_url = task_context.get("base_url")
    api_key = task_context.get("api_key")
    model = task_context.get("model")
    api_documentation = task_context.get("api_documentation", "(not provided)")
    max_retries = task_context.get("max_retries", 2)
    temperature = task_context.get("temperature", 1)
    api_allowlist: Optional[Set[str]] = task_context.get("api_allowlist")
    use_api_structured_output = bool(
        task_context.get("use_api_structured_output", True)
    )

    client = OpenAI(base_url=base_url, api_key=api_key)

    rf_round1 = (
        openai_json_schema_response_format(Round1Response, "llm_trace_round1")
        if use_api_structured_output
        else None
    )
    rf_round2 = (
        openai_json_schema_response_format(Round2Response, "llm_trace_round2")
        if use_api_structured_output
        else None
    )

    metadata: Dict[str, Any] = {
        "model": model,
        "input_program_raw": program,
        "use_api_structured_output": use_api_structured_output,
    }

    # ── Extract transition chains from the original program ───────────
    full_program_text = program
    if program_info.get("init_local_str"):
        full_program_text = program_info["init_local_str"] + "\n" + full_program_text
    if program_info.get("init_load_str"):
        full_program_text = program_info["init_load_str"] + "\n" + full_program_text

    transition_chains = utils.extract_transition_chains(
        full_program_text, api_allowlist=api_allowlist,
    )
    original_transitions = utils.extract_state_transitions(
        full_program_text, api_allowlist=api_allowlist,
    )
    metadata["original_chains"] = transition_chains
    metadata["original_transitions"] = {
        f"{k[0]}||{k[1]}": v for k, v in original_transitions.items()
    }

    # ── Build the conversation (multi-turn: shared across rounds) ──────
    system_msg = {
        "role": "system",
        "content": (
            "You are an expert Software Engineer specializing in code analysis "
            "and refactoring. "
            + (
                "The API returns a single JSON object matching the requested schema."
                if use_api_structured_output
                else "Always respond with valid JSON."
            )
        ),
    }
    round1_user_text = _build_round1_user_content(
        program,
        program_info,
        transition_chains,
        api_documentation,
        use_api_structured_output=use_api_structured_output,
    )
    conversation: List[Dict[str, Any]] = [
        system_msg,
        {"role": "user", "content": round1_user_text},
    ]

    # ── Round 1: Analysis ─────────────────────────────────────────────
    round1_max_tokens = task_context.get("round1_max_tokens", 16384)
    round1_parsed: Optional[Round1Response] = None
    round1_raw_text: str = ""
    for attempt in range(1, max_retries + 1):
        try:
            raw_text, usage = _call_llm(
                client,
                model,
                messages=conversation,
                max_tokens=round1_max_tokens,
                temperature=temperature,
                response_format=rf_round1,
            )
            round1_raw_text = raw_text
            metadata["round1_usage"] = usage
            metadata["round1_raw"] = raw_text
            round1_parsed = _parse_round1_response_text(
                raw_text,
                use_api_structured_output=use_api_structured_output,
            )
            metadata["round1_parsed"] = round1_parsed.model_dump()
            break
        except Exception as e:
            logger.warning(f"Round 1 attempt {attempt}/{max_retries} failed: {e}")
            if attempt == max_retries:
                logger.error("Round 1 exhausted retries; returning program unchanged.")
                return LLMModificationResult(
                    modified_program=program,
                    llm_metadata={**metadata, "error": f"round1_failed: {e}"},
                )

    assert round1_parsed is not None
    logger.info(
        f"Round 1 complete: label={round1_parsed.modification_label}, "
        f"issues={len(round1_parsed.unnaturalness_issues)}"
    )

    # ── Short-circuit if label == 0 (no modification needed) ──────────
    if round1_parsed.modification_label == 0:
        logger.info("Round 1 label=0 → no modification, returning original program.")
        metadata["program_intent_summary"] = {
            "source_round": "round1",
            "text": round1_parsed.overall_program_intent_summary.strip(),
        }
        return LLMModificationResult(
            modified_program=program,
            llm_metadata=metadata,
        )

    # ── Round 2: Modification (multi-turn continuation) ───────────────
    # Append Round 1 assistant reply + Round 2 user follow-up to the
    # same conversation so the LLM retains full context from Round 1.
    conversation.append({"role": "assistant", "content": round1_raw_text})

    modification_ratio = task_context.get("modification_ratio", 0.3)
    num_apis_in_program = _count_api_calls(full_program_text, api_allowlist)
    if num_apis_in_program <= 0:
        num_apis_in_program = 1
    max_api_modifications = max(1, int(num_apis_in_program * modification_ratio))

    pair_transition_str = _format_pair_transitions(round_transitions)
    round2_followup = _build_round2_followup(
        max_api_modifications=max_api_modifications,
        pair_transition_str=pair_transition_str,
        use_api_structured_output=use_api_structured_output,
    )
    metadata["max_api_modifications"] = max_api_modifications
    conversation.append({"role": "user", "content": round2_followup})

    round2_max_tokens = task_context.get("round2_max_tokens", 32768)

    round2_parsed: Optional[Round2Response] = None
    for attempt in range(1, max_retries + 1):
        try:
            raw_text, usage = _call_llm(
                client,
                model,
                messages=conversation,
                max_tokens=round2_max_tokens,
                temperature=temperature,
                response_format=rf_round2,
            )
            metadata["round2_usage"] = usage
            metadata["round2_raw"] = raw_text
            round2_parsed = _parse_round2_response_text(
                raw_text,
                use_api_structured_output=use_api_structured_output,
            )
            metadata["round2_parsed"] = round2_parsed.model_dump()
            break
        except Exception as e:
            logger.warning(f"Round 2 attempt {attempt}/{max_retries} failed: {e}")
            if attempt == max_retries:
                logger.error("Round 2 exhausted retries; returning program unchanged.")
                return LLMModificationResult(
                    modified_program=program,
                    llm_metadata={**metadata, "error": f"round2_failed: {e}"},
                )

    assert round2_parsed is not None
    logger.info(f"Round 2 complete: {round2_parsed.change_summary}")
    metadata["program_intent_summary"] = {
        "source_round": "round2",
        "text": round2_parsed.overall_program_intent_summary.strip(),
    }

    # ── Diff transitions via static analysis ──────────────────────────
    modified_program = round2_parsed.modified_program
    modified_program = _fix_code_indentation(modified_program)

    modified_full_text = modified_program
    if program_info.get("init_local_str"):
        modified_full_text = program_info["init_local_str"] + "\n" + modified_full_text
    if program_info.get("init_load_str"):
        modified_full_text = program_info["init_load_str"] + "\n" + modified_full_text
    # modified_full_text = _fix_code_indentation(modified_full_text)

    try:
        modified_transitions = utils.extract_state_transitions(
            modified_full_text, api_allowlist=api_allowlist,
        )
    except Exception as e:
        logger.warning(
            f"Failed to extract transitions from modified program: {e}. "
            "Attempting fix-LLM fallback if enabled."
        )
        modified_transitions = {}
        try:
            fix_result = _fix_program_with_small_llm(
                modified_program, str(e), task_context,
            )
            if fix_result is not None:
                fixed_program, fix_usage = fix_result
                fixed_full_text = fixed_program
                if program_info.get("init_local_str"):
                    fixed_full_text = program_info["init_local_str"] + "\n" + fixed_full_text
                if program_info.get("init_load_str"):
                    fixed_full_text = program_info["init_load_str"] + "\n" + fixed_full_text
                modified_transitions = utils.extract_state_transitions(
                    fixed_full_text, api_allowlist=api_allowlist,
                )
                modified_program = fixed_program
                modified_full_text = fixed_full_text
                metadata["fix_llm"] = {
                    "used": True,
                    "model": task_context.get("fix_llm_model", ""),
                    "usage": fix_usage,
                }
                logger.info("fix-LLM successfully repaired modified program.")
        except Exception as fix_err:
            logger.warning(
                f"fix-LLM repair attempt failed: {fix_err}. Falling back to empty diff."
            )
            metadata["fix_llm"] = {
                "used": True,
                "error": str(fix_err),
                "model": task_context.get("fix_llm_model", ""),
            }

    modified_chains = utils.extract_transition_chains(
        modified_full_text, api_allowlist=api_allowlist,
    )
    metadata["modified_chains"] = modified_chains
    metadata["modified_transitions"] = {
        f"{k[0]}||{k[1]}": v for k, v in modified_transitions.items()
    }

    removed: List[Tuple[str, str]] = [
        pair for pair in original_transitions if pair not in modified_transitions
    ]
    added: Dict[Tuple[str, str], int] = {}
    for pair, count in modified_transitions.items():
        old_count = original_transitions.get(pair, 0)
        if count > old_count:
            added[pair] = count - old_count

    logger.info(
        f"Transition diff: {len(removed)} removed, {len(added)} added."
    )

    original_apis = {api for chain in transition_chains for api in chain} - {"NONE"}
    modified_apis = {api for chain in modified_chains for api in chain} - {"NONE"}
    apis_removed = original_apis - modified_apis
    apis_added = modified_apis - original_apis
    if apis_removed:
        logger.debug(f"APIs removed: {apis_removed}")
    if apis_added:
        logger.debug(f"APIs added: {apis_added}")
    if removed:
        logger.debug(
            "Transition pairs removed: "
            + ", ".join(f"({s} -> {t})" for s, t in removed)
        )
    if added:
        logger.debug(
            "Transition pairs added: "
            + ", ".join(f"({s} -> {t}) x{c}" for (s, t), c in added.items())
        )

    return LLMModificationResult(
        modified_program=modified_program,
        removed_transitions=removed,
        added_transitions=added,
        llm_metadata=metadata,
    )


def _save_llm_logs(
    llm_results_recorder: Dict[int, LLMModificationResult],
    save_dir: str,
) -> None:
    """Persist per-call LLM logs and aggregate statistics.

    Creates ``<save_dir>/llm_logs/`` with:

    *   ``<idx>.json`` — one file per test case containing the model,
        prompts sent, raw responses, token usage, and latency for each
        round.
    *   ``summary.json`` — aggregate statistics across all test cases.
    """
    log_dir = os.path.join(save_dir, "llm_logs")
    os.makedirs(log_dir, exist_ok=True)

    total_round1 = 0
    total_round2 = 0
    total_label_0 = 0
    total_label_1 = 0
    total_errors = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_latency = 0.0

    for idx, result in llm_results_recorder.items():
        meta = result.llm_metadata
        if meta.get("placeholder"):
            continue

        record: Dict[str, Any] = {
            "id": idx,
            "model": meta.get("model"),
            "program_raw": meta.get("input_program_raw", ""),
        }

        # ── Round 1 ──────────────────────────────────────────────────
        has_round1 = "round1_usage" in meta
        if has_round1:
            total_round1 += 1
            r1_usage = meta["round1_usage"]
            record["round1"] = {
                "response_raw": meta.get("round1_raw", ""),
                "response_parsed": meta.get("round1_parsed"),
                "usage": r1_usage,
            }
            total_prompt_tokens += r1_usage.get("prompt_tokens") or 0
            total_completion_tokens += r1_usage.get("completion_tokens") or 0
            total_latency += r1_usage.get("latency_s") or 0.0

            label = (meta.get("round1_parsed") or {}).get("modification_label")
            if label == 0:
                total_label_0 += 1
            elif label == 1:
                total_label_1 += 1

        # ── Round 2 ──────────────────────────────────────────────────
        has_round2 = "round2_usage" in meta
        if has_round2:
            total_round2 += 1
            r2_usage = meta["round2_usage"]
            record["round2"] = {
                "response_raw": meta.get("round2_raw", ""),
                "response_parsed": meta.get("round2_parsed"),
                "usage": r2_usage,
            }
            total_prompt_tokens += r2_usage.get("prompt_tokens") or 0
            total_completion_tokens += r2_usage.get("completion_tokens") or 0
            total_latency += r2_usage.get("latency_s") or 0.0

        # ── API delta ──────────────────────────────────────────────────
        if "api_delta" in meta:
            record["api_delta"] = meta["api_delta"]

        # ── Transition diff ──────────────────────────────────────────
        record["transition_diff"] = {
            "removed": [list(p) for p in result.removed_transitions],
            "added": {f"{p[0]}||{p[1]}": c for p, c in result.added_transitions.items()},
            "original_chains": meta.get("original_chains"),
            "modified_chains": meta.get("modified_chains"),
        }

        if "error" in meta:
            record["error"] = meta["error"]
            total_errors += 1

        log_path = os.path.join(log_dir, f"{idx}.json")
        with open(log_path, "w") as f:
            json.dump(record, f, indent=2, default=str)

    # ── Summary ──────────────────────────────────────────────────────
    summary = {
        "total_test_cases": len(llm_results_recorder),
        "round1_calls": total_round1,
        "round2_calls": total_round2,
        "round2_rate": (
            f"{total_round2 / total_round1 * 100:.1f}%"
            if total_round1 > 0 else "N/A"
        ),
        "label_0_no_modification": total_label_0,
        "label_1_modification_needed": total_label_1,
        "errors": total_errors,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "total_latency_s": round(total_latency, 2),
        "avg_latency_per_call_s": (
            round(total_latency / (total_round1 + total_round2), 2)
            if (total_round1 + total_round2) > 0 else 0
        ),
    }

    summary_path = os.path.join(log_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(
        f"LLM logs saved to {log_dir}/ — "
        f"R1={total_round1}, R2={total_round2} ({summary['round2_rate']}), "
        f"tokens={summary['total_tokens']}, errors={total_errors}"
    )


def _save_program_intent_summaries(
    llm_results_recorder: Dict[int, LLMModificationResult],
    save_dir: str,
) -> None:
    """Persist per-case overall program intent summaries to local files."""
    out_dir = os.path.join(save_dir, "program_intent_summaries")
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for idx, result in llm_results_recorder.items():
        meta = result.llm_metadata or {}
        summary_block = meta.get("program_intent_summary") or {}
        summary_text = (summary_block.get("text") or "").strip()
        if not summary_text:
            continue
        payload = {
            "id": idx,
            "source_round": summary_block.get("source_round"),
            "summary": summary_text,
        }
        save_path = os.path.join(out_dir, f"{idx}.json")
        with open(save_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        saved += 1
    logger.info(f"Program intent summaries saved to {out_dir}/ ({saved} cases).")


def llm_modify_program_mock(
    program: str,
    program_info: Dict[str, Any],
    round_transitions: Dict[Tuple[str, str], int],
    task_context: Optional[Dict[str, Any]] = None,
) -> LLMModificationResult:
    """**PLACEHOLDER** — returns the program unchanged (for testing)."""
    logger.info(
        "[LLM PLACEHOLDER] Returning program unchanged. "
        "Replace llm_modify_fn with llm_modify_program for real LLM calls."
    )
    return LLMModificationResult(
        modified_program=program,
        removed_transitions=[],
        added_transitions={},
        llm_metadata={
            "placeholder": True,
            "program_intent_summary": {
                "source_round": "mock",
                "text": "",
            },
        },
    )


# ---------------------------------------------------------------------------
# OccurrenceBook reconciliation
# ---------------------------------------------------------------------------

def reconcile_occurence_book(
    book: OccurenceBook,
    llm_result: LLMModificationResult,
    apply_immediately: bool = True,
) -> Set[Tuple[str, str]]:
    """Update *book* to reflect the LLM's transition-level changes.

    * Pairs in ``llm_result.removed_transitions`` are flagged for full
      removal (``count=-1``).
    * Pairs in ``llm_result.added_transitions`` are injected with
      *max* semantics.
    * When *apply_immediately* is ``True`` (the default), discards are
      applied at once and the set of fully-removed pairs is returned.
      Pass ``False`` to defer removal to a later ``apply_discards()``
      call (e.g. at the start of the next generation run).

    Returns
    -------
    set of (str, str)c
        Pairs that were fully removed.  Empty when
        *apply_immediately* is ``False``.
    """
    for pair in llm_result.removed_transitions:
        book.mark_discarded(pair, count=-1)

    if llm_result.added_transitions:
        book.inject_transitions(llm_result.added_transitions)

    if apply_immediately:
        return book.apply_discards()
    return set()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_program_info(result: dict) -> dict:
    """Extract the ``program_info`` dict consumed by evaluators."""
    if result["main_trace"] is None:
        implict_list = []
    else:
        implict_list = list(result["main_trace"][1].values())
    if result["if_trace"] is not None:
        for key in result["if_trace"][1]:
            if result["main_trace"] is None or key not in result["main_trace"][1]:
                implict_list.append(result["if_trace"][1][key])
    if result["else_trace"] is not None:
        for key in result["else_trace"][1]:
            if result["main_trace"] is None or key not in result["main_trace"][1]:
                implict_list.append(result["else_trace"][1][key])

    return {
        "init_local_str": result["init_block"][0],
        "init_local_info": result["init_block"][1],
        "init_implicit_dict": result["init_implict_dict"],
        "end_implict_list": implict_list,
        "init_load_str": (
            result["init_load_info"][0] if result["init_load_info"] is not None else None
        ),
        "init_load_info": (
            result["init_load_info"][1] if result["init_load_info"] is not None else None
        ),
    }


@dataclass
class PostCheckResult:
    """Result of the post-LLM-modification sanity checks."""
    passed: bool
    api_count_original: int = 0
    api_count_modified: int = 0
    api_counts_by_type_original: Dict[str, int] = None
    api_counts_by_type_modified: Dict[str, int] = None
    api_delta: int = 0
    max_allowed_delta: int = 0
    details: str = ""


def _count_api_calls(
    code: str,
    api_allowlist: Optional[Set[str]] = None,
) -> int:
    """Count API call nodes in *code*, filtered by *api_allowlist* if given."""
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return 0
    if api_allowlist is not None:
        from Sgenerator.utils import APICallAnalyzer
        analyzer = APICallAnalyzer(api_allowlist=api_allowlist)
        analyzer.visit(tree)
        return len(analyzer.calls_in_order)
    count = 0
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Call):
            count += 1
    return count


def _count_api_calls_by_type(
    code: str,
    api_allowlist: Optional[Set[str]] = None,
) -> Dict[str, int]:
    """Count API calls in *code* grouped by function name.

    Returns a dict mapping each function name to the number of times it
    is called.  When *api_allowlist* is provided only matching calls are
    included.
    
    This is not necessarily the real difference between the original and modified program.
    However, under current situation, we believe this is a good enough approximation especially
    when reasoning LLMs are powerful enough.
    """
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return {}
    counts: Dict[str, int] = {}
    if api_allowlist is not None:
        from Sgenerator.utils import APICallAnalyzer
        analyzer = APICallAnalyzer(api_allowlist=api_allowlist)
        analyzer.visit(tree)
        for _, _, func_name, _ in analyzer.calls_in_order:
            counts[func_name] = counts.get(func_name, 0) + 1
    else:
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Call):
                func_name = _ast.unparse(node.func)
                counts[func_name] = counts.get(func_name, 0) + 1
    return counts


def _contains_result_identifier(code: str) -> bool:
    """Return True if ``RESULT`` appears in the code."""
    try:
        tree = _ast.parse(code)
        return any(
            isinstance(node, _ast.Name) and node.id == "RESULT"
            for node in _ast.walk(tree)
        )
    except SyntaxError:
        return bool(re.search(r"\bRESULT\b", code))


def _has_result_store_target(code: str) -> bool:
    """Return True when ``RESULT`` is assigned in the code."""
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return bool(re.search(r"\bRESULT\s*=", code))

    for node in _ast.walk(tree):
        if isinstance(node, _ast.Name) and node.id == "RESULT" and isinstance(node.ctx, _ast.Store):
            return True
    return False


def _extract_name_from_target(target: _ast.expr) -> Optional[str]:
    """Extract a plain variable name from an assignment target."""
    if isinstance(target, _ast.Name):
        return target.id
    if isinstance(target, (_ast.Tuple, _ast.List)):
        for elt in target.elts:
            name = _extract_name_from_target(elt)
            if name:
                return name
    return None


def _infer_result_expr_from_stmt(stmt: _ast.stmt) -> Optional[_ast.expr]:
    """Best-effort infer an expression to bind to ``RESULT``."""
    if isinstance(stmt, _ast.Assign):
        for target in stmt.targets:
            name = _extract_name_from_target(target)
            if name:
                return _ast.Name(id=name, ctx=_ast.Load())
        return stmt.value
    if isinstance(stmt, _ast.AnnAssign):
        name = _extract_name_from_target(stmt.target)
        if name:
            return _ast.Name(id=name, ctx=_ast.Load())
        if stmt.value is not None:
            return stmt.value
    if isinstance(stmt, _ast.AugAssign):
        name = _extract_name_from_target(stmt.target)
        if name:
            return _ast.Name(id=name, ctx=_ast.Load())
        return stmt.value
    if isinstance(stmt, _ast.Expr):
        return stmt.value
    if isinstance(stmt, _ast.Return) and stmt.value is not None:
        return stmt.value
    if isinstance(stmt, _ast.If):
        return (
            _infer_result_expr_from_block(stmt.orelse)
            or _infer_result_expr_from_block(stmt.body)
        )
    if isinstance(stmt, (_ast.For, _ast.AsyncFor, _ast.While)):
        return (
            _infer_result_expr_from_block(stmt.orelse)
            or _infer_result_expr_from_block(stmt.body)
        )
    if isinstance(stmt, (_ast.With, _ast.AsyncWith)):
        return _infer_result_expr_from_block(stmt.body)
    if isinstance(stmt, _ast.Try):
        for block in (stmt.finalbody, stmt.orelse):
            expr = _infer_result_expr_from_block(block)
            if expr is not None:
                return expr
        for handler in stmt.handlers:
            expr = _infer_result_expr_from_block(handler.body)
            if expr is not None:
                return expr
        return _infer_result_expr_from_block(stmt.body)
    return None


def _infer_result_expr_from_block(stmts: List[_ast.stmt]) -> Optional[_ast.expr]:
    """Walk a block backwards and infer the best sink expression."""
    for stmt in reversed(stmts):
        expr = _infer_result_expr_from_stmt(stmt)
        if expr is not None:
            return expr
    return None


def _ensure_result_sink_variable(original_program: str, modified_program: str) -> str:
    """Inject ``RESULT = ...`` into modified code when original uses RESULT.

    Rules:
    - If original code does not contain ``RESULT``, leave modified code unchanged.
    - If modified code already assigns ``RESULT``, leave it unchanged.
    - Otherwise parse modified code and append a best-effort sink assignment.
    """
    if not _contains_result_identifier(original_program):
        return modified_program
    if _has_result_store_target(modified_program):
        return modified_program

    repaired = _fix_code_indentation(modified_program)
    try:
        tree = _ast.parse(repaired)
    except SyntaxError:
        return repaired.rstrip() + "\nRESULT = None\n"

    inferred = _infer_result_expr_from_block(tree.body)
    if inferred is None:
        inferred = _ast.Constant(value=None)

    assign_node = _ast.Assign(
        targets=[_ast.Name(id="RESULT", ctx=_ast.Store())],
        value=inferred,
    )
    tree.body.append(assign_node)
    _ast.fix_missing_locations(tree)

    try:
        return _ast.unparse(tree)
    except Exception:
        try:
            sink_expr = _ast.unparse(inferred)
        except Exception:
            sink_expr = "None"
        return repaired.rstrip() + f"\nRESULT = {sink_expr}\n"


def post_check_llm_modification(
    original_program: str,
    modified_program: str,
    program_info: Dict[str, Any],
    max_api_modifications: int,
    api_allowlist: Optional[Set[str]] = None,
    graceful_gap: int = 1,
) -> PostCheckResult:
    """Validate that the LLM respected the modification constraints.

    Checks
    ------
    **API type-aware delta** — API calls are counted per function name
    (type).  The delta is ``max(total_additions, total_removals)`` where
    additions/removals are summed across all API types.  This means
    swapping 5 ``post`` calls for 5 ``put`` calls yields a delta of 5,
    not 0.  When *api_allowlist* is provided, only calls matching the
    allowlist are counted.

    Init variable preservation is *not* checked here because the init
    block (``program_info``) is never part of the LLM's output — it is
    preserved by construction and prepended separately by the evaluator.
    If the LLM introduces undeclared variables, test-case collection
    (Step 5) will catch the error.
    
    graceful_gap: int = 1, graceful_gap + max_api_modifications is the total allowed delta.

    Returns a ``PostCheckResult`` with ``passed=True`` if the check
    succeeds.
    """
    result = PostCheckResult(passed=True)

    orig_by_type = _count_api_calls_by_type(original_program, api_allowlist)
    mod_by_type = _count_api_calls_by_type(modified_program, api_allowlist)

    orig_count = sum(orig_by_type.values())
    mod_count = sum(mod_by_type.values())

    result.api_count_original = orig_count
    result.api_count_modified = mod_count
    result.api_counts_by_type_original = orig_by_type
    result.api_counts_by_type_modified = mod_by_type
    result.max_allowed_delta = max_api_modifications

    all_types = set(orig_by_type) | set(mod_by_type)
    additions = sum(
        max(0, mod_by_type.get(t, 0) - orig_by_type.get(t, 0))
        for t in all_types
    )
    removals = sum(
        max(0, orig_by_type.get(t, 0) - mod_by_type.get(t, 0))
        for t in all_types
    )
    delta = max(additions, removals)

    result.api_delta = delta
    if delta > max_api_modifications + graceful_gap:
        result.passed = False
        result.details = (
            f"API type-aware delta {delta} exceeds limit "
            f"{max_api_modifications} "
            f"(original={orig_by_type}, modified={mod_by_type})."
        )

    return result


def _compute_net_diff(
    original_book: OccurenceBook,
    final_book: OccurenceBook,
) -> Dict[Tuple[str, str], int]:
    """Net additions between two books (tolerates LLM-caused removals).

    Unlike ``utils.get_added_changes`` this does **not** assert monotonic
    growth — the LLM may have removed transitions that existed before.
    """
    diff: Dict[Tuple[str, str], int] = {}
    for pair, count in final_book.items():
        old = original_book.get(pair, 0)
        if count > old:
            diff[pair] = count - old
    return diff


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------

def generate_and_collect_with_llm(
    schema_class,
    random_init_class,
    evaluator_class,
    trace_config: Dict[str, Any],
    num_of_apis: int = 5,
    control_position_candidate: Optional[List[int]] = None,
    occurence_book: Union[OccurenceBook, Dict, None] = None,
    evaluation_config: Optional[Dict[str, Any]] = None,
    enable_if_else: bool = True,
    enable_coverage: bool = True,
    llm_modify_fn: Optional[Callable[..., LLMModificationResult]] = None,
    llm_context: Optional[Dict[str, Any]] = None,
):
    """Generate a test case with a fuzzer → LLM modification → collection.

    The signature mirrors ``generate_and_collect_test_case`` with two
    additional keyword arguments (``llm_modify_fn``, ``llm_context``)
    and one extra return value (``llm_result``).

    Parameters
    ----------
    schema_class / random_init_class / evaluator_class
        Exactly as in ``generate_and_collect_test_case``.
    llm_modify_fn : callable, optional
        Function with the same signature as ``llm_modify_program_mock``.
        Defaults to the built-in placeholder that returns the program
        unchanged.
    llm_context : dict, optional
        Opaque context forwarded to *llm_modify_fn* (API keys, model
        name, few-shot examples, …).

    Returns
    -------
    (evaluator, is_success, occurence_book, added_changes, llm_result)

    *   ``evaluator`` — the populated evaluator, or ``None`` on failure.
    *   ``is_success`` — ``True`` when a valid test case was produced.
    *   ``occurence_book`` — the reconciled ``OccurenceBook``.
    *   ``added_changes`` — net new transitions (may be ``None`` on
        failure).
    *   ``llm_result`` — the ``LLMModificationResult`` (``None`` if the
        fuzzing step itself failed before the LLM was invoked).
    """
    if control_position_candidate is None:
        control_position_candidate = [3, 4]
    if evaluation_config is None:
        evaluation_config = {}
    if llm_modify_fn is None:
        llm_modify_fn = llm_modify_program_mock

    # Normalise incoming book
    if occurence_book is None:
        occurence_book = OccurenceBook()
    elif isinstance(occurence_book, dict):
        occurence_book = OccurenceBook.from_raw_dict(occurence_book)

    # ── Snapshot the book so every failure path can roll back ─────────
    # ``original_book`` is the pristine state before this round.  On
    # any failure we return it instead of a half-mutated copy.
    original_book = copy.deepcopy(occurence_book)

    def _fail(
        evaluator=None, llm_result=None, error_msg: str = "",
    ):
        """Return a failure tuple, rolling back to ``original_book``."""
        if error_msg:
            logger.warning(f"[generate_and_collect_with_llm] {error_msg}")
        return evaluator, False, original_book, None, llm_result

    # ── Step 1: Fuzzing engine generates a program ────────────────────
    occurence_book.begin_round()

    try:
        state_schema = schema_class()
        random_init = random_init_class()
        trace_generator = TraceGenerator(
            state_schema, random_init, trace_config, occurence_book,
        )
        trace_generator.prepare_initial_state()

        result, is_success = generate_program(
            trace_generator,
            num_of_apis,
            control_position_candidate,
            enable_if_else,
            enable_coverage,
        )
    except Exception as e:
        return _fail(error_msg=f"Fuzzing engine crashed: {e}")

    if not is_success:
        return _fail(error_msg="generate_program returned is_success=False")

    # Adopt the deep-copied book that generate_program updated.
    occurence_book = result["occurence_book"]

    # round_diff: transitions the fuzzer added in this round.
    round_diff = occurence_book.end_round()

    # ── Step 2: Build program_info ────────────────────────────────────
    program_info = _build_program_info(result)

    # ── Step 2.5: Pre-LLM collect/evaluate gate ───────────────────────
    # If the original generated program cannot even produce/evaluate a
    # valid first test case, skip LLM rewriting for this round.
    pre_llm_evaluator = evaluator_class(evaluation_config)
    try:
        pre_llm_cases = pre_llm_evaluator.collect_test_case(
            program_info=program_info,
            program=result["program"],
        )
        if pre_llm_cases is None:
            return _fail(
                evaluator=pre_llm_evaluator,
                error_msg=(
                    "Pre-LLM collect_test_case failed on original program; "
                    "skip this round."
                ),
            )
        pre_pass_list, pre_test_detail = pre_llm_evaluator.evaluate(result["program"])
        if (not pre_pass_list) or (pre_pass_list[0] is not True):
            return _fail(
                evaluator=pre_llm_evaluator,
                error_msg=(
                    "Pre-LLM evaluation failed on original program: "
                    f"{pre_test_detail[0] if pre_test_detail else 'no detail'}; "
                    "skip this round."
                ),
            )
    except Exception as e:
        return _fail(
            evaluator=pre_llm_evaluator,
            error_msg=f"Pre-LLM collect/evaluate crashed: {e}; skip this round.",
        )

    # ── Step 3: LLM modifies the program ──────────────────────────────
    try:
        llm_result = llm_modify_fn(
            program=result["program"],
            program_info=program_info,
            round_transitions=round_diff,
            task_context=llm_context,
        )
    except Exception as e:
        return _fail(error_msg=f"LLM modify function raised: {e}")

    # ── Step 3.5: Post-check LLM output ──────────────────────────────
    repaired_program = _ensure_result_sink_variable(
        original_program=result["program"],
        modified_program=llm_result.modified_program,
    )
    if repaired_program != llm_result.modified_program:
        logger.info("Auto-repaired modified program by reintroducing RESULT sink variable.")
        llm_result.modified_program = repaired_program

    _allowlist: Optional[Set[str]] = (llm_context or {}).get("api_allowlist")
    max_api_mods = llm_result.llm_metadata.get("max_api_modifications")
    if max_api_mods is None:
        modification_ratio = (llm_context or {}).get("modification_ratio", 0.3)
        full_text = result["program"]
        #_pi = _build_program_info(result)
        #if _pi.get("init_local_str"):
        #    full_text = _pi["init_local_str"] + "\n" + full_text
        #if _pi.get("init_load_str"):
        #    full_text = _pi["init_load_str"] + "\n" + full_text
        n_apis = _count_api_calls(full_text, _allowlist)
        if n_apis <= 0:
            n_apis = 1
        max_api_mods = max(1, int(n_apis * modification_ratio))

    check = post_check_llm_modification(
        original_program=result["program"],
        modified_program=llm_result.modified_program,
        program_info=program_info,
        max_api_modifications=max_api_mods,
        api_allowlist=_allowlist,
    )
    llm_result.llm_metadata["api_delta"] = {
        "delta": check.api_delta,
        "max_allowed": check.max_allowed_delta,
        "original_total": check.api_count_original,
        "modified_total": check.api_count_modified,
        "original_by_type": check.api_counts_by_type_original,
        "modified_by_type": check.api_counts_by_type_modified,
    }
    if not check.passed:
        return _fail(
            llm_result=llm_result,
            error_msg=f"Post-check failed: {check.details}",
        )

    # ── Step 4: Reconcile occurrence book ─────────────────────────────
    try:
        removed = reconcile_occurence_book(occurence_book, llm_result)
        if removed:
            logger.info(
                f"LLM reconciliation removed {len(removed)} transition pairs."
            )
    except Exception as e:
        return _fail(
            llm_result=llm_result,
            error_msg=f"OccurrenceBook reconciliation failed: {e}",
        )

    # ── Step 5: Collect test case from the (modified) program ─────────
    modified_program = llm_result.modified_program
    effective_program_info = (
        llm_result.modified_program_info
        if llm_result.modified_program_info is not None
        else program_info
    )

    def _run_collect_and_evaluate(prog: str) -> Tuple[object, list, list]:
        """Create a fresh evaluator, collect one test case, evaluate, return (evaluator, pass_list, test_detail)."""
        ev = evaluator_class(evaluation_config)
        tc = ev.collect_test_case(program_info=effective_program_info, program=prog)
        if tc is None:
            raise RuntimeError("collect_test_case returned None")
        pl, td = ev.evaluate(prog)
        assert pl[0] is True, (
            f"Evaluation failed on first test case: {td[0] if td else 'no detail'}"
        )
        return ev, pl, td

    evaluator = evaluator_class(evaluation_config)
    try:
        evaluator, pass_list, test_detail = _run_collect_and_evaluate(modified_program)
    except Exception as e:
        first_error = str(e)
        if first_error.startswith("Evaluation failed on first test case"):
            return _fail(
                evaluator=evaluator, llm_result=llm_result,
                error_msg=f"Test case collection/evaluation failed: {first_error}",
            )
        logger.warning(
            f"Test case collection/evaluation failed: {first_error}. "
            "Attempting Round 3 repair…"
        )
        line_content = None
        if getattr(e, "__cause__", None) is not None:
            cause = e.__cause__
            cause_line = getattr(cause, "line_content", None)
            if isinstance(cause_line, str) and cause_line.strip():
                line_content = cause_line
        repair_result = _repair_program_with_llm(
            modified_program=modified_program,
            error_message=first_error,
            line_content=line_content,
            task_context=llm_context or {},
        )
        if repair_result is None:
            return _fail(
                evaluator=evaluator, llm_result=llm_result,
                error_msg=f"Test case collection/evaluation failed: {first_error}",
            )
        repaired_program, repair_usage = repair_result
        repaired_program = _fix_code_indentation(repaired_program)
        llm_result.modified_program = repaired_program
        modified_program = repaired_program

        # Re-run post-check and refresh api_delta against the repaired program.
        repaired_check = post_check_llm_modification(
            original_program=result["program"],
            modified_program=repaired_program,
            program_info=program_info,
            max_api_modifications=max_api_mods,
            api_allowlist=_allowlist,
        )
        llm_result.llm_metadata["api_delta"] = {
            "delta": repaired_check.api_delta,
            "max_allowed": repaired_check.max_allowed_delta,
            "original_total": repaired_check.api_count_original,
            "modified_total": repaired_check.api_count_modified,
            "original_by_type": repaired_check.api_counts_by_type_original,
            "modified_by_type": repaired_check.api_counts_by_type_modified,
        }
        if not repaired_check.passed:
            logger.warning(
                f"Post-check failed on Round 3 repaired program: {repaired_check.details}. "
                "Proceeding since evaluation succeeded."
            )

        llm_result.llm_metadata["round3_repair"] = {
            "used": True,
            "usage": repair_usage,
            "post_check_passed": repaired_check.passed,
        }
        logger.info("Round 3 repair complete; retrying evaluation.")
        try:
            evaluator, pass_list, test_detail = _run_collect_and_evaluate(modified_program)
        except Exception as e2:
            return _fail(
                evaluator=evaluator, llm_result=llm_result,
                error_msg=f"Test case collection/evaluation failed after Round 3 repair: {e2}",
            )

    # ── Step 6: Reversed if-else branch (same as the original) ────────
    if result["condition_info"] is not None:
        try:
            init_local_info_new = copy.deepcopy(result["init_block"][1])
            match_idx = None
            for idx, item in enumerate(init_local_info_new):
                if item[0] == result["condition_info"]["if_condition_name"]:
                    match_idx = idx
                    break
            if match_idx is None:
                raise ValueError(
                    f"Condition variable '{result['condition_info']['if_condition_name']}' "
                    f"not found in init_local_info after LLM modification."
                )
            init_local_info_new[match_idx] = (
                init_local_info_new[match_idx][0],
                Schema.reverse_if_condition(init_local_info_new[match_idx][1]),
            )
            reversed_program_info = {
                "init_local_str": Schema.return_init_local_info(init_local_info_new)[0],
                "init_local_info": init_local_info_new,
                "init_implicit_dict": result["init_implict_dict"],
                "end_implict_list": effective_program_info["end_implict_list"],
                "init_load_str": effective_program_info["init_load_str"],
                "init_load_info": effective_program_info["init_load_info"],
            }
            evaluator.collect_test_case(
                program_info=reversed_program_info,
                program=modified_program,
            )
            pass_list, test_detail = evaluator.evaluate(modified_program)
            assert pass_list[-1] is True, (
                f"Reversed test case evaluation failed: "
                f"{test_detail[-1] if test_detail else 'no detail'}"
            )
        except Exception as e:
            logger.warning(
                f"Error reversing if-condition after LLM modification: {e}, skip."
            )

    added_changes = _compute_net_diff(original_book, occurence_book)
    return evaluator, True, occurence_book, added_changes, llm_result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    pretty_exceptions_short=False,
)

def _ensure_parent_dir(path_like: Any) -> None:
    """Create parent directory for a file path if needed."""
    if not path_like:
        return
    parent_dir = os.path.dirname(str(path_like))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


@app.command()
def main(
    config_file: Annotated[Path, typer.Option()],
    trace_save_path: Annotated[Path, typer.Option()] = None,
    occurence_book_path: Annotated[
        Path,
        typer.Option(help="Path to persistent OccurenceBook JSON."),
    ] = None,
    api_doc_file: Annotated[
        Optional[Path],
        typer.Option(help="Path to API documentation file (json/md)."),
    ] = None,
    simplify_doc: Annotated[
        bool,
        typer.Option(
            "--simplify-doc",
            help=(
                "Use generation_config.api_doc_file_simplified from config "
                "when --api-doc-file is not explicitly provided."
            ),
        ),
    ] = False,
    use_llm: Annotated[
        bool,
        typer.Option(help="Use real LLM calls instead of the mock placeholder."),
    ] = False,
):
    """Generate test cases using the fuzzer + LLM pipeline."""
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    generation_config = config_dict["generation_config"]

    num_of_apis = generation_config["num_of_apis"]
    control_position_candidate = generation_config["control_position_candidate"]
    num_of_tests = generation_config["num_of_tests"]

    if trace_save_path is None:
        trace_save_path = config_dict["env"]["trace_save_path"]
    os.makedirs(str(trace_save_path), exist_ok=True)

    log_file_path = (
        generation_config.get("log_file_path")
        or config_dict.get("env", {}).get("log_file_path")
        or os.environ.get("LOG_FILE_PATH")
    )
    if log_file_path:
        _ensure_parent_dir(log_file_path)

    if occurence_book_path is None:
        occurence_book_path = os.path.join(
            str(trace_save_path), "occurence_book.json"
        )

    occurence_book = OccurenceBook.load(str(occurence_book_path))

    if occurence_book.has_pending_discards():
        logger.info(
            f"Applying {len(occurence_book.pending_discards)} pending "
            "discards from previous LLM review."
        )
        removed = occurence_book.apply_discards()
        logger.info(f"Removed {len(removed)} fully-zeroed transition pairs.")

    # -- Load API documentation if provided --
    api_documentation = None
    if api_doc_file is None:
        doc_key = "api_doc_file_simplified" if simplify_doc else "api_doc_file"
        if doc_key in config_dict["generation_config"]:
            api_doc_file = os.path.join(
                PROJECT_ROOT,
                config_dict["generation_config"][doc_key],
            )
        elif simplify_doc and "api_doc_file" in config_dict["generation_config"]:
            logger.warning(
                "simplify_doc=True but api_doc_file_simplified is missing; "
                "falling back to api_doc_file."
            )
            api_doc_file = os.path.join(
                PROJECT_ROOT,
                config_dict["generation_config"]["api_doc_file"],
            )
    if api_doc_file is not None:
        with open(api_doc_file, "r") as f:
            api_documentation = f.read()

    # -- Build llm_context from config + CLI overrides --
    llm_config = {}
    assert api_documentation is not None
    llm_config.setdefault("api_documentation", api_documentation)
    base_url = os.environ.get("BASE_URL", None)
    assert base_url is not None
    llm_config["base_url"] = base_url
    model = os.environ.get("MODEL", None)
    assert model is not None
    llm_config["model"] = model
    api_key = os.environ.get("API_KEY", "")
    assert api_key is not None
    llm_config["api_key"] = api_key
    llm_config["enable_fix_llm"] = bool(
        os.environ.get("FIX_LLM_MODEL")
    )
    llm_config["fix_llm_model"] = os.environ.get("FIX_LLM_MODEL", "")
    llm_config["fix_llm_api_key"] = os.environ.get("FIX_LLM_API_KEY", api_key)
    llm_modify_fn = llm_modify_program if use_llm else llm_modify_program_mock
    use_api_structured_output = generation_config.get("use_api_structured_output", True)
    llm_config["use_api_structured_output"] = use_api_structured_output
    if use_llm:
        logger.info(
            f"Using real LLM: model={llm_config['model']}, "
            f"base_url={llm_config['base_url']}, "
            f"use_api_structured_output={llm_config['use_api_structured_output']}"
        )
        if llm_config["enable_fix_llm"]:
            msg = (
                f"Fix LLM enabled: model={llm_config['fix_llm_model']}, "
                f"base_url={llm_config['base_url']}"
            )
            logger.info(msg)
            print(msg)
    if "modification_ratio" in config_dict["generation_config"]:
        llm_config["modification_ratio"] = config_dict["generation_config"]["modification_ratio"]
    else:
        llm_config["modification_ratio"] = DEFAULT_MODIFICATION_RATIO
        logger.warning("Modification ratio not set, using default value {DEFAULT_MODIFICATION_RATIO}.")

    uao_env = os.environ.get("LLM_TRACE_USE_API_STRUCTURED_OUTPUT")
    if uao_env is not None and str(uao_env).strip():
        use_api_structured_output = str(uao_env).strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
    else:
        use_api_structured_output = bool(
            generation_config.get("use_api_structured_output", True),
        )
    llm_config["use_api_structured_output"] = use_api_structured_output

    # -- Derive API allowlist from the documentation --
    api_allowlist = _extract_api_names_from_doc(api_documentation)
    if api_allowlist:
        llm_config["api_allowlist"] = api_allowlist
        logger.info(f"API allowlist ({len(api_allowlist)} entries): {sorted(api_allowlist)}")
    else:
        logger.warning(
            "Could not extract any API names from the documentation; "
            "falling back to blacklist-based filtering."
        )

    # -- Task-specific wiring --
    evaluation_config: Dict[str, Any] = {}
    if config_dict["task"] == "session":
        from Sgenerator.session_state import (
            SessionEvaluator,
            SessionRandomInitializer,
            SessionVariableSchema,
        )
        evaluation_config["base_url"] = config_dict["env"]["base_url"]
        schema_class = SessionVariableSchema
        random_init_class = SessionRandomInitializer
        evaluator_class = SessionEvaluator
    elif config_dict["task"] == "tensor":
        from Sgenerator.tensor_state import (
            TensorEvaluator,
            TensorRandomInitializer,
            TensorVariableSchema,
        )
        schema_class = TensorVariableSchema
        random_init_class = TensorRandomInitializer
        evaluator_class = TensorEvaluator
    elif config_dict["task"] == "voice":
        from Sgenerator.voice_state import (
            VoiceEvaluator,
            VoiceRandomInitializer,
            VoiceVariableSchema,
        )
        schema_class = VoiceVariableSchema
        random_init_class = VoiceRandomInitializer
        evaluator_class = VoiceEvaluator
    else:
        raise ValueError(f"Task {config_dict['task']} is not supported.")

    evaluator_book: Dict[int, Any] = {}
    occ_book_diff_recorder: Dict[int, Any] = {}
    llm_results_recorder: Dict[int, LLMModificationResult] = {}

    enable_coverage = generation_config.get("enable_coverage", True)
    if not enable_coverage:
        logger.info("Coverage-guided trace generation is disabled.")

    max_consecutive_failures = generation_config.get(
        "max_consecutive_failures", DEFAULT_MAX_CONSECUTIVE_FAILURES,
    )
    idx = 0
    consecutive_failures = 0
    total_failures = 0
    pbar = tqdm(total=num_of_tests, desc="Generating", unit="test")
    while idx < num_of_tests:
        evaluator, is_success, new_book, occ_diff, llm_result = (
            generate_and_collect_with_llm(
                schema_class=schema_class,
                random_init_class=random_init_class,
                evaluator_class=evaluator_class,
                trace_config=generation_config["trace_config"],
                evaluation_config=evaluation_config,
                num_of_apis=num_of_apis,
                control_position_candidate=control_position_candidate,
                occurence_book=occurence_book,
                enable_coverage=enable_coverage,
                llm_modify_fn=llm_modify_fn,
                llm_context=llm_config,
            )
        )
        if is_success:
            occurence_book = new_book
            occ_book_diff_recorder[idx] = occ_diff
            evaluator_book[idx] = evaluator
            llm_results_recorder[idx] = llm_result
            idx += 1
            consecutive_failures = 0
            pbar.update(1)
        else:
            consecutive_failures += 1
            total_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                logger.error(
                    f"Aborting: {consecutive_failures} consecutive failures "
                    f"(limit={max_consecutive_failures}). "
                    f"Generated {idx}/{num_of_tests} test cases."
                )
                break
        pbar.set_postfix(ok=idx, fail=total_failures)
    pbar.close()
    logger.info(
        f"Generation complete: {idx}/{num_of_tests} succeeded, "
        f"{total_failures} total failures."
    )

    # -- Persist results --
    for eidx in evaluator_book:
        evaluator_save_path = os.path.join(
            str(trace_save_path), f"evaluator_{eidx}.json"
        )
        evaluator_book[eidx].store(evaluator_save_path)

    occurence_book.save(str(occurence_book_path))
    logger.info(f"OccurenceBook saved to {occurence_book_path}")

    metadata_save_path = os.path.join(str(trace_save_path), "metadata.pkl")
    with open(metadata_save_path, "wb") as f:
        pickle.dump(
            {
                "occurence_book": occurence_book.to_dict(),
                "config": config_dict,
                "occ_book_diff_recorder": occ_book_diff_recorder,
                "llm_results": {
                    k: {
                        "removed_transitions": v.removed_transitions,
                        "added_transitions": {
                            str(p): c for p, c in v.added_transitions.items()
                        },
                        "llm_metadata": v.llm_metadata,
                    }
                    for k, v in llm_results_recorder.items()
                },
            },
            f,
        )

    # -- Save LLM call logs and statistics --
    _save_llm_logs(llm_results_recorder, str(trace_save_path))
    _save_program_intent_summaries(llm_results_recorder, str(trace_save_path))


if __name__ == "__main__":
    app()
