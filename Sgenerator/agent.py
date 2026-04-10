from enum import Enum
import os
import json
import re
import time
from typing import AbstractSet, Any, Dict, List, Literal, Optional, Tuple

from loguru import logger
from pydantic import BaseModel, Field

from Sgenerator.utils import write_jsonl, generate_jsonl_for_openai, submit_batch_request_openai


# ---------------------------------------------------------------------------
# Structured LLM output (same pattern as llm_trace_generation.py)
# ---------------------------------------------------------------------------

_JSON_OUTPUT_INSTRUCTION = (
    "\n\n## Output Format\n"
    "Respond with a single valid JSON object only (no markdown fences, no extra text). Schema:\n"
)


def _compact_schema(schema: dict) -> str:
    """Strip verbose Pydantic metadata for shorter prompt text."""

    def _strip(obj: Any) -> Any:
        if isinstance(obj, dict):
            drop = {"title", "default", "examples"}
            return {k: _strip(v) for k, v in obj.items() if k not in drop}
        if isinstance(obj, list):
            return [_strip(v) for v in obj]
        return obj

    return json.dumps(_strip(schema), indent=2)


def _iter_balanced_top_level_json_objects(text: str) -> List[dict]:
    """Parse every top-level ``{ ... }`` segment that is valid JSON (dict).

    Greedy ``\\{.*\\}`` is unsafe: it often grabs an inner schema fragment
    (e.g. ``{"description": "...", "type": "object"}``) instead of the full
    model payload. Bracket matching avoids that when the payload is the
    outermost object.

    Does not treat braces inside JSON strings specially; good enough for
    typical structured LLM outputs.
    """
    found: List[dict] = []
    depth = 0
    start: Optional[int] = None
    for i, c in enumerate(text):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    chunk = text[start : i + 1]
                    try:
                        val = json.loads(chunk)
                    except json.JSONDecodeError:
                        pass
                    else:
                        if isinstance(val, dict):
                            found.append(val)
                    start = None
    return found


def _json_object_candidates(text: str) -> List[dict]:
    """Collect distinct dict candidates from whole text, fenced blocks, and segments."""
    seen: set[str] = set()
    out: List[dict] = []

    def _add(d: dict) -> None:
        sig = json.dumps(d, sort_keys=True, ensure_ascii=False, default=str)
        if sig not in seen:
            seen.add(sig)
            out.append(d)

    raw = text.strip()
    if raw:
        try:
            v = json.loads(raw)
            if isinstance(v, dict):
                _add(v)
        except json.JSONDecodeError:
            pass

    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            v = json.loads(m.group(1).strip())
            if isinstance(v, dict):
                _add(v)
        except json.JSONDecodeError:
            pass

    for obj in _iter_balanced_top_level_json_objects(text):
        _add(obj)

    return out


def _extract_json_from_response(
    text: str,
    required_keys: Optional[AbstractSet[str]] = None,
) -> dict:
    """Pick the best JSON object from messy LLM text (fallback path).

    When ``use_api_structured_output`` is True, the API should return a single
    JSON body and we parse it with ``model_validate_json`` first; this helper
    is only for providers that do *not* enforce schema (prompt-only JSON,
    markdown fences, or echoed schema fragments).

    If *required_keys* is set, prefer a dict that contains all of them
    (largest such dict wins). Otherwise return the largest dict by serialized
    size (heuristic for “full” response vs. nested schema snippets).
    """
    candidates = _json_object_candidates(text)
    if not candidates:
        raise ValueError(f"Could not extract JSON from LLM response:\n{text[:500]}")

    def _size(d: dict) -> int:
        try:
            return len(json.dumps(d, ensure_ascii=False, default=str))
        except TypeError:
            return len(str(d))

    if required_keys:
        good = [d for d in candidates if required_keys <= d.keys()]
        if good:
            good.sort(key=_size, reverse=True)
            return good[0]

    candidates.sort(key=_size, reverse=True)
    return candidates[0]


GENERATOR_RESPONSE_JSON_KEYS = frozenset(
    {"user_variable_definitions", "task_instructions"}
)
EVALUATOR_RESPONSE_JSON_KEYS = frozenset({"verdict"})


def openai_json_schema_response_format(
    model_cls: type[BaseModel],
    name: str,
    *,
    strict: bool = False,
) -> dict:
    """``response_format`` payload for ``chat.completions.create`` / batch body.

    Uses OpenAI ``json_schema`` mode. ``strict=True`` requires a schema with
    ``additionalProperties: false`` everywhere; Pydantic JSON Schema may not
    satisfy that on all models, so default is ``False`` for portability.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": strict,
            "schema": model_cls.model_json_schema(),
        },
    }


# When the API enforces JSON schema, do not paste the schema into the prompt
# (models often echo it and break naive JSON extraction).
STRUCTURED_API_GENERATOR_PROMPT_TAIL = (
    "\n\nThe API enforces the response shape: a JSON object with string fields "
    "``user_variable_definitions`` and ``task_instructions`` only.\n"
)
STRUCTURED_API_EVALUATOR_PROMPT_TAIL = (
    "\n\nThe API enforces the response shape: a JSON object with string field "
    "``verdict`` (one of: ok, needs_revision, impossible) and optional string "
    "fields ``diagnosis`` and ``suggestions``.\n"
)


class TranslationGeneratorResponse(BaseModel):
    """Structured generator (translation) output."""

    user_variable_definitions: str = Field(
        description=(
            "Natural-language definitions of user variables/constants from the init block "
            "(types, structure); refer to names only, not concrete values."
        )
    )
    task_instructions: str = Field(
        description=(
            "Human-like, scenario-driven task instructions implementing the program semantics; "
            "no code."
        )
    )


class TranslationEvaluatorResponse(BaseModel):
    """Structured evaluator output."""

    verdict: Literal["ok", "needs_revision", "impossible"] = Field(
        description=(
            "ok = description meets all criteria; needs_revision = issues found; "
            "impossible = cannot satisfy criteria."
        )
    )
    diagnosis: str = Field(
        default="",
        description="Brief diagnosis when verdict is needs_revision or impossible; empty if ok.",
    )
    suggestions: str = Field(
        default="",
        description="Actionable improvement suggestions when verdict is needs_revision; empty if ok.",
    )


def _parse_translation_generator_text(
    text: str,
    *,
    use_api_structured_output: bool = False,
) -> TranslationGeneratorResponse:
    """Parse generator JSON; prefer raw body when API structured output is enabled."""
    if use_api_structured_output:
        try:
            return TranslationGeneratorResponse.model_validate_json(text.strip())
        except Exception:
            pass
    return TranslationGeneratorResponse.model_validate(
        _extract_json_from_response(text, GENERATOR_RESPONSE_JSON_KEYS)
    )


def _parse_evaluator_response_text(
    text: str,
    *,
    use_api_structured_output: bool = False,
) -> TranslationEvaluatorResponse:
    if use_api_structured_output:
        try:
            return TranslationEvaluatorResponse.model_validate_json(text.strip())
        except Exception:
            pass
    return TranslationEvaluatorResponse.model_validate(
        _extract_json_from_response(text, EVALUATOR_RESPONSE_JSON_KEYS)
    )


def _format_generator_for_evaluator(
    raw_content: str,
    *,
    use_api_structured_output: bool = False,
) -> str:
    """Turn generator JSON (or legacy tagged text) into text for evaluator prompts."""
    try:
        parsed = _parse_translation_generator_text(
            raw_content, use_api_structured_output=use_api_structured_output
        )
        return (
            "<User Variable Definition>\n"
            f"{parsed.user_variable_definitions.strip()}\n"
            "<Task Instructions>\n"
            f"{parsed.task_instructions.strip()}"
        )
    except Exception:
        return raw_content


def _parse_evaluator_content(
    return_string: str,
    *,
    use_api_structured_output: bool = False,
) -> Optional[TranslationEvaluatorResponse]:
    try:
        return _parse_evaluator_response_text(
            return_string, use_api_structured_output=use_api_structured_output
        )
    except Exception:
        return None


def _flatten_system_messages_for_chat_api(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build OpenAI-style messages that Gemini-compatible proxies accept.

    Our agents use a single initial ``role: system`` message with the full task
    prompt and no following ``user`` turn on the first request. OpenAI allows
    that; Gemini's GenerateContent path often requires at least one *user*
    content, otherwise the gateway returns ``contents is not specified``.

    This copies messages, coerces ``content: None`` to ``""``, concatenates
    leading ``system`` messages, and prepends that text to the first ``user``
    message (or emits a lone ``user`` message if there is no user turn yet).
    """
    msgs: List[Dict[str, Any]] = []
    for m in messages:
        mm = dict(m)
        if mm.get("content") is None:
            mm["content"] = ""
        msgs.append(mm)

    i = 0
    system_chunks: List[str] = []
    while i < len(msgs) and msgs[i].get("role") == "system":
        system_chunks.append(str(msgs[i].get("content") or ""))
        i += 1
    rest = msgs[i:]
    sys_block = "\n\n".join(s for s in system_chunks if s.strip())

    if not system_chunks:
        if not rest:
            raise ValueError("chat completion messages are empty")
        return rest

    if not rest:
        return [{"role": "user", "content": sys_block or "."}]

    if rest[0].get("role") == "user":
        u0 = dict(rest[0])
        sep = "\n\n---\n\n" if sys_block.strip() else ""
        u0["role"] = "user"
        u0["content"] = sys_block + sep + str(u0.get("content") or "")
        return [u0] + rest[1:]

    # e.g. [system]*, assistant, ... — Gemini expects user before model turn
    prefix = {"role": "user", "content": sys_block or "Follow the instructions above."}
    return [prefix] + rest


def _translation_chat_completion(
    client: Any,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: Optional[int],
    temperature: float = 1.0,
    response_format: Optional[dict] = None,
) -> Tuple[str, dict]:
    """Synchronous chat completion (same shape as llm_trace_generation._call_llm)."""
    api_messages = _flatten_system_messages_for_chat_api(messages)
    approx_chars = sum(
        len(m.get("content", ""))
        if isinstance(m.get("content"), str)
        else "".join(
            b.get("text", "")
            for b in (m.get("content") or [])
            if isinstance(b, dict)
        )
        for m in api_messages
    )
    logger.debug(
        f"Translation LLM (chat): ~{approx_chars} prompt chars, "
        f"model={model}, max_tokens={max_tokens}"
    )
    t0 = time.time()
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": api_messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if response_format is not None:
        kwargs["response_format"] = response_format
    response = client.chat.completions.create(**kwargs)
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
            f"max_tokens={max_tokens}). Increase max_tokens."
        )
    return text, usage


def _record_assistant_reply(
    agent_book: dict, custom_id: str, content: str, role: str = "assistant"
) -> None:
    item_id = int(custom_id.split("_")[1])
    is_generator = custom_id.split("_")[0] == "generator"
    response = {"content": content, "role": role}
    if is_generator:
        agent_book[item_id]["generator"].record_response(response)
    else:
        agent_book[item_id]["evaluator"].record_response(response)


def _apply_batch_line_to_agents(agent_book: dict, item: dict) -> None:
    """Map one batch output line to generator/evaluator record_response."""
    custom_id = item["custom_id"]
    body = item["response"]["body"]["choices"][0]["message"]
    _record_assistant_reply(agent_book, custom_id, body["content"], body["role"])
GENERATOR_PROMPT = '''You are given a block of sequential API calls produced by a fuzzing engine, along with an optional initialization block. Your task is to translate these API calls into precise natural language descriptions that could be used as programming instructions for a developer.

Requirements:
	1.	The description must be unambiguous and accurately capture the semantics of the program.
	2.	The output should closely resemble how a programmer would describe a task in a typical software development scenario (e.g., feature request, ticket description, or implementation notes).
	3.	Instead of word-by-word translation, write the description in a humanlike and scenario-driven style, as if you are specifying a real-world feature or task.
	4.	The initialization block gives contextual information. While the values might differ, the variable's names should be fixed. You should provide variable names and explain them in the task descriptions. They are the parameters to the program. 
	5.	When describing the task requirements, include as few constants as possible. 
	6.	Do not include code in the output — only natural language.
    7.  Since an API documentation will be provided later along with the natural language descriptions, it is OK to omit some details regarding API formal specifications -- only natural language descriptions.

{application_description}

{program_intent_context}

Program:
```
# ===== Init Block Begin =====
{init_block}
# ===== Init Block End =====

{program}
```

In your description, first define user variable/constant name (such as user_variable_0) in the init block with necessary descriptions of their types and structures (e.g., dict with what key), 
but do not directly show their values since they may change with respect to different test cases. The user-defined contents will be automatically loaded upon evaluation.
Provide necessary instructions when these variable/constant names are referred to. ONLY refer to names, not values. 
All user-provided content should be used (e.g., when the variable value matches user_constant_0) if they are referred in the program.
If there is RESULT variable, specify it as its vaule will be checked using test cases.

Note: The generated description will be used to evaluate LLMs' ability to understand natural language and produce correct stateful code.
Aim to make the description as unambiguous as possible. When appropriate, allow the LLM to infer implicit states instead of explicitly specifying every detail.

Encode your answer as a single JSON object (see Output Format section appended to this prompt). Use fields ``user_variable_definitions`` and ``task_instructions`` for the prose that would previously have appeared under <User Variable Definition> and <Task Instructions> respectively. Do not emit those XML-like tags in the JSON string values unless they are natural-language content.
'''


def _format_program_intent_context_for_generator(program_intent_summary: str) -> str:
    """Optional block injected into GENERATOR_PROMPT when a trace intent summary exists."""
    text = (program_intent_summary or "").strip()
    if not text:
        return ""
    return (
        "## Reference: program intent summary\n"
        "The following was produced when this trace was generated or refined. "
        "Use it as auxiliary context; the initialization block and program "
        "below remain the source of truth for what to describe.\n\n"
        f"{text}\n"
    )


GENERATOR_IMPROVE_PROMPT = ''' An evaluator agent has checked the descriptions and offered some advices. Please improve your description:
{evaluator_output}
'''

EVALUATOR_PROMPT = '''
You are a checker agent tasked with evaluating the quality of a natural language description that was generated from a sequence of API calls. Your job is to determine whether the description meets two key criteria:

1.  Fidelity to the Program Logic
	a. Does the natural language description accurately reflect all steps of the given API call sequence?
	b. Would a developer be able to reconstruct the original program logic (or a functionally equivalent one) based solely on the description?
	c. Are any steps missing, incorrectly described, or added?
	d. Is there any ambiguity? For example, sometimes "update" can refer to either "local update" or "remote update through APIs." 
2.  Human-likeness and Fluency
	a. Does the description sound natural and fluent, as if it were written by a human programmer?
    b. Does the description present the task in a plausible use-case or user scenario?
	c. Is the tone and phrasing consistent with how developers describe implementation tasks?
3.  Redundancy
	a. Is the description redundant? For example, if the description is too verbose or if it contains unnecessary details that could be inferred from other parts of the description.

Respond with a single JSON object (see Output Format section appended to this prompt). Use verdict ``ok`` if all criteria are met, ``needs_revision`` if issues are found (fill diagnosis and suggestions), or ``impossible`` if no satisfactory description exists.
Below is the program:
```
# ===== Init Block Begin =====
{init_block}
# ===== Init Block End =====

{program}
```

Below is the description:
# ===== Description Begin =====
{description}
# ===== Description End =====
'''

EVALUATOR_FURTHER_PROMPT = '''
Here are the updated descriptions according to your suggestions:
{description}

Re-evaluate using the same criteria. Respond with a single JSON object (see Output Format section appended to this prompt): verdict ok, needs_revision, or impossible; include diagnosis and suggestions when applicable.
'''

class AgentStatus(Enum):
    BEGIN = 0
    CONTINUE = 1
    END = 2


class Generator():
    def __init__(self, application_description,
                 max_iterations,
                 generator_prompt=GENERATOR_PROMPT, 
                 improvement_prompt=GENERATOR_IMPROVE_PROMPT,
                 use_api_structured_output: bool = False):
        self.application_description = application_description
        self.generator_prompt = generator_prompt
        self.improvement_prompt = improvement_prompt
        self.use_api_structured_output = use_api_structured_output
        self.status = AgentStatus.BEGIN
        self.request_counter = 0
        self.max_iterations = max_iterations
        self.message = [{
            "role": "system",
            "content": None,
        }]
        
    def get_generate_prompt(self, init_block, program, program_intent_summary: str = ""):
        program_intent_context = _format_program_intent_context_for_generator(
            program_intent_summary
        )
        body = self.generator_prompt.format(
            application_description=self.application_description,
            program_intent_context=program_intent_context,
            init_block=init_block,
            program=program,
        )
        if self.use_api_structured_output:
            return body + STRUCTURED_API_GENERATOR_PROMPT_TAIL
        schema_json = _compact_schema(TranslationGeneratorResponse.model_json_schema())
        return body + _JSON_OUTPUT_INSTRUCTION + schema_json
    
    def get_improve_prompt(self, evaluator_output):
        return self.improvement_prompt.format(
            evaluator_output=evaluator_output,
        )
    
    def analyze_output(self, return_string):
        try:
            parsed = _parse_translation_generator_text(
                return_string,
                use_api_structured_output=self.use_api_structured_output,
            )
            return {
                "init_info": parsed.user_variable_definitions.strip(),
                "description": parsed.task_instructions.strip(),
            }
        except Exception:
            sections = return_string.split("<User Variable Definition>")
            if len(sections) < 2:
                raise ValueError(
                    "Invalid response format: expected JSON or legacy "
                    "<User Variable Definition> / <Task Instructions> sections"
                ) from None
            var_section, remaining = sections[1].split("<Task Instructions>")
            return {
                "init_info": var_section.strip(),
                "description": remaining.strip(),
            }

    def get_all_generated_description(self):
        assert self.message[-1]["role"] == "assistant"
        return _format_generator_for_evaluator(
            self.message[-1]["content"],
            use_api_structured_output=self.use_api_structured_output,
        )
    
    def interact(self, info_dict):
        if self.status == AgentStatus.BEGIN:
            assert "init_block" in info_dict
            assert "program" in info_dict
            self.message[0]["content"] = self.get_generate_prompt(
                info_dict["init_block"],
                info_dict["program"],
                info_dict.get("program_intent_summary") or "",
            )
            self.request_counter += 1
            return True, self.message
        elif self.status == AgentStatus.CONTINUE:
            assert "evaluator_output" in info_dict
            self.message.append({
                "role": "user",
                "content": self.get_improve_prompt(info_dict["evaluator_output"])
            })
            self.request_counter += 1
            return True, self.message
        elif self.status == AgentStatus.END:
            return False, None
    
    def record_response(self, message):
        if self.status == AgentStatus.BEGIN:
            self.status = AgentStatus.CONTINUE
        self.message.append(message)
        if self.request_counter >= self.max_iterations:
            self.status = AgentStatus.END
        return self.status
    
    
class Evaluator():
    def __init__(self, application_description,
                 max_iterations, 
                 evaluator_prompt=EVALUATOR_PROMPT, 
                 further_prompt=EVALUATOR_FURTHER_PROMPT,
                 use_api_structured_output: bool = False):
        self.application_description = application_description
        self.evaluator_prompt = evaluator_prompt
        self.further_prompt = further_prompt
        self.use_api_structured_output = use_api_structured_output
        self.status = AgentStatus.BEGIN
        self.request_counter = 0
        self.max_iterations = max_iterations
        # Even when using previous_response_id, all previous input tokens for responses in the chain are billed as input tokens in the API.
        self.message = [{
            "role": "system",
            "content": None,
        }]
        
    def get_evaluate_prompt(self, init_block, program, description):
        body = self.evaluator_prompt.format(
            init_block=init_block,
            program=program,
            description=description,
        )
        if self.use_api_structured_output:
            return body + STRUCTURED_API_EVALUATOR_PROMPT_TAIL
        schema_json = _compact_schema(TranslationEvaluatorResponse.model_json_schema())
        return body + _JSON_OUTPUT_INSTRUCTION + schema_json

    def get_further_prompt(self, description):
        body = self.further_prompt.format(description=description)
        if self.use_api_structured_output:
            return body + STRUCTURED_API_EVALUATOR_PROMPT_TAIL
        schema_json = _compact_schema(TranslationEvaluatorResponse.model_json_schema())
        return body + _JSON_OUTPUT_INSTRUCTION + schema_json

    def whether_continue(self, return_string):
        parsed = _parse_evaluator_content(
            return_string,
            use_api_structured_output=self.use_api_structured_output,
        )
        if parsed is not None:
            return parsed.verdict == "needs_revision"
        if "<OK>" in return_string or "<IMPOSSIBLE>" in return_string:
            return False
        return True
    
    def interact(self, info_dict):
        if self.status == AgentStatus.BEGIN:
            assert "init_block" in info_dict
            assert "program" in info_dict
            assert "description" in info_dict
            self.status = AgentStatus.CONTINUE
            self.message[0]["content"] = self.get_evaluate_prompt(info_dict["init_block"], info_dict["program"], info_dict["description"])
            self.request_counter += 1
            return True, self.message
        elif self.status == AgentStatus.CONTINUE:
            assert "description" in info_dict
            user_prompt = self.get_further_prompt(info_dict["description"])
            self.message.append({
                "role": "user",
                "content": user_prompt
            })
            self.request_counter += 1
            return True, self.message
        elif self.status == AgentStatus.END:
            return False, None
    
    def record_response(self, message):
        self.message.append(message)
        flag = self.whether_continue(message["content"])
        if flag:
            self.status = AgentStatus.CONTINUE
        else:
            self.status = AgentStatus.END
        if self.request_counter >= self.max_iterations:
            self.status = AgentStatus.END
        return flag
    
    def get_evaluator_output(self):
        assert self.message[-1]["role"] == "assistant"
        raw = self.message[-1]["content"]
        parsed = _parse_evaluator_content(
            raw, use_api_structured_output=self.use_api_structured_output
        )
        if parsed is not None and parsed.verdict == "needs_revision":
            parts = [p for p in (parsed.diagnosis.strip(), parsed.suggestions.strip()) if p]
            text = "\n".join(parts) if parts else raw
            return {"evaluator_output": text}
        return {"evaluator_output": raw}

class MultiAgentTrans():
    def __init__(self,
                 program_info,
                 max_iterations,
                 application_description,
                 openai_client,
                 wait_time=10,
                 generator_prompt=GENERATOR_PROMPT,
                 evaluator_prompt=EVALUATOR_PROMPT,
                 max_tokens=None,
                 model_type="gpt-4.1",
                 url="/v1/chat/completions",
                 evaluator_agent_params=None,
                 batch_mode: bool = False,
                 temperature: float = 1.0,
                 use_api_structured_output: bool = True):
        '''
        Args:
            program_info: list of program info
                Each item is a dict with "init_block" and "program". Optional
                ``program_intent_summary`` (str) is forwarded to the generator's
                first prompt when present (e.g. from ``program_intent_summaries/``).
            max_iterations: maximum iterations
            application_description: application description
            openai_client: openai client
            wait_time: wait time
            batch_mode: If True, use OpenAI Batch API (jsonl + poll). If False,
                use synchronous ``chat.completions.create`` like llm_trace_generation.
            temperature: Used only when ``batch_mode`` is False.
            use_api_structured_output: If True, send ``response_format`` json_schema
                and parse replies with ``model_validate_json`` first. If the provider
                does not support it (e.g. some Gemini proxies), set False to use
                prompt-only JSON plus heuristic extraction.
        '''
        self.use_api_structured_output = use_api_structured_output
        self.agent_book = self.prepare_agent(
            program_info,
            max_iterations,
            application_description,
            generator_prompt,
            evaluator_prompt,
            use_api_structured_output=use_api_structured_output,
        )
        self.max_iterations = max_iterations
        self.message_recorder = dict()
        self.max_tokens = max_tokens
        self.model_type = model_type
        self.url = url
        # Optional evaluator-specific request overrides.
        # If not provided, evaluator reuses generator/default request params.
        self.evaluator_agent_params = evaluator_agent_params or {}
        self.evaluator_max_tokens = self.evaluator_agent_params.get("max_tokens", self.max_tokens)
        self.evaluator_model_type = self.evaluator_agent_params.get("model_type", self.model_type)
        self.evaluator_url = self.evaluator_agent_params.get("url", self.url)
        self.client = openai_client
        self.wait_time = wait_time
        self.batch_mode = batch_mode
        self.temperature = temperature
        
    def wait_for_batch_completion(self, batch_id):
        """Poll batch status until completion"""
        import time
        while True:
            batch_status = self.client.batches.retrieve(batch_id)
            if batch_status.status in ['completed', 'failed', 'cancelled']:
                return
            time.sleep(self.wait_time)  # Check every 10 seconds
            
    def prepare_agent(
        self,
        program_info,
        max_iterations,
        application_description,
        generator_prompt=GENERATOR_PROMPT,
        evaluator_prompt=EVALUATOR_PROMPT,
        *,
        use_api_structured_output: bool = False,
    ):
        agent_book = dict()
        for idx, item in enumerate(program_info):
            agent_book[idx] = {
                "generator": Generator(
                    application_description,
                    max_iterations,
                    generator_prompt,
                    use_api_structured_output=use_api_structured_output,
                ),
                "evaluator": Evaluator(
                    application_description,
                    max_iterations,
                    evaluator_prompt,
                    use_api_structured_output=use_api_structured_output,
                ),
            }
        return agent_book

    def wrap_multi_agent_message(
        self,
        program_info,
        agent_book,
        turn
        ):
        '''
        Perform multi-agent translation.
        Args:
            program_info: list of program info
            agent_book: agent book returned by prepare_agent
            turn: Generator -> Evaluator -> Generator -> ...
        '''
        message_list = []
        assert turn in ["Generator", "Evaluator"]
        for idx, item in enumerate(program_info):
            init_block = item["init_block"]
            init_load_str = item["init_load_str"]
            if init_load_str is not None:
                init_block = init_load_str + "\n" + init_block
            
            program = item["program"]
            generator = agent_book[idx]["generator"]
            evaluator = agent_book[idx]["evaluator"]
            if generator.status == AgentStatus.BEGIN:
                # Sequence begin with the generator
                generator_info = {
                    "init_block": init_block,
                    "program": program,
                    "program_intent_summary": item.get("program_intent_summary") or "",
                }
                flag, generator_output = generator.interact(generator_info)
                message_list.append(("generator_{}".format(idx), generator_output))
            else:
                if generator.status == AgentStatus.CONTINUE:
                    if evaluator.status == AgentStatus.BEGIN:
                        # The first time the evaluator is called, it needs the init_block and program information.
                        all_generated_description = generator.get_all_generated_description()
                        evaluator_info = {
                            "init_block": init_block,
                            "program": program,
                            "description": all_generated_description
                        }
                        flag, evaluator_output = evaluator.interact(evaluator_info)
                        message_list.append(("evaluator_{}".format(idx), evaluator_output))
                    elif evaluator.status == AgentStatus.CONTINUE:
                        if turn == "Evaluator":
                            all_generated_description = generator.get_all_generated_description()
                            evaluator_info = {
                                "description": all_generated_description
                            }
                            flag, evaluator_output = evaluator.interact(evaluator_info)
                            message_list.append(("evaluator_{}".format(idx), evaluator_output))
                        elif turn == "Generator":
                            evaluator_output = evaluator.get_evaluator_output()
                            flag, generator_output = generator.interact(evaluator_output)
                            message_list.append(("generator_{}".format(idx), generator_output))
                    else:
                        generator.status = AgentStatus.END
                        continue
                else:
                    evaluator.status = AgentStatus.END
                    continue
        return message_list

    def collect_multi_agent_message(
        self,
        openai_message):
        '''
        '''
        for line in openai_message.iter_lines():
            if not line.strip():
                continue
            _apply_batch_line_to_agents(self.agent_book, json.loads(line))

    def _prepare_round_jsonl(
        self,
        rml,
        round_info,
        model_type,
        url,
        max_tokens,
        output_dir,
        response_format=None,
    ):
        request_id_list = [item[0] for item in rml]
        message_list = [item[1] for item in rml]
        output_path = os.path.join(output_dir, "round_{}_openai_input.jsonl".format(round_info))
        _, _ = generate_jsonl_for_openai(
            request_id_list,
            message_list,
            output_path,
            max_tokens,
            model_type,
            url,
            response_format=response_format,
        )
        return output_path

    def _run_round_chat(
        self,
        request_message_list,
        output_path,
        recorder_key,
        response_format: Optional[dict] = None,
    ):
        """Synchronous chat.completions per request (llm_trace_generation-style)."""
        sync_calls = []
        rf = response_format if self.use_api_structured_output else None
        for custom_id, messages in request_message_list:
            is_generator = custom_id.split("_")[0] == "generator"
            model = self.model_type if is_generator else self.evaluator_model_type
            max_tok = self.max_tokens if is_generator else self.evaluator_max_tokens
            text, usage = _translation_chat_completion(
                self.client,
                model,
                messages,
                max_tok,
                self.temperature,
                response_format=rf,
            )
            _record_assistant_reply(self.agent_book, custom_id, text, "assistant")
            sync_calls.append({"custom_id": custom_id, "usage": usage})
        self.message_recorder[recorder_key] = {
            "mode": "chat",
            "input_jsonl": output_path,
            "sync_calls": sync_calls,
        }

    def _run_round_batch(
        self,
        request_message_list,
        output_path,
        url,
        recorder_key,
        description,
    ):
        batch_submit_info, _ = submit_batch_request_openai(
            self.client, output_path, url, description=description
        )
        logger.info("Waiting for batch completion: {}".format(description))
        self.wait_for_batch_completion(batch_submit_info.id)
        batch_result_info = self.client.batches.retrieve(batch_submit_info.id)
        file_response = self.client.files.content(batch_result_info.output_file_id)
        self.collect_multi_agent_message(file_response)
        self.message_recorder[recorder_key] = {
            "mode": "batch",
            "file_response": file_response,
            "batch_submit_info": batch_submit_info,
            "batch_result_info": batch_result_info,
            "input_jsonl": output_path,
        }

    def interact_loop(self, program_info, output_dir):
        rf_generator = (
            openai_json_schema_response_format(
                TranslationGeneratorResponse, "translation_generator"
            )
            if self.use_api_structured_output
            else None
        )
        rf_evaluator = (
            openai_json_schema_response_format(
                TranslationEvaluatorResponse, "translation_evaluator"
            )
            if self.use_api_structured_output
            else None
        )

        request_message_list = self.wrap_multi_agent_message(
            program_info, self.agent_book, "Generator"
        )
        output_path = self._prepare_round_jsonl(
            request_message_list,
            "{}_generator".format(0),
            self.model_type,
            self.url,
            self.max_tokens,
            output_dir,
            response_format=rf_generator,
        )
        if self.batch_mode:
            self._run_round_batch(
                request_message_list,
                output_path,
                self.url,
                "{}_generator".format(0),
                "translation_{}_generator".format(0),
            )
        else:
            self._run_round_chat(
                request_message_list,
                output_path,
                "{}_generator".format(0),
                response_format=rf_generator,
            )

        for round in range(1, self.max_iterations):
            request_message_list = self.wrap_multi_agent_message(
                program_info, self.agent_book, "Evaluator"
            )
            if len(request_message_list) == 0:
                break
            output_path = self._prepare_round_jsonl(
                request_message_list,
                "{}_evaluator".format(round),
                self.evaluator_model_type,
                self.evaluator_url,
                self.evaluator_max_tokens,
                output_dir,
                response_format=rf_evaluator,
            )
            if self.batch_mode:
                self._run_round_batch(
                    request_message_list,
                    output_path,
                    self.evaluator_url,
                    "{}_evaluator".format(round),
                    "translation_{}_evaluator".format(round),
                )
            else:
                self._run_round_chat(
                    request_message_list,
                    output_path,
                    "{}_evaluator".format(round),
                    response_format=rf_evaluator,
                )

            request_message_list = self.wrap_multi_agent_message(
                program_info, self.agent_book, "Generator"
            )
            if len(request_message_list) == 0:
                break
            output_path = self._prepare_round_jsonl(
                request_message_list,
                "{}_generator".format(round),
                self.model_type,
                self.url,
                self.max_tokens,
                output_dir,
                response_format=rf_generator,
            )
            if self.batch_mode:
                self._run_round_batch(
                    request_message_list,
                    output_path,
                    self.url,
                    "{}_generator".format(round),
                    "translation_{}_generator".format(round),
                )
            else:
                self._run_round_chat(
                    request_message_list,
                    output_path,
                    "{}_generator".format(round),
                    response_format=rf_generator,
                )

        return self.agent_book

    def save_agent_data(self):
        """Save agent states including message history and last evaluator output"""
        saved_data = {}
        for idx, agent_pair in self.agent_book.items():
            generator = agent_pair["generator"]
            evaluator = agent_pair["evaluator"]
            
            # Last generator assistant message (raw JSON or legacy text)
            last_generator_output = None
            for msg in reversed(generator.message):
                if msg["role"] == "assistant":
                    last_generator_output = msg["content"]
                    break

            last_generator_structured: Optional[dict] = None
            if last_generator_output:
                try:
                    last_generator_structured = _parse_translation_generator_text(
                        last_generator_output,
                        use_api_structured_output=self.use_api_structured_output,
                    ).model_dump()
                except Exception as e:
                    logger.warning(
                        "Could not parse last generator reply as "
                        "TranslationGeneratorResponse for idx={}: {}",
                        idx,
                        e,
                    )

            last_evaluator_output = None
            last_evaluator_structured: Optional[dict] = None
            for msg in reversed(evaluator.message):
                if msg["role"] == "assistant":
                    last_evaluator_output = msg["content"]
                    break
            if last_evaluator_output:
                try:
                    last_evaluator_structured = _parse_evaluator_response_text(
                        last_evaluator_output,
                        use_api_structured_output=self.use_api_structured_output,
                    ).model_dump()
                except Exception as e:
                    logger.warning(
                        "Could not parse last evaluator reply as "
                        "TranslationEvaluatorResponse for idx={}: {}",
                        idx,
                        e,
                    )

            saved_data[idx] = {
                "generator_messages": generator.message,
                "evaluator_messages": evaluator.message,
                "last_generator_output": last_generator_output,
                "last_generator_structured": last_generator_structured,
                "last_evaluator_output": last_evaluator_output,
                "last_evaluator_structured": last_evaluator_structured,
            }
        return saved_data


LLM_PROGRAM_GENERATION_PROMPT = '''Program Generation from API Documentation:
Given the following API documentation, generate a program that uses these APIs. The program should adhere to the following constraints:
1. Number of API Calls:
    - The execution trace of the program must contain exactly 5 API calls from start to finish, regardless of any branching logic.
2. Program Structure:
    - The program should start with an initialization block (e.g., variable setup, object creation, etc.).
    - After initialization, the main script should execute the API calls.
3. Branching Logic:
    - The program may or may not include an if-else branch.
    - If an if-else branch is present, ensure that both the 'if' and 'else' branches result in exactly 5 API calls in their respective execution traces (including any calls before the branch, if applicable).
4. Fuzzing-Like Diversity:
    - The generated programs should be diverse, exploring different combinations and orders of API calls, as in fuzz testing.
    - You may use random or varied input values for API calls to increase diversity.
    
API Documentation: {api_doc}

Output Format:
Output only the code for the generated program enclosed in ``` ``` tags.
Do not include explanations or comments unless specified.
Example:
```
{example_program}
```

Split the generation of each program through ``` ``` tags. Please generate {num_programs} programs in sequential order.
'''

SESSION_EXAMPLE = '''
User_variable_0 = {'id': None, 'source': 'user_portal', 'type': 'custom_gene_list', 'checksum': '', 'data': {'title': 'Ovarian Cancer Analysis', 'members': 'josephleblanc@example.com', 'similarities': '17'}}
user_constant_2 = "17"

if User_variable_0["data"]["similarities"] == user_constant_2:
    source = User_variable_0['source']
    type = User_variable_0['type']
    url = f'{BASE_URL}/api/sessions/{source}/{type}'
    response_4 = requests.get(url)
'''

TENSOR_EXAMPLE = '''
user_constant_0 = "torch.float32"

response_1 = torch.nn.functional.conv2d(user_tensor_0, user_tensor_0, stride=4, padding=0, dilation=1)
response_2 = torch.cat((response_1, response_1), 0) # Output shape: torch.Size([82, 41, 1, 1])
'''

VOICE_EXAMPLE = '''
user_variable_0 = "True once camera while beat voice."
response_1 = text_to_speech(text=user_variable_0, style=0.65)
'''

# ===== Test =====
# Test Generator class
def test_generator_initialization():
    generator = Generator("Test App", 3)
    assert generator.status == AgentStatus.BEGIN
    assert generator.request_counter == 0

def test_generate_prompt_formatting():
    generator = Generator("Test Description", 3)
    init_block = "User_constant_0 = 5"
    program = "API.call(param=User_constant_0)"
    
    prompt = generator.get_generate_prompt(init_block, program)
    assert "Test Description" in prompt
    prompt_with_hint = generator.get_generate_prompt(
        init_block, program, program_intent_summary="Summarize session flow."
    )
    assert "Summarize session flow." in prompt_with_hint
    assert "program intent summary" in prompt_with_hint.lower()
    assert init_block in prompt
    assert program in prompt

def test_response_parsing():
    generator = Generator("", 3)
    test_response = '''<User Variable Definition>
    user_variable_0 is a dict
    <Task Instructions>
    Create something'''

    parsed = generator.analyze_output(test_response)
    assert parsed["init_info"] == "user_variable_0 is a dict"
    assert parsed["description"] == "Create something"

    json_resp = (
        '{"user_variable_definitions": "user_variable_0 is a dict", '
        '"task_instructions": "Create something"}'
    )
    parsed_json = generator.analyze_output(json_resp)
    assert parsed_json["init_info"] == "user_variable_0 is a dict"
    assert parsed_json["description"] == "Create something"


def test_format_generator_for_evaluator():
    raw = (
        '{"user_variable_definitions": "v0 is a dict", '
        '"task_instructions": "Do the thing"}'
    )
    formatted = _format_generator_for_evaluator(raw)
    assert "<User Variable Definition>" in formatted
    assert "v0 is a dict" in formatted
    assert "<Task Instructions>" in formatted
    assert "Do the thing" in formatted


def test_extract_json_prefers_full_generator_object_over_schema_snippet():
    text = (
        'Preamble {"description": "Structured output", "type": "object"} trailer '
        '{"user_variable_definitions": "v0 is int", "task_instructions": "Do work"}'
    )
    d = _extract_json_from_response(text, GENERATOR_RESPONSE_JSON_KEYS)
    assert d["user_variable_definitions"] == "v0 is int"
    assert d["task_instructions"] == "Do work"


def test_flatten_system_messages_lone_system():
    out = _flatten_system_messages_for_chat_api([{"role": "system", "content": "Do X."}])
    assert len(out) == 1
    assert out[0]["role"] == "user"
    assert "Do X." in out[0]["content"]


def test_flatten_system_messages_system_then_user():
    out = _flatten_system_messages_for_chat_api(
        [{"role": "system", "content": "Sys."}, {"role": "user", "content": "Hi"}]
    )
    assert len(out) == 1
    assert out[0]["role"] == "user"
    assert "Sys." in out[0]["content"] and "Hi" in out[0]["content"]


def test_flatten_system_messages_system_then_assistant():
    out = _flatten_system_messages_for_chat_api(
        [
            {"role": "system", "content": "Rules."},
            {"role": "assistant", "content": "{}"},
        ]
    )
    assert out[0]["role"] == "user"
    assert out[1]["role"] == "assistant"


# Test Evaluator class
def test_evaluator_decision_making():
    evaluator = Evaluator("", 3)
    assert evaluator.whether_continue("<OK>") is False
    assert evaluator.whether_continue("Needs improvement") is True
    assert (
        evaluator.whether_continue(
            '{"verdict": "ok", "diagnosis": "", "suggestions": ""}'
        )
        is False
    )
    assert (
        evaluator.whether_continue(
            '{"verdict": "needs_revision", "diagnosis": "x", "suggestions": "y"}'
        )
        is True
    )

def test_evaluator_prompt_generation():
    evaluator = Evaluator("", 3)
    prompt = evaluator.get_evaluate_prompt(
        init_block="init",
        program="program steps",
        description="test description"
    )
    assert "init" in prompt
    assert "program steps" in prompt
    assert "test description" in prompt

# Test state transitions
def test_generator_state_flow():
    generator = Generator("", 3)
    generator.interact({"init_block": "", "program": ""})
    assert generator.status == AgentStatus.CONTINUE
    assert generator.request_counter == 1

def test_evaluator_state_transitions():
    evaluator = Evaluator("", 3)
    evaluator.interact({
        "init_block": "",
        "program": "",
        "description": ""
    })
    assert evaluator.status == AgentStatus.CONTINUE
    assert evaluator.request_counter == 1
# ===== Test =====


