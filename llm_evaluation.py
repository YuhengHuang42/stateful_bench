"""
This script is used to evaluate the LLM performance on the (local version of) stateful benchmark.
"""
import typer
from Sgenerator.state import StateEval
from typing import Annotated, Optional, Union
from pathlib import Path
import yaml
from loguru import logger
import openai
import json
import traceback
import pickle
import os
from tqdm import tqdm
import numpy as np


try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None
# StateEval: parent_path: str, task: str, config_dict: dict, api_doc: str

from Sgenerator.utils import generate_jsonl_for_openai, submit_batch_request_openai
DEFAULT_MAX_TOKENS = 16384

app = typer.Typer(pretty_exceptions_show_locals=False, pretty_exceptions_short=False)
PROJECT_ROOT = Path(__file__).resolve().parent
if load_dotenv:
    load_dotenv(PROJECT_ROOT / ".env")

def wait_for_batch_completion(client, batch_id, wait_time=10):
    """Poll batch status until completion"""
    import time
    while True:
        batch_status = client.batches.retrieve(batch_id)
        if batch_status.status in ['completed', 'failed', 'cancelled']:
            return
        time.sleep(wait_time)  # Check every 10 seconds


def _resolve_eval_batch_mode(env_config: dict, cli_batch_mode: Optional[bool]) -> bool:
    if cli_batch_mode is not None:
        return cli_batch_mode
    batch_mode_env = os.environ.get("LLM_EVAL_BATCH_MODE")
    if batch_mode_env is not None and str(batch_mode_env).strip():
        return str(batch_mode_env).strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
    return bool(env_config.get("eval_batch_mode", True))


def _resolve_openai_compat_credentials(env_config: dict, cli_base_url: Optional[str]):
    base_url = cli_base_url
    if base_url is None:
        base_url = os.environ.get("BASE_URL") or env_config.get("open_source_base_url")
    if base_url is not None and str(base_url).strip() == "":
        base_url = None
    api_key = (
        os.environ.get("API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or env_config.get("openai_api_key", "")
    )
    return base_url, api_key


def _resolve_openrouter_credentials(env_config: dict, cli_base_url: Optional[str]):
    """
    Resolve OpenRouter connection settings.
    Priority:
      - base_url: CLI override -> OPENROUTER_BASE_URL -> default OpenRouter endpoint
      - api_key: OPENROUTER_API_KEY -> config env.openrouter_api_key
    """
    base_url = cli_base_url
    if base_url is None:
        base_url = os.environ.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
    if base_url is not None and str(base_url).strip() == "":
        base_url = "https://openrouter.ai/api/v1"
    api_key = os.environ.get("OPENROUTER_API_KEY") or env_config.get("openrouter_api_key", "")
    return base_url, api_key


def _resolve_openrouter_headers(env_config: dict) -> dict:
    """
    Optional OpenRouter attribution headers.
    See: https://openrouter.ai/docs
    """
    referer = os.environ.get("OPENROUTER_SITE_URL") or env_config.get("openrouter_site_url")
    title = os.environ.get("OPENROUTER_APP_NAME") or env_config.get("openrouter_app_name")
    headers = {}
    if referer:
        headers["HTTP-Referer"] = str(referer)
    if title:
        headers["X-Title"] = str(title)
    return headers


def _should_use_openai_compat(target_llm: str, cli_base_url: Optional[str]) -> bool:
    """
    Route OpenAI-compatible model ids (including provider/model forms like tensorblock/*)
    to the OpenAI-compatible client branch.
    """
    model = str(target_llm).strip().lower()
    if ("gpt" in model) or ("claude" in model):
        return True
    if cli_base_url is not None and str(cli_base_url).strip():
        return True
    # Provider-prefixed ids are typically served via OpenAI-compatible gateways.
    if "/" in model and not model.startswith("models/"):
        return True
    return False


def _is_openrouter_model(target_llm: str) -> bool:
    """
    Heuristic for common OpenRouter model-id patterns:
    - provider/model (e.g. deepseek/deepseek-v3.2)
    - explicit openrouter namespace
    """
    model = str(target_llm).strip().lower()
    if model.startswith("openrouter/"):
        return True
    return "/" in model and not model.startswith("models/")


def _safe_artifact_tag(model: str) -> str:
    """Model ids may contain slashes (e.g. provider/model); strip for filesystem-safe filenames."""
    s = model.strip()
    for ch in ("/", "\\", "\x00"):
        s = s.replace(ch, "_")
    return s or "model"


def _ensure_parent_dir(path: Union[str, os.PathLike]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)


def _persist_result(path: Union[str, os.PathLike], result: dict) -> None:
    """Best-effort persistence helper used for crash-safe checkpoints."""
    _ensure_parent_dir(path)
    with open(path, "wb") as file:
        pickle.dump(result, file)


def _openai_batch_supported(base_url: Optional[str]) -> bool:
    """
    OpenAI Batch + Files APIs exist only on the official host. Most OpenAI-compatible
    servers (custom base_url) implement chat completions only and return 404 on batch.
    """
    if base_url is None:
        return True
    u = base_url.strip().rstrip("/").lower()
    return u in ("https://api.openai.com/v1", "http://api.openai.com/v1")

def _is_responses_api(url: str) -> bool:
    return str(url).strip().rstrip("/").endswith("/responses")


def _should_retry_llm_with_max_completion_tokens(error_text: str) -> bool:
    """Detect providers requiring max_completion_tokens instead of max_tokens."""
    if not error_text:
        return False
    t = error_text.lower()
    return (
        "max_completion_tokens" in error_text
        and "max_tokens" in error_text
        and ("unsupported" in t or "invalid_request" in t or "not supported" in t)
    )


def _extract_text_from_response_payload(payload):
    """Best-effort extraction for both chat.completions and responses APIs."""
    def _flatten_content_text(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_chunks = []
            for part in content:
                if isinstance(part, str):
                    text_chunks.append(part)
                elif isinstance(part, dict):
                    txt = part.get("text")
                    if isinstance(txt, str):
                        text_chunks.append(txt)
                else:
                    txt = getattr(part, "text", None)
                    if isinstance(txt, str):
                        text_chunks.append(txt)
            if text_chunks:
                return "\n".join(text_chunks)
        return ""

    if payload is None:
        return ""
    # 1) OpenAI Python responses API convenience field.
    output_text = getattr(payload, "output_text", None)
    if isinstance(output_text, str) and output_text:
        return output_text
    # 2) Chat completions-like shape.
    try:
        content = payload["choices"][0]["message"]["content"]
        flattened = _flatten_content_text(content)
        if flattened:
            return flattened
    except Exception:
        pass
    # 2b) Object access for chat.completions SDK objects.
    try:
        choices = getattr(payload, "choices", [])
        if choices:
            message = getattr(choices[0], "message", None)
            if message is not None:
                content = getattr(message, "content", None)
                flattened = _flatten_content_text(content)
                if flattened:
                    return flattened
    except Exception:
        pass
    # 3) Responses-like nested shape.
    try:
        output = payload.get("output", [])
        text_chunks = []
        for item in output:
            for c in item.get("content", []):
                txt = c.get("text")
                if isinstance(txt, str):
                    text_chunks.append(txt)
        if text_chunks:
            return "\n".join(text_chunks)
    except Exception:
        pass
    # 4) Object access for responses SDK objects.
    try:
        output = getattr(payload, "output", [])
        text_chunks = []
        for item in output:
            contents = getattr(item, "content", [])
            for c in contents:
                txt = getattr(c, "text", None)
                if isinstance(txt, str):
                    text_chunks.append(txt)
        if text_chunks:
            return "\n".join(text_chunks)
    except Exception:
        pass
    return ""


def _serialize_openai_payload(payload):
    """Convert SDK response objects to plain dicts for durable logging."""
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    # OpenAI SDK objects expose model_dump(); keep a broad fallback.
    try:
        dumped = payload.model_dump()
        if isinstance(dumped, dict):
            return dumped
    except Exception:
        pass
    try:
        return dict(payload)
    except Exception:
        return {"_raw_repr": repr(payload)}


def _extract_choice_error(payload_dict: dict) -> Optional[str]:
    """Return provider-side choice error text (if present) from chat payload."""
    if not isinstance(payload_dict, dict):
        return None
    try:
        choices = payload_dict.get("choices", [])
        if not choices:
            return None
        first = choices[0]
        if not isinstance(first, dict):
            return None
        err = first.get("error")
        if not isinstance(err, dict):
            return None
        code = err.get("code")
        msg = err.get("message")
        meta = err.get("metadata")
        provider = payload_dict.get("provider")
        return (
            f"provider={provider}, code={code}, message={msg}, metadata={meta}"
        )
    except Exception:
        return None


def _chat_completion_with_retry(
    client,
    target_llm: str,
    messages: list,
    chat_max_tokens: int,
):
    """
    Call chat.completions once and fail on any provider-side payload error.
    Keeps max_tokens -> max_completion_tokens fallback for compatibility.
    """
    kwargs = {
        "model": target_llm,
        "messages": messages,
        "max_tokens": chat_max_tokens,
    }
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as e:
        if _should_retry_llm_with_max_completion_tokens(str(e)):
            kwargs.pop("max_tokens", None)
            kwargs["max_completion_tokens"] = chat_max_tokens
            logger.warning(
                "Provider rejected max_tokens; retrying with "
                "max_completion_tokens."
            )
            response = client.chat.completions.create(**kwargs)
        else:
            raise

    body = _serialize_openai_payload(response)
    choice_error = _extract_choice_error(body)
    if choice_error is not None:
        raise RuntimeError(
            f"Provider returned choice-level error payload: {choice_error}"
        )
    return response


def _log_response_diagnostics(raw_message_list):
    """Summarize finish reasons and suspiciously short outputs."""
    if not raw_message_list:
        logger.warning("No raw responses captured.")
        return
    finish_reason_counts = {}
    content_len_list = []
    for item in raw_message_list:
        body = item.get("response", {}).get("body", {})
        finish_reason = None
        completion_tokens = None
        prompt_tokens = None
        try:
            choices = body.get("choices", [])
            if choices:
                finish_reason = choices[0].get("finish_reason")
        except Exception:
            finish_reason = None
        try:
            usage = body.get("usage", {}) if isinstance(body, dict) else {}
            completion_tokens = usage.get("completion_tokens")
            prompt_tokens = usage.get("prompt_tokens")
        except Exception:
            completion_tokens = None
            prompt_tokens = None
        finish_reason = str(finish_reason)
        finish_reason_counts[finish_reason] = finish_reason_counts.get(finish_reason, 0) + 1
        content = _extract_text_from_response_payload(body)
        content_len_list.append(len(content or ""))
        if isinstance(completion_tokens, int):
            item.setdefault("_diag", {})["completion_tokens"] = completion_tokens
        if isinstance(prompt_tokens, int):
            item.setdefault("_diag", {})["prompt_tokens"] = prompt_tokens

    n = len(content_len_list)
    short_count = sum(1 for x in content_len_list if x < 64)
    zero_count = sum(1 for x in content_len_list if x == 0)
    logger.info(f"Response finish_reason counts: {finish_reason_counts}")
    logger.info(
        f"Response content length stats: min={min(content_len_list)}, "
        f"median={int(np.median(content_len_list))}, max={max(content_len_list)}, n={n}"
    )
    if short_count / max(1, n) >= 0.5:
        logger.warning(
            f"{short_count}/{n} responses are shorter than 64 chars. "
            "This often indicates provider-side truncation, stop-sequence mismatch, "
            "or incompatible max_tokens handling."
        )
    if zero_count > 0:
        logger.warning(
            f"{zero_count}/{n} responses are empty after extraction. "
            "Check raw_message_list response bodies for provider-specific schema."
        )
    completion_token_list = [
        item.get("_diag", {}).get("completion_tokens")
        for item in raw_message_list
        if isinstance(item.get("_diag", {}).get("completion_tokens"), int)
    ]
    prompt_token_list = [
        item.get("_diag", {}).get("prompt_tokens")
        for item in raw_message_list
        if isinstance(item.get("_diag", {}).get("prompt_tokens"), int)
    ]
    if completion_token_list:
        logger.info(
            f"Completion token stats: min={min(completion_token_list)}, "
            f"median={int(np.median(completion_token_list))}, "
            f"max={max(completion_token_list)}"
        )
    if prompt_token_list:
        logger.info(
            f"Prompt token stats: min={min(prompt_token_list)}, "
            f"median={int(np.median(prompt_token_list))}, "
            f"max={max(prompt_token_list)}"
        )


def extract_code(text: str) -> str:
    """
    Extract code enclosed by triple backticks (```) from a text string.
    Handles optional language specifiers and different code block formats.
    
    Args:
        text (str): Input text that may contain code blocks
        
    Returns:
        str: Extracted code block if found, empty string otherwise
    """
    import re
    # Match code blocks with optional language specifier
    pattern = r"```(?:\w*?\n)?(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        # Remove leading/trailing whitespace and potential language specifier
        code = matches[-1].strip()
        # Remove any remaining language specifier on first line
        code = re.sub(r'^\w+\s*\n', '', code, count=1)
        return code
    else:
        # Handle case with no code blocks by removing all triple backticks
        cleaned_code = text.replace('```', '').strip()
        return cleaned_code if cleaned_code else ''

 
def _classify_eval_item(eval_item: dict) -> str:
    """
    Classify one evaluation item into a coarse outcome bucket.
    """
    result = eval_item.get("result")
    error = eval_item.get("error")
    if error is not None:
        return "generation_or_eval_error"
    if result is None:
        return "no_result"

    try:
        if np.all(result):
            return "pass"
    except Exception:
        # Keep going and try detailed parsing below.
        pass

    # result is typically a tuple: (overall_pass, per_test_case_results)
    if isinstance(result, (list, tuple)) and len(result) >= 2:
        per_case = result[1]
        if isinstance(per_case, list):
            for case_result in per_case:
                if not isinstance(case_result, dict):
                    continue
                error_info = case_result.get("error_info")
                if error_info:
                    if "SyntaxError" in error_info:
                        return "syntax_error"
                    if "Traceback" in error_info:
                        return "runtime_error"
                    return "error_with_message"
            return "wrong_output_or_state"
    return "failed_unclassified"


@app.command()
def report(
    pkl_path: Annotated[
        Path,
        typer.Option(
            help="Path to llm evaluation pickle file generated by llm_evaluation.py."
        ),
    ] = None,
):
    """
    Load a saved llm evaluation pickle and print a concise report.
    """
    pkl_path = Path(pkl_path).expanduser()
    if pkl_path is None or not pkl_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    eval_result = data.get("eval_result", [])
    prompt_message_list = data.get("prompt_message_list", [])
    code_message_list = data.get("code_message_list", [])
    raw_message_list = data.get("raw_message_list", [])

    summary = {
        "pass": [],
        "generation_or_eval_error": [],
        "no_result": [],
        "syntax_error": [],
        "runtime_error": [],
        "error_with_message": [],
        "wrong_output_or_state": [],
        "failed_unclassified": [],
    }

    for idx, eval_item in enumerate(eval_result):
        category = _classify_eval_item(eval_item)
        summary[category].append(idx)

    total = len(eval_result)
    passed = len(summary["pass"])
    failed = total - passed
    pass_rate = (passed / total * 100.0) if total else 0.0

    typer.echo(f"=== LLM Evaluation Report ===")
    typer.echo(f"file: {pkl_path}")
    typer.echo(f"total_eval_items: {total}")
    typer.echo(f"pass: {passed}")
    typer.echo(f"fail: {failed}")
    typer.echo(f"pass_rate: {pass_rate:.2f}%")
    typer.echo("")
    typer.echo("failure_breakdown:")
    for key in (
        "generation_or_eval_error",
        "no_result",
        "syntax_error",
        "runtime_error",
        "error_with_message",
        "wrong_output_or_state",
        "failed_unclassified",
    ):
        typer.echo(f"- {key}: {len(summary[key])}")

    typer.echo("")
    typer.echo("artifact_sizes:")
    typer.echo(f"- prompt_message_list: {len(prompt_message_list)}")
    typer.echo(f"- code_message_list: {len(code_message_list)}")
    typer.echo(f"- raw_message_list: {len(raw_message_list)}")

    typer.echo("")
    typer.echo("sample_failed_indices (up to 20):")
    failed_indices = []
    for key in (
        "generation_or_eval_error",
        "no_result",
        "syntax_error",
        "runtime_error",
        "error_with_message",
        "wrong_output_or_state",
        "failed_unclassified",
    ):
        failed_indices.extend(summary[key])
    failed_indices = sorted(set(failed_indices))
    typer.echo(str(failed_indices[:20]))

@app.command()
def main(
    config_file: Annotated[Path, typer.Option()],
    api_doc_file: Annotated[Path, typer.Option()],
    target_llm: Annotated[str, typer.Option()], # gpt-4.1-nano, gpt-4.1
    parent_path: Annotated[Path, typer.Option()] = None,
    result_dir: Annotated[Path, typer.Option()] = None,
    openai_url: Annotated[str, typer.Option()] = "/v1/chat/completions",
    first_n: Annotated[int, typer.Option()] = -1,
    base_url: Annotated[
        Optional[str],
        typer.Option(help="OpenAI-compatible API base (e.g. https://api.openai.com/v1). Overrides BASE_URL / config env.open_source_base_url."),
    ] = None,
    openrouter: Annotated[
        Optional[bool],
        typer.Option(
            "--openrouter/--no-openrouter",
            help="Force OpenRouter mode. Uses OPENROUTER_API_KEY (or env.openrouter_api_key) and defaults base_url to https://openrouter.ai/api/v1.",
        ),
    ] = None,
    batch_mode: Annotated[
        Optional[bool],
        typer.Option(
            "--batch/--no-batch",
            help="Use OpenAI Batch API (only on api.openai.com). Use --no-batch for chat completions (vLLM, etc.). Overrides LLM_EVAL_BATCH_MODE / env.eval_batch_mode.",
        ),
    ] = None,
    max_tokens: Annotated[
        Optional[int],
        typer.Option(help="Max completion tokens for chat (non-batch) OpenAI-compatible calls."),
    ] = DEFAULT_MAX_TOKENS,
    print_prompt_sample: Annotated[
        bool,
        typer.Option(
            "--print-prompt-sample",
            help="Print one assembled evaluation prompt and exit (no API calls).",
        ),
    ] = False,
    prompt_index: Annotated[
        int,
        typer.Option(
            help="Prompt index to print when --print-prompt-sample is used.",
        ),
    ] = 0,
    prompt_max_chars: Annotated[
        Optional[int],
        typer.Option(
            help="Optional max number of characters to print for the prompt sample.",
        ),
    ] = None,
):
    with open(config_file, 'r') as file:
        config_dict = yaml.safe_load(file)
    with open(api_doc_file, 'r') as file:
        api_doc = file.read()
    env_config = config_dict.get("env", {})
    if parent_path is None:
        parent_path = config_dict["env"]["trace_save_path"]
    if result_dir is not None:
        Path(result_dir).expanduser().mkdir(parents=True, exist_ok=True)

    artifact_tag = _safe_artifact_tag(target_llm)

    logger.info(f"Evaluating {config_dict['task']} with {api_doc_file}, Loading StateEval...")
    stateful_bench = StateEval(parent_path, config_dict["task"], config_dict, api_doc)
    logger.info(f"StateEval loaded, {len(stateful_bench)} test cases in total.")
    if first_n > 0:
        logger.info(f"Evaluating the first {first_n} test cases.")
    if print_prompt_sample:
        available = len(stateful_bench) if first_n <= 0 else min(first_n, len(stateful_bench))
        if available <= 0:
            raise ValueError("No prompts available to print.")
        if prompt_index < 0 or prompt_index >= available:
            raise IndexError(
                f"prompt_index={prompt_index} out of range for available prompts [0, {available - 1}]"
            )
        prompt_text = stateful_bench[prompt_index]
        if prompt_max_chars is not None and prompt_max_chars > 0:
            prompt_text = prompt_text[:prompt_max_chars]
        typer.echo("=== Prompt Sample Begin ===")
        typer.echo(prompt_text)
        typer.echo("=== Prompt Sample End ===")
        return

    open_source_model_list = ['llama-4-scout-17b-16e-instruct', "qwen3-32b-fp8", "deepseek-r1-0528", "qwen25-coder-32b-instruct"]

    result = {}
    prompt_message_list = []
    code_message_list = []
    message_list = []
    if _should_use_openai_compat(target_llm, base_url):
        # OPENAI-compatible mode (official OpenAI Batch API or per-request chat.completions)
        use_openrouter = bool(openrouter) if openrouter is not None else _is_openrouter_model(target_llm)
        if use_openrouter:
            compat_base, compat_api_key = _resolve_openrouter_credentials(env_config, base_url)
            compat_headers = _resolve_openrouter_headers(env_config)
            if compat_headers:
                client = openai.OpenAI(
                    api_key=compat_api_key, base_url=compat_base, default_headers=compat_headers
                )
            else:
                client = openai.OpenAI(api_key=compat_api_key, base_url=compat_base)
        else:
            compat_base, compat_api_key = _resolve_openai_compat_credentials(env_config, base_url)
            client = openai.OpenAI(api_key=compat_api_key, base_url=compat_base)
        eval_batch_mode = _resolve_eval_batch_mode(env_config, batch_mode)
        chat_max_tokens = max_tokens
        if chat_max_tokens is None:
            te = os.environ.get("LLM_EVAL_MAX_TOKENS")
            if te is not None and str(te).strip():
                chat_max_tokens = int(te)
            else:
                chat_max_tokens = int(env_config.get("eval_max_tokens", DEFAULT_MAX_TOKENS))
        if not compat_api_key:
            if use_openrouter:
                raise ValueError(
                    "Missing OpenRouter API key. Set OPENROUTER_API_KEY "
                    "or env.openrouter_api_key in config."
                )
            raise ValueError(
                "Missing API key for OpenAI-compatible evaluation. Set API_KEY/OPENAI_API_KEY "
                "or env.openai_api_key in config."
            )
        if eval_batch_mode and not _openai_batch_supported(compat_base):
            logger.warning(
                "Batch mode was requested but base_url is not https://api.openai.com/v1; "
                "OpenAI Batch API is unavailable on this host. Using chat completions instead "
                "(set --no-batch explicitly to silence this)."
            )
            eval_batch_mode = False
        logger.info(
            f"OpenAI-compatible eval ({'OpenRouter' if use_openrouter else 'generic'}): "
            f"base_url={compat_base or '[default OpenAI]'}, batch_mode={eval_batch_mode}, "
            f"max_tokens={chat_max_tokens}"
        )
        request_id_list = []
        for idx, prompt in enumerate(stateful_bench):
            if idx >= first_n and first_n > 0:
                break
            request_id_list.append(f"request_{idx}")
            message_list.append([{
                "role": "user",
                "content": prompt
            }])
            prompt_message_list.append(prompt)

        if eval_batch_mode:
            jsonl_path = os.path.join(result_dir, "openai_requests.jsonl")
            _ensure_parent_dir(jsonl_path)
            generate_jsonl_for_openai(
                request_id_list=request_id_list,
                message_list=message_list,
                output_path=jsonl_path,
                model_type=target_llm,
                url=openai_url,
                max_tokens=chat_max_tokens,
            )
            batch_submit_info, batch_result_info = submit_batch_request_openai(
                client=client,
                input_file_path=jsonl_path,
                url=openai_url,
                description=f"benchmark {config_dict['task']} for {target_llm}",
            )
            logger.info("Batch submitted, waiting for completion...")
            wait_for_batch_completion(client, batch_submit_info.id)
            batch_result_info = client.batches.retrieve(batch_submit_info.id)
            file_response = client.files.content(batch_result_info.output_file_id)
            raw_message_list = []
            for i in file_response.iter_lines():
                info = json.loads(i)
                raw_message_list.append(info)
                response_body = info.get("response", {}).get("body", {})
                content = _extract_text_from_response_payload(response_body)
                code = extract_code(content)
                custom_id = info["custom_id"]
                item_id = int(custom_id.split("_")[1])
                code_message_list.append({
                    "item_id": item_id,
                    "code": code
                })
            result = {
                "raw_message_list": raw_message_list,
                "code_message_list": code_message_list,
                "batch_submit_info": batch_submit_info,
                "batch_result_info": batch_result_info,
                "prompt_message_list": prompt_message_list,
            }
        else:
            raw_message_list = []
            for idx, messages in enumerate(
                tqdm(message_list, desc="Evaluating (sync completions)")
            ):
                if _is_responses_api(openai_url):
                    response = client.responses.create(
                        model=target_llm,
                        input=messages,
                        max_output_tokens=chat_max_tokens,
                    )
                    content = _extract_text_from_response_payload(response)
                else:
                    response = _chat_completion_with_retry(
                        client=client,
                        target_llm=target_llm,
                        messages=messages,
                        chat_max_tokens=chat_max_tokens,
                    )
                body = _serialize_openai_payload(response)
                content = _extract_text_from_response_payload(body)
                info = {
                    "custom_id": request_id_list[idx],
                    "response": {"body": body},
                }
                raw_message_list.append(info)
                code = extract_code(content)
                code_message_list.append({
                    "item_id": idx,
                    "code": code
                })
            _log_response_diagnostics(raw_message_list)
            result = {
                "raw_message_list": raw_message_list,
                "code_message_list": code_message_list,
                "batch_submit_info": None,
                "batch_result_info": None,
                "prompt_message_list": prompt_message_list,
            }
    elif "gemini" in target_llm:
        # Google API
        google_api_key = config_dict["env"]["google_api_key"]
        from google import genai
        client = genai.Client(api_key=google_api_key)
        for idx, prompt in tqdm(enumerate(stateful_bench), total=len(stateful_bench), desc="Evaluating Google Gemini"):
            if idx >= first_n and first_n > 0:
                break
            response = client.models.generate_content(
                model=target_llm,
                contents=prompt
            )
            code = extract_code(response.text)
            message_list.append(response.text)
            code_message_list.append({
                "item_id": idx,
                "code": code
            })
            prompt_message_list.append(prompt)
        result["raw_message_list"] = message_list
        result["code_message_list"] = code_message_list
        result["prompt_message_list"] = prompt_message_list
    elif target_llm in open_source_model_list:
        # Open Source Model
        open_source_api_key = config_dict["env"]["open_source_api_key"]
        open_source_base_url = config_dict["env"]["open_source_base_url"]
        message_list = []
        client = openai.OpenAI(api_key=open_source_api_key, base_url=open_source_base_url)
        for idx, prompt in tqdm(enumerate(stateful_bench), total=len(stateful_bench), desc="Evaluating Open Source Model"):
            if idx >= first_n and first_n > 0:
                break
            prompt_message_list.append(prompt)
            response = client.completions.create(
                prompt=prompt,
                model=target_llm,
                max_tokens=DEFAULT_MAX_TOKENS,
            )
            code = extract_code(response.choices[0].text)
            message_list.append(response.choices[0].text)
            code_message_list.append({
                "item_id": idx,
                "code": code
            })
        result["raw_message_list"] = message_list
        result["code_message_list"] = code_message_list
        result["prompt_message_list"] = prompt_message_list
    result["eval_result"] = []
    
    logger.info(f"Evaluating {len(code_message_list)} test cases...")
    pkl_path = os.path.join(result_dir, f"llm_evaluation_{artifact_tag}.pkl")
    _persist_result(pkl_path, result)
    last_completed_eval_idx = -1
    try:
        for idx, item in enumerate(code_message_list):
            eval_item = {"result": None, "error": None}
            item_id = item["item_id"]
            code = item["code"]
            if len(code) == 0:
                eval_item["result"] = None
                eval_item["error"] = "No code generated"
                result["eval_result"].append(eval_item)
                last_completed_eval_idx = idx
                _persist_result(pkl_path, result)
                continue
            try:
                code = code.replace("exit()", "") # We do not allow exit.
                eval_result = stateful_bench.evaluate(item_id, code)
                eval_item["result"] = eval_result
            except KeyboardInterrupt:
                raise
            except BaseException:
                # Keep evaluation alive even if generated code triggers SystemExit/BaseException.
                error_info = traceback.format_exc()
                eval_item["error"] = error_info
            result["eval_result"].append(eval_item)
            last_completed_eval_idx = idx
            _persist_result(pkl_path, result)
    except KeyboardInterrupt:
        result["fatal_error"] = "KeyboardInterrupt"
        logger.warning("Interrupted by user. Saving partial evaluation results.")
        raise
    except BaseException:
        result["fatal_error"] = traceback.format_exc()
        logger.exception("Fatal error during evaluation loop. Saving partial results.")
    finally:
        result["last_completed_eval_idx"] = last_completed_eval_idx
        logger.info(f"Saving results to {pkl_path}")
        _persist_result(pkl_path, result)
        
        
    
if __name__ == "__main__":
    app()