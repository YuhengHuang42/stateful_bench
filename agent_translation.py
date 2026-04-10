from enum import Enum
import os
import json
from loguru import logger
import typer
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from typing import Annotated
from pathlib import Path

from Sgenerator.agent import MultiAgentTrans
from Sgenerator.utils import load_program_intent_summaries_index

app = typer.Typer(pretty_exceptions_show_locals=False, pretty_exceptions_short=False)
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

@app.command()
def main(
    config_file: Annotated[Path, typer.Option()],
    trace_save_path: Annotated[Path, typer.Option()] = None,
    agent_save_path: Annotated[Path, typer.Option()] = None,
):
    with open(config_file, 'r') as file:
        config_dict = yaml.safe_load(file)
    if trace_save_path is None:
        trace_save_path = config_dict["env"]["trace_save_path"]
    if agent_save_path is None:
        agent_save_path = config_dict["env"]["agent_save_path"]
    
    if not os.path.exists(agent_save_path):
        os.makedirs(agent_save_path)
    env_config = config_dict.get("env", {})
    agent_config = config_dict.get("agent_config", {})
    base_url = os.environ.get("BASE_URL") or env_config.get("open_source_base_url")
    api_key = (
        os.environ.get("API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or env_config.get("openai_api_key", "")
    )
    translation_model = (
        os.environ.get("TRANSLATION_MODEL")
        or os.environ.get("MODEL")
        or agent_config.get("translation_model")
        or agent_config.get("model", "gpt-4.1")
    )
    evaluator_model = (
        os.environ.get("EVALUATOR_MODEL")
        or agent_config.get("evaluator_model")
        or translation_model
    )
    shared_max_tokens = agent_config.get("max_tokens")
    if os.environ.get("AGENT_MAX_TOKENS", "").strip():
        shared_max_tokens = int(os.environ["AGENT_MAX_TOKENS"])
    shared_chat_url = (
        os.environ.get("AGENT_CHAT_COMPLETIONS_URL")
        or agent_config.get("chat_completions_url", "/v1/chat/completions")
    )
    batch_mode_env = os.environ.get("TRANSLATION_BATCH_MODE")
    if batch_mode_env is not None and str(batch_mode_env).strip():
        translation_batch_mode = str(batch_mode_env).strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
    else:
        translation_batch_mode = bool(agent_config.get("batch_mode", False))
    temperature = agent_config.get("temperature", 1.0)
    te = os.environ.get("TRANSLATION_TEMPERATURE")
    if te is not None and str(te).strip():
        temperature = float(te)
    uao_env = os.environ.get("TRANSLATION_USE_API_STRUCTURED_OUTPUT")
    if uao_env is not None and str(uao_env).strip():
        use_api_structured_output = str(uao_env).strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
    else:
        use_api_structured_output = bool(
            agent_config.get("use_api_structured_output", True)
        )
    if not api_key:
        raise ValueError(
            "Missing API key. Set API_KEY/OPENAI_API_KEY in environment or openai_api_key in config."
        )
    task = config_dict["task"]
    max_iterations = config_dict["agent_config"]["max_iterations"]
    assert task in ["session", "tensor", "voice"]
    if task == "session":
        from Sgenerator.session_state import SessionEvaluator, SESSION_SUMMARY_PROMPT
        trace_evaluator_class = SessionEvaluator
        summary_prompt = SESSION_SUMMARY_PROMPT
    elif task == "tensor":
        from Sgenerator.tensor_state import TensorEvaluator, TENSOR_SUMMARY_PROMPT
        trace_evaluator_class = TensorEvaluator
        summary_prompt = TENSOR_SUMMARY_PROMPT
    elif task == "voice":
        from Sgenerator.voice_state import VoiceEvaluator, VOICE_SUMMARY_PROMPT
        trace_evaluator_class = VoiceEvaluator
        summary_prompt = VOICE_SUMMARY_PROMPT
    
    trace_evaluator_book = {}
    for file in os.listdir(trace_save_path):
        if file.endswith('.json') and 'evaluator' in file:
            file_path = os.path.join(trace_save_path, file)
            trace_evaluator = trace_evaluator_class.load(file_path)
            idx = int(file.split('_')[1].split('.')[0])
            trace_evaluator_book[idx] = trace_evaluator
    
    logger.info(f"Loaded {len(trace_evaluator_book)} trace evaluators.")
    trace_root = os.path.abspath(str(trace_save_path))
    intent_by_trace_idx = load_program_intent_summaries_index(trace_root)
    if intent_by_trace_idx:
        logger.info(
            f"Found program_intent_summaries under trace path; loaded "
            f"{len(intent_by_trace_idx)} non-empty summaries for the generator."
        )

    program_info = []
    for idx in sorted(trace_evaluator_book.keys()):
        row = {
            "init_block": trace_evaluator_book[idx].test_cases[0]["program_info"]["init_local_str"],
            "init_load_str": trace_evaluator_book[idx].test_cases[0]["program_info"]["init_load_str"],
            "program": trace_evaluator_book[idx].test_cases[0]["program"],
        }
        summary = intent_by_trace_idx.get(idx)
        if summary:
            row["program_intent_summary"] = summary
        program_info.append(row)
    
    client = OpenAI(base_url=base_url, api_key=api_key)
    logger.info(
        f"Translation LLM: model={translation_model}, evaluator model={evaluator_model}, "
        f"base_url={base_url or '[default]'}, "
        f"batch_mode={translation_batch_mode}, temperature={temperature}, "
        f"use_api_structured_output={use_api_structured_output}"
    )
    agent_manager = MultiAgentTrans(
        program_info=program_info,
        max_iterations=max_iterations,
        application_description=summary_prompt,
        openai_client=client,
        max_tokens=shared_max_tokens,
        model_type=translation_model,
        url=shared_chat_url,
        evaluator_agent_params={"model_type": evaluator_model},
        batch_mode=translation_batch_mode,
        temperature=temperature,
        use_api_structured_output=use_api_structured_output,
    )
    
    agent_manager.interact_loop(program_info, agent_save_path)
    data = agent_manager.save_agent_data()
    save_path = os.path.join(agent_save_path, "agent_data.json")
    with open(save_path, "w") as f:
        json.dump(data, f)

if __name__ == "__main__":
    app()