from enum import Enum
import os
import json
from loguru import logger
import typer
import yaml
import openai
from typing import Annotated
from pathlib import Path

from Sgenerator.agent import MultiAgent

app = typer.Typer(pretty_exceptions_show_locals=False, pretty_exceptions_short=False)

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
        
    api_key = config_dict["env"]["api_key"]
    task = config_dict["task"]
    max_iterations = config_dict["agent_config"]["max_iterations"]
    assert task in ["session"]
    if task == "session":
        from Sgenerator.session_state import SessionEvaluator, SESSION_SUMMARY_PROMPT
        trace_evaluator_class = SessionEvaluator
        summary_prompt = SESSION_SUMMARY_PROMPT
    
    trace_evaluator_book = {}
    for file in os.listdir(trace_save_path):
        if file.endswith('.json'):
            file_path = os.path.join(trace_save_path, file)
            trace_evaluator = trace_evaluator_class.load(file_path)
            idx = int(file.split('_')[1].split('.')[0])
            trace_evaluator_book[idx] = trace_evaluator
    
    program_info = []
    for idx in sorted(trace_evaluator_book.keys()):
        program_info.append({
            "init_block": trace_evaluator_book[idx].test_cases[0]["program_info"]["init_local_str"],
            "init_load_info": trace_evaluator_book[idx].test_cases[0]["program_info"]["init_load_info"],
            "program": trace_evaluator_book[idx].test_cases[0]["program"]
        })
    
    client = openai.OpenAI(api_key=api_key)
    agent_manager = MultiAgent(
        program_info = program_info,
        max_iterations = max_iterations,
        application_description = summary_prompt,
        openai_client = client
    )
    
    agent_manager.interact_loop(program_info, agent_save_path)
    data = agent_manager.save_agent_data()
    save_path = os.path.join(agent_save_path, "agent_data.json")
    with open(save_path, "w") as f:
        json.dump(data, f)

if __name__ == "__main__":
    app()