"""
This script is used to evaluate the LLM performance on the (local version of) stateful benchmark.
"""
import typer
from Sgenerator.state import StateEval
from typing import Annotated
from pathlib import Path
import yaml
from loguru import logger
import openai
import json
import traceback
import pickle
import os
from tqdm import tqdm
# StateEval: parent_path: str, task: str, config_dict: dict, api_doc: str

from Sgenerator.utils import generate_jsonl_for_openai, submit_batch_request_openai


app = typer.Typer(pretty_exceptions_show_locals=False, pretty_exceptions_short=False)

def wait_for_batch_completion(client, batch_id, wait_time=10):
    """Poll batch status until completion"""
    import time
    while True:
        batch_status = client.batches.retrieve(batch_id)
        if batch_status.status in ['completed', 'failed', 'cancelled']:
            return
        time.sleep(wait_time)  # Check every 10 seconds

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

 
@app.command()
def main(
    config_file: Annotated[Path, typer.Option()],
    api_doc_file: Annotated[Path, typer.Option()],
    target_llm: Annotated[str, typer.Option()], # gpt-4.1-nano, gpt-4.1
    parent_path: Annotated[Path, typer.Option()] = None,
    result_dir: Annotated[Path, typer.Option()] = None,
    openai_url: Annotated[str, typer.Option()] = "/v1/chat/completions",
    first_n: Annotated[int, typer.Option()] = -1,
):
    with open(config_file, 'r') as file:
        config_dict = yaml.safe_load(file)
    with open(api_doc_file, 'r') as file:
        api_doc = file.read()
    if parent_path is None:
        parent_path = config_dict["env"]["trace_save_path"]
    
    logger.info(f"Evaluating {config_dict['task']} with {api_doc_file}, Loading StateEval...")
    stateful_bench = StateEval(parent_path, config_dict["task"], config_dict, api_doc)
    logger.info(f"StateEval loaded, {len(stateful_bench)} test cases in total.")
    if first_n > 0:
        logger.info(f"Evaluating the first {first_n} test cases.")

    open_source_model_list = ['llama-4-scout-17b-16e-instruct', "qwen3-32b-fp8", "deepseek-r1-0528", "qwen25-coder-32b-instruct"]

    result = {}
    prompt_message_list = []
    code_message_list = []
    message_list = []
    if "gpt" in target_llm:
        # OPENAI mode
        # Generate OpenAI batch requests
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
        jsonl_path = os.path.join(result_dir, "openai_requests.jsonl")
        generate_jsonl_for_openai(
            request_id_list=request_id_list,
            message_list=message_list,
            output_path=jsonl_path,
            model_type=target_llm
        )
        
        # Initialize OpenAI client and submit batch
        client = openai.OpenAI(api_key=config_dict["env"]["openai_api_key"])
        batch_submit_info, batch_result_info = submit_batch_request_openai(
            client=client,
            input_file_path=jsonl_path,
            url=openai_url,
            description=f"benchmark {config_dict['task']} for {target_llm}",
        )
        
        logger.info(f"Batch submitted, waiting for completion...")
        wait_for_batch_completion(client, batch_submit_info.id)
        batch_result_info = client.batches.retrieve(batch_submit_info.id)
        file_response = client.files.content(batch_result_info.output_file_id)
        
        raw_message_list = []
        for i in file_response.iter_lines():
            info = json.loads(i)
            raw_message_list.append(info)
            code = extract_code(info["response"]["body"]["choices"][0]["message"]["content"])
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
            "prompt_message_list": prompt_message_list
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
                max_tokens=4096,
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
    # Temporarily save the result
    with open(os.path.join(result_dir, f"llm_evaluation_{target_llm}.pkl"), "wb") as file:
        pickle.dump(result, file)
    for item in code_message_list:
        eval_item = {"result": None, "error": None}
        item_id = item["item_id"]
        code = item["code"]
        if len(code) == 0:
            eval_item["result"] = None
            eval_item["error"] = "No code generated"
            result["eval_result"].append(eval_item)
            continue
        try:
            code = code.replace("exit()", "") # We do not allow exit.
            eval_result = stateful_bench.evaluate(item_id, code)
            eval_item["result"] = eval_result
        except Exception as e:
            error_info = traceback.format_exc()
            eval_item["error"] = error_info
        result["eval_result"].append(eval_item)
    
    logger.info(f"Saving results to {os.path.join(result_dir, f'llm_evaluation_{target_llm}.pkl')}")
    with open(os.path.join(result_dir, f"llm_evaluation_{target_llm}.pkl"), "wb") as file:
        pickle.dump(result, file)
        
        
    
if __name__ == "__main__":
    app()