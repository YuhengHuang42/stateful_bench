import typer
from typing import Annotated
from pathlib import Path
import yaml
import pickle
import os
from loguru import logger

from Sgenerator.state import generate_and_collect_test_case

app = typer.Typer(pretty_exceptions_show_locals=False, pretty_exceptions_short=False)

@app.command()
def main(
    config_file: Annotated[Path, typer.Option()],
    trace_save_path: Annotated[Path, typer.Option()] = None,
):
    with open(config_file, 'r') as file:
        config_dict = yaml.safe_load(file)
    generation_config = config_dict["generation_config"]
    
    num_of_apis = generation_config["num_of_apis"]
    control_position_candidate = generation_config["control_position_candidate"]
    num_of_tests = generation_config["num_of_tests"]

    if trace_save_path is None:
        trace_save_path = config_dict["env"]["trace_save_path"]
    
    evaluation_config = {}
    if config_dict["task"] == "session":
        from Sgenerator.session_state import SessionEvaluator, SessionRandomInitializer, SessionVariableSchema
        base_url = config_dict["env"]["base_url"]
        evaluation_config["base_url"] = base_url
        schema_class = SessionVariableSchema
        random_init_class = SessionRandomInitializer
        evaluator_class = SessionEvaluator
    elif config_dict["task"] == "tensor":
        from Sgenerator.tensor_state import TensorEvaluator, TensorRandomInitializer, TensorVariableSchema
        schema_class = TensorVariableSchema
        random_init_class = TensorRandomInitializer
        evaluator_class = TensorEvaluator
    elif config_dict["task"] == "voice":
        from Sgenerator.voice_state import VoiceEvaluator, VoiceRandomInitializer, VoiceVariableSchema
        schema_class = VoiceVariableSchema
        random_init_class = VoiceRandomInitializer
        evaluator_class = VoiceEvaluator
    else:
        raise ValueError(f"Task {config_dict['task']} is not supported.")
        
    occurence_book = {}
    evaluator_book = {}
    occ_book_diff_recorder = {}
    idx = 0
    enable_coverage = generation_config["enable_coverage"] if "enable_coverage" in generation_config else True
    if enable_coverage == False:
        logger.info("Coverage-guided trace generation is disabled.")
    while idx < num_of_tests:
        evaluator, is_success, new_occurence_book, occ_diff = generate_and_collect_test_case(
            schema_class = schema_class,
            random_init_class = random_init_class,
            evaluator_class = evaluator_class,
            trace_config = generation_config["trace_config"],
            evaluation_config = evaluation_config,
            num_of_apis = num_of_apis,
            control_position_candidate = control_position_candidate,
            occurence_book=occurence_book,
            enable_coverage=enable_coverage
        )
        if is_success:
            occurence_book = new_occurence_book
            occ_book_diff_recorder[idx] = occ_diff
            evaluator_book[idx] = evaluator
            idx += 1
    
    for idx in evaluator_book:
        evaluator_save_path = os.path.join(trace_save_path, f"evaluator_{idx}.json")
        evaluator_book[idx].store(evaluator_save_path)
    metadata_save_path = os.path.join(trace_save_path, "metadata.pkl")
    with open(metadata_save_path, 'wb') as file:
        pickle.dump({
            "occurence_book": occurence_book,
            "config": config_dict,
            "occ_book_diff_recorder": occ_book_diff_recorder
        }, file)

if __name__ == "__main__":
    app()