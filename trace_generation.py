import typer
from typing import Annotated
from pathlib import Path
import yaml
import pickle
import os

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
    
    if config_dict["task"] == "session":
        from Sgenerator.session_state import SessionEvaluator, SessionRandomInitializer, SessionVariableSchema
        base_url = config_dict["env"]["base_url"]
        schema_class = SessionVariableSchema
        random_init_class = SessionRandomInitializer
        evaluator_class = SessionEvaluator
    else:
        raise ValueError(f"Task {config_dict['task']} is not supported.")
        
    occurence_book = {}
    evaluator_book = {}
    occ_book_diff_recorder = {}
    idx = 0
    while idx < num_of_tests:
        evaluator, is_success, new_occurence_book, occ_diff = generate_and_collect_test_case(
            schema_class = schema_class,
            random_init_class = random_init_class,
            evaluator_class = evaluator_class,
            trace_config = generation_config["trace_config"],
            base_url = base_url,
            num_of_apis = num_of_apis,
            control_position_candidate = control_position_candidate,
            occurence_book=occurence_book
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