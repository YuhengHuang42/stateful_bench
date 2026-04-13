import typer
from typing import Annotated, Any, Dict
from pathlib import Path
import yaml
import pickle
import os
from loguru import logger
from tqdm import tqdm

from Sgenerator.state import generate_and_collect_test_case, OccurenceBook

app = typer.Typer(pretty_exceptions_show_locals=False, pretty_exceptions_short=False)

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
        typer.Option(
            help=(
                "Path to persistent OccurenceBook JSON. "
                "Created if missing; reloaded across runs."
            )
        ),
    ] = None,
):
    with open(config_file, "r") as file:
        config_dict = yaml.safe_load(file)
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
        occurence_book_path = os.path.join(str(trace_save_path), "occurence_book.json")

    occurence_book = OccurenceBook.load(str(occurence_book_path))

    if occurence_book.has_pending_discards():
        logger.info(
            f"Applying {len(occurence_book.pending_discards)} pending "
            "discards from previous LLM review."
        )
        removed = occurence_book.apply_discards()
        logger.info(f"Removed {len(removed)} fully-zeroed transition pairs.")

    evaluation_config: Dict[str, Any] = {}
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

    evaluator_book: Dict[int, Any] = {}
    occ_book_diff_recorder: Dict[int, Any] = {}
    idx = 0
    consecutive_failures = 0
    total_failures = 0
    max_consecutive_failures = generation_config.get("max_consecutive_failures", 20)
    enable_coverage = generation_config.get("enable_coverage", True)
    if not enable_coverage:
        logger.info("Coverage-guided trace generation is disabled.")

    pbar = tqdm(total=num_of_tests, desc="Generating", unit="test")
    while idx < num_of_tests:
        evaluator, is_success, new_occurence_book, occ_diff = generate_and_collect_test_case(
            schema_class=schema_class,
            random_init_class=random_init_class,
            evaluator_class=evaluator_class,
            trace_config=generation_config["trace_config"],
            evaluation_config=evaluation_config,
            num_of_apis=num_of_apis,
            control_position_candidate=control_position_candidate,
            occurence_book=occurence_book,
            enable_coverage=enable_coverage,
        )
        if is_success:
            occurence_book = new_occurence_book
            occ_book_diff_recorder[idx] = occ_diff
            evaluator_book[idx] = evaluator
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

    for eidx in evaluator_book:
        evaluator_save_path = os.path.join(str(trace_save_path), f"evaluator_{eidx}.json")
        evaluator_book[eidx].store(evaluator_save_path)

    occurence_book.save(str(occurence_book_path))
    logger.info(f"OccurenceBook saved to {occurence_book_path}")

    metadata_save_path = os.path.join(str(trace_save_path), "metadata.pkl")
    with open(metadata_save_path, "wb") as file:
        pickle.dump({
            "occurence_book": occurence_book.to_dict(),
            "config": config_dict,
            "occ_book_diff_recorder": occ_book_diff_recorder,
            "generation_stats": {
                "succeeded": idx,
                "requested": num_of_tests,
                "total_failures": total_failures,
                "max_consecutive_failures": max_consecutive_failures,
            },
        }, file)

if __name__ == "__main__":
    app()