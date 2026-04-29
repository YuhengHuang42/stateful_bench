import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    from typing import Annotated
except ImportError:  # Python < 3.9
    from typing_extensions import Annotated

import typer
import yaml
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

app = typer.Typer(help="Utilities for packaging StateGen eval data into Hugging Face datasets.")


def _iter_evaluator_files(trace_dir: Path) -> Iterable[Path]:
    yield from sorted(trace_dir.glob("evaluator_*.json"), key=lambda p: int(p.stem.split("_")[1]))


def _find_metadata(trace_dir: Path) -> dict:
    """Load the single *metadata.json file under the trace directory."""
    matches = sorted(trace_dir.glob("*metadata.json"))
    if not matches:
        raise FileNotFoundError(f"No metadata json found under {trace_dir}")
    if len(matches) > 1:
        typer.echo(f"[WARN] Multiple metadata files found, using {matches[0].name}", err=True)
    with matches[0].open("r") as fh:
        return json.load(fh)


def _load_agent_data_index(trace_dir: Path, agent_save_path: Optional[str]) -> Dict[str, dict]:
    candidates = [trace_dir / "agent_data" / "agent_data.json"]
    if agent_save_path:
        candidates.append(Path(agent_save_path).expanduser().resolve() / "agent_data.json")
    for candidate in candidates:
        if candidate.exists():
            with candidate.open("r") as fh:
                payload = json.load(fh)
            if isinstance(payload, dict):
                return {str(k): v for k, v in payload.items()}
    return {}


def _read_api_doc_text(repo_root: Path, api_doc_file: Optional[str]) -> str:
    if not api_doc_file:
        return ""
    doc_path = (repo_root / api_doc_file).resolve()
    if not doc_path.exists():
        typer.echo(f"[WARN] api_doc_file not found at {doc_path}; metadata.doc will be empty.", err=True)
        return ""
    return doc_path.read_text()


def _is_agent_conv_accepted(agent_conv: dict) -> bool:
    structured = agent_conv.get("last_evaluator_structured")
    if isinstance(structured, dict) and structured.get("verdict") == "ok":
        return True
    messages = agent_conv.get("evaluator_messages")
    if not messages:
        return False
    last_msg = messages[-1] if isinstance(messages[-1], dict) else {}
    content = (last_msg.get("content") or "").strip()
    if "OK" in content:
        return True
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(parsed, dict) and parsed.get("verdict") == "ok"


def _extract_prompt_from_agent_conv(agent_conv: dict) -> str:
    if not isinstance(agent_conv, dict) or not _is_agent_conv_accepted(agent_conv):
        return ""
    generator_messages = agent_conv.get("generator_messages")
    if not isinstance(generator_messages, list) or not generator_messages:
        return ""
    for msg in reversed(generator_messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return msg.get("content") or ""
    return ""


def _build_dataset_dict(
    trace_dir: Path,
    split: str,
    metadata_payload: Optional[dict] = None,
    agent_data_by_idx: Optional[Dict[str, dict]] = None,
) -> DatasetDict:
    trace_dir = trace_dir.expanduser().resolve()
    if metadata_payload is None:
        metadata_payload = _find_metadata(trace_dir)
    agent_data_by_idx = agent_data_by_idx or {}
    # Separate prompt_book to avoid duplicating it across every row
    prompt_book = metadata_payload.get("prompt_book", {})
    metadata_wo_prompt = {k: v for k, v in metadata_payload.items() if k != "prompt_book"}

    rows: List[Dict[str, str]] = []
    for evaluator_file in _iter_evaluator_files(trace_dir):
        idx = int(evaluator_file.stem.split("_")[1])
        with evaluator_file.open("r") as fh:
            evaluator_payload = json.load(fh)
        agent_conv = agent_data_by_idx.get(str(idx))
        prompt = prompt_book.get(str(idx), "")
        if not prompt and agent_conv is not None:
            prompt = _extract_prompt_from_agent_conv(agent_conv)
        rows.append(
            {
                "example_id": idx,
                "evaluator_json": json.dumps(evaluator_payload, separators=(",", ":")),
                "prompt": prompt,
                "metadata_json": json.dumps(metadata_wo_prompt, separators=(",", ":")),
                "agent_data_json": (
                    json.dumps(agent_conv, separators=(",", ":")) if agent_conv is not None else ""
                ),
            }
        )
    if not rows:
        raise RuntimeError(f"No evaluator files found in {trace_dir}")
    dataset = Dataset.from_list(rows).sort("example_id")
    return DatasetDict({split: dataset})


def _persist_dataset(dataset_dict: DatasetDict, output_dir: Optional[Path], repo_id: Optional[str], push: bool, private: bool):
    if output_dir:
        output_dir = output_dir.expanduser().resolve()
        dataset_dict.save_to_disk(str(output_dir))
        typer.echo(f"Dataset saved to {output_dir}")
    if push:
        if not repo_id:
            raise typer.BadParameter("repo_id is required when --push is enabled.")
        dataset_dict.push_to_hub(repo_id, private=private)
        typer.echo(f"Pushed dataset to https://huggingface.co/{repo_id}")
    if not output_dir and not push:
        typer.echo("No output chosen; specify --output-dir and/or --push to Hub.")


@app.command("pack")
def pack_to_hf(
    trace_dir: Annotated[Path, typer.Option(help="Directory containing evaluator_*.json files.")],
    output_dir: Annotated[Optional[Path], typer.Option(help="Directory to save the HF dataset locally.")]=None,
    repo_id: Annotated[Optional[str], typer.Option(help="Optional Hugging Face repo id (e.g. org/name).")]=None,
    split: Annotated[str, typer.Option(help="Dataset split name to store the examples under.")]="eval",
    push: Annotated[bool, typer.Option(help="Push the dataset to the Hugging Face Hub. Requires repo_id.")]=False,
    private: Annotated[bool, typer.Option(help="Mark the uploaded dataset as private.")]=False,
):
    """
    Package the locally generated evaluation artifacts into a Hugging Face Dataset.
    """
    dataset_dict = _build_dataset_dict(trace_dir, split)
    _persist_dataset(dataset_dict, output_dir, repo_id, push, private)


@app.command("bulk-pack")
def bulk_pack(
    config_files: Annotated[List[Path], typer.Argument(help="YAML config files whose trace_save_path should be packed.")],
    output_root: Annotated[Optional[Path], typer.Option(help="Optional base directory; per-task subfolders will be created.")]=None,
    repo_template: Annotated[Optional[str], typer.Option(help="Optional template for repo ids, e.g. your-org/stategen-{task}. Available fields: {task}, {config}.")]=None,
    split: Annotated[str, typer.Option(help="Dataset split name.")]="eval",
    push: Annotated[bool, typer.Option(help="Push each dataset to the Hub.")]=False,
    private: Annotated[bool, typer.Option(help="Mark uploaded datasets as private.")]=False,
):
    """
    Convenience helper to package multiple tasks (session/tensor/voice) in one go.
    """
    if output_root:
        output_root = output_root.expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)

    for config_file in config_files:
        with config_file.open("r") as fh:
            config = yaml.safe_load(fh)
        task = config.get("task", config_file.stem)
        trace_dir = Path(config["env"]["trace_save_path"])
        agent_data_by_idx = _load_agent_data_index(trace_dir, config.get("env", {}).get("agent_save_path"))
        try:
            metadata_payload = _find_metadata(trace_dir)
        except FileNotFoundError:
            repo_root = Path(__file__).resolve().parent.parent
            metadata_payload = {
                "evaluation_config": {"base_url": config.get("env", {}).get("base_url")} if task == "session" else {},
                "doc": _read_api_doc_text(repo_root, config.get("generation_config", {}).get("api_doc_file")),
                "prompt_book": {},
            }
            typer.echo(
                f"[WARN] No metadata file under {trace_dir}; building metadata from {config_file} and agent_data.json.",
                err=True,
            )
        dataset_dict = _build_dataset_dict(
            trace_dir=trace_dir,
            split=split,
            metadata_payload=metadata_payload,
            agent_data_by_idx=agent_data_by_idx,
        )
        dataset_name = config_file.stem
        output_dir = output_root / dataset_name if output_root else None
        repo_id = None
        if repo_template:
            repo_id = repo_template.format(task=task, config=dataset_name)
        typer.echo(f"[INFO] Packing {dataset_name} ({task}) from {trace_dir}")
        _persist_dataset(dataset_dict, output_dir, repo_id, push, private)


@app.command("materialize")
def materialize_from_hf(
    output_dir: Annotated[Path, typer.Option(help="Directory to recreate evaluator files into.")],
    repo_id: Annotated[Optional[str], typer.Option(help="Hugging Face repo id to download.")] = None,
    split: Annotated[str, typer.Option(help="Dataset split to download.")] = "eval",
    revision: Annotated[Optional[str], typer.Option(help="Optional git revision on the Hub.")]=None,
    local_dataset: Annotated[Optional[Path], typer.Option(help="Optional path to a dataset saved via save_to_disk.")]=None,
):
    """
    Download (or load) the Hugging Face dataset and recreate the on-disk layout required by llm_evaluation.py.
    """
    if not repo_id and not local_dataset:
        raise typer.BadParameter("Provide either repo_id or local_dataset.")

    if repo_id:
        dataset = load_dataset(repo_id, split=split, revision=revision)
    else:
        dataset_dict = load_from_disk(str(local_dataset.expanduser().resolve()))
        if isinstance(dataset_dict, DatasetDict):
            if split not in dataset_dict:
                raise KeyError(f"Split '{split}' not found in dataset saved at {local_dataset}")
            dataset = dataset_dict[split]
        else:
            dataset = dataset_dict

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_written = False

    for row in dataset:
        idx = row["example_id"]
        evaluator_data = json.loads(row["evaluator_json"])
        evaluator_path = output_dir / f"evaluator_{idx}.json"
        with evaluator_path.open("w") as fh:
            json.dump(evaluator_data, fh)

        metadata_json = row.get("metadata_json")
        if metadata_json and not metadata_written:
            metadata_payload = json.loads(metadata_json)
            metadata_path = output_dir / f"{split}_metadata.json"
            with metadata_path.open("w") as fh:
                json.dump(metadata_payload, fh)
            metadata_written = True

    typer.echo(f"Recreated {len(dataset)} evaluators under {output_dir} (metadata written: {metadata_written})")


if __name__ == "__main__":
    app()

