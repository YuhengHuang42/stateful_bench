# StateEval

StateGen is an automated framework for benchmarking large language models (LLMs) on **sequential, stateful API calls**. It generates coverage-guided execution traces, translates them into natural-language programming tasks, and evaluates LLM-produced programs against executable test suites. StateEval, containing 120 test cases, is a benchmark created based on StateGen and verified manually.

**News:** We now support the evaluation of the benchmark from HuggingFace. You can use the `StateEvalHF` class to load the dataset from HuggingFace.
```python
data = state.StateEvalHF(task="session", hf_repo_id="yuhenghuang/StateEval", hf_split="session")
```
The `StateEvalHF` class is a HuggingFace versiion of the `StateEval` class. It is used to load the dataset from HuggingFace.

---

## Repository Layout

| Path | Description |
| --- | --- |
| `trace_generation.py` | CLI that synthesizes executable traces for session/tensor/voice tasks. |
| `agent_translation.py` | Multi-agent translator that turns traces into human-readable task descriptions. |
| `llm_evaluation.py` | Batch evaluator that feeds prompts to LLMs and scores returned code. |
| `Sgenerator/` | Core trace generation + evaluation logic (schemas, evaluators, utilities, agents). |
| `config/` | Ready-to-edit YAML templates controlling tasks, API keys, and generation budgets. |
| `api_doc/` | Canonical API references provided to LLMs during evaluation. |
| `session-service/` | Spring Boot reference backend used by the session task (Docker + Maven options). |

---

## Prerequisites

- Python 3.10+ with `pip install typer loguru numpy pyyaml openai tqdm torch google-genai`.
- (Optional) Access credentials for the models you plan to evaluate (OpenAI, Google Gemini, or supported open-source gateways).
- Storage for generated traces and prompts (set via config files, StateEval available through HuggingFace).
- A running backend for the task under study (e.g., launch `session-service` via Docker when benchmarking the session APIs).
- Pytorch version should be generally OK for the tensor task. But we only test the code on Pytorch 2.4.1.

> **Keep secrets out of version control.** Copy the example configs, fill in your own keys, and avoid committing them.

---

## Evaluation Benchmark

We have two implementations of the benchmark. One is the local version, which is stored in a folder. The other is the HuggingFace version, which is stored in a HuggingFace dataset. The former one is used to help you prepare your locally generated dataset for the evaluation. And the latter one is used to evaluate the LLMs on the benchmark from HuggingFace.

### Publishing / consuming the dataset via Hugging Face

You can now distribute the exact same evaluation artifacts via the Hugging Face Hub and load them with `datasets.load_dataset`.

1. **Package a single dataset directory**
   ```
   python scripts/hf_dataset.py pack \
     --trace-dir /abs/path/outputs/session_traces \
     --output-dir /abs/path/outputs/session_traces_hf \
     --repo-id your-hf-org/stategen-session \
     --push  # optional
   ```
   - `trace-dir` must contain the `evaluator_*.json` files and `agent_data/agent_data.json`.
   - When `--push` is omitted the dataset is only saved locally (`save_to_disk`). Add `--push` to upload to the Hub (requires `huggingface-cli login` beforehand). Use `--private` if the repo should be private.
   - Notice: you need to prepare a metadata file under the trace directory. For example:
   ```
    voice_metadata = {
      "evaluation_config": {},
      "doc": api_doc,
      "prompt_book": voice_bench.prompt_book
      
  }
   ```
   The prompt_book can be obtained by StateEval class (it is actually obtained from the agent_data.json returned by our translation agent). For example:
   ```
   voice_bench = StateEval(parent_path, config_dict["task"], config_dict, api_doc)
   prompt_book = voice_bench.prompt_book
   ```


2. **Use the dataset directly in Python (optional)**
   ```python
   import Sgenerator.state as state
   data = state.StateEvalHF(task="session", hf_repo_id="yuhenghuang/StateEval", hf_split="session")
   ```
   StateEvalHF class is a HuggingFace versiion of StateEval class. It is used to load the dataset from HuggingFace.



We load the dataset through `StateEval` class, which is defined in `Sgenerator/state.py`. It is used in `llm_evaluation.py`, but you can also use in your own scripts as well.

```python
stateful_bench = StateEval(parent_path, config_dict["task"], config_dict, api_doc)
```

The dataset can be directly iterated through. And the evaluation is done by calling the `evaluate` method.

## End-to-End Workflow

1. **Configure a task** (session, tensor, or voice) via YAML.
2. **Generate traces** to harvest executable programs plus test cases.
3. **Translate traces** into natural-language task descriptions.
4. **Evaluate LLMs** by prompting them with the descriptions + API docs and executing their code.

Each CLI is a Typer app, so `python <script>.py --help` lists the full set of flags.

### 1. Configure a task

Copy one of the templates in `config/` (e.g., `example_session_generation.yaml`) and edit the sections below:

- `task`: `session`, `tensor`, or `voice`.
- `env`:
  - `trace_save_path` / `agent_save_path`: absolute folders where artifacts land.
  - Model credentials (`openai_api_key`, `google_api_key`, `open_source_*`) as needed.
  - Task-specific knobs (`base_url` for session service, etc.).
- `generation_config`:
  - `num_of_tests`, `num_of_apis`: coverage budget.
  - `control_position_candidate`: where if/else branches may appear.
  - `trace_config`: initialization ranges and randomization hints.
  - Optional `enable_coverage` to toggle coverage-guided selection (defaults to `True`).
- `agent_config`: currently `max_iterations` for the generator/evaluator conversation loop.

Keep example files (e.g., `config/example_voice_generation.yaml`) as references but do not reuse the placeholder secrets.

### 2. Generate coverage-guided traces

```
python trace_generation.py \
  --config-file /abs/path/config/session_generation.yaml \
  --trace-save-path /abs/path/outputs/session_traces
```

Outputs per test include:

- `evaluator_<idx>.json`: serialized evaluator with test cases and metadata.
- `metadata.pkl`: coverage counts + config snapshot for reproducibility.

Ensure supporting services are reachable (e.g., start `session-service` if `task=session`).

### 3. Translate traces into instructions

```
python agent_translation.py \
  --config-file /abs/path/config/session_generation.yaml \
  --trace-save-path /abs/path/outputs/session_traces \
  --agent-save-path /abs/path/outputs/session_agents
```

The script loads every evaluator, spins up a generator/evaluator agent pair per trace, and iteratively refines the textual description. Artifacts:

- `round_*.jsonl`: batched OpenAI requests/responses (for auditability).
- `agent_data/agent_data.json`: consolidated prompts plus agent dialogue, consumed later by `StateEval`.

### 4. Evaluate LLMs

```
python llm_evaluation.py \
  --config-file /abs/path/config/session_generation.yaml \
  --api-doc-file api_doc/session-service.json \
  --target-llm gpt-4.1 \
  --result-dir /abs/path/outputs/session_eval \
  --first-n 50  # optional
```

What happens:

1. `StateEval` loads traces + agent data.
2. Prompts are built as `LLM_EVAL_PROMPT + API doc + task description`.
3. Requests are sent in batches (OpenAI), sequentially (Gemini), or via the configured OSS endpoint.
4. Returned code is sanitized, executed inside the evaluator, and scored.

Results are stored in `llm_evaluation_<model>.pkl` alongside raw prompts, model outputs, and execution verdicts.

---

## Supporting Assets

- `Sgenerator/session_state.py`, `tensor_state.py`, `voice_state.py`: schema implementations defining variables, transitions, and evaluators for each task family.
- `Sgenerator/voice_lab.py`: lightweight mock harness for ElevenLabs’ MCP-like workflow.
- `api_doc/`: curated API specs (Markdown or JSON) that match each task; pass the relevant file to `llm_evaluation.py`.
- `session-service/`: full Spring Boot project for the session benchmark. Use `docker-compose up` or follow the included README for Maven-based runs.