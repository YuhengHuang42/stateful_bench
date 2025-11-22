# StateGen

StateGen is an automated framework for benchmarking large language models (LLMs) on **sequential, stateful API calls**. It generates coverage-guided execution traces, translates them into natural-language programming tasks, and evaluates LLM-produced programs against executable test suites.

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
- Storage for generated traces and prompts (set via config files, StateEval available in our Google Drive).
- A running backend for the task under study (e.g., launch `session-service` via Docker when benchmarking the session APIs).

> **Keep secrets out of version control.** Copy the example configs, fill in your own keys, and avoid committing them.

---

## Evaluation Benchmark

The dataset is available at: https://drive.google.com/drive/folders/1k_86uiFLU7MWcuS3YD8qwGLjK-Z8aP03?usp=sharing

Please download the dataset and configure the corresponding files under `config`. For evaluation purpose, you can skip the trace generation from step 1 to step 3 and directly evaluate the LLMs with the generated instructions.

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