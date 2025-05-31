# StateGen

This is the related code relevant to the work StateGen: An Automated Framework for Evaluating LLMs on Sequential API Call.

## File Structure

```
├── agent_translation.py
├── api_doc
├── config
├── EMB
├── llm_evaluation.py
├── README.md
├── session-service
├── Sgenerator
│   ├── agent.py
│   ├── __init__.py
│   ├── session_state.py
│   ├── state.py
│   ├── tensor_state.py
│   ├── utils.py
│   ├── voice_lab.py
│   └── voice_state.py
└── trace_generation.py
```

- agent_translation.py: This file is used to translate the generated code to natural language description.

- api_doc: This directory contains the API documentation for the LLM.

- config: This directory contains the configuration files for experiments

- llm_evaluation.py: This file is used to evaluate the LLM on the sequential API call.

- session-service: This directory contains the Session-service.

- Sgenerator: This directory contains the Sgenerator.
  - agent.py: Core functionalities for multi-agent system.
  - state.py: State class for the Sgenerator.
  - voice_lab.py: supporting backend for the mock testing of ElevenLabs MCP.
  - voice_state.py: State class for the ElevenLabs MCP.
  - tensor_state.py: State class for the Pytorch Tensor Manipulation.
  - session_state.py: State class for the Session-service.
  - utils.py: Utility functions for the Sgenerator.


## Benchmark Datasets

Will be released on Google Drive.

## Application

1. Session-service

2. Pytorch Tensor Manipulation

3. ElevenLabs MCP

For MCP application, we follow similar styles here :https://www.philschmid.de/mcp-example-llama