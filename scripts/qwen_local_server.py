import argparse
import time
import uuid
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL
    messages: List[ChatMessage]
    max_tokens: int = Field(default=8192, ge=1, le=16384)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    stream: bool = False


def build_app(
    model_id: str,
    device_map: str,
    torch_dtype: str,
) -> FastAPI:
    app = FastAPI(title="Qwen Local Inference Server")

    if torch_dtype == "auto":
        dtype = "auto"
    elif torch_dtype == "float16":
        dtype = torch.float16
    elif torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
    )
    model.eval()

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "model": model_id}

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest) -> Dict[str, Any]:
        if req.stream:
            raise HTTPException(
                status_code=400,
                detail="stream=true is not implemented in this local server.",
            )

        if req.model != model_id:
            raise HTTPException(
                status_code=400,
                detail=f"Requested model {req.model} does not match loaded model {model_id}.",
            )

        if not req.messages:
            raise HTTPException(status_code=400, detail="messages cannot be empty.")

        try:
            prompt = tokenizer.apply_chat_template(
                [m.model_dump() for m in req.messages],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            prompt_tokens = int(inputs["input_ids"].shape[-1])

            do_sample = req.temperature > 0.0
            temperature = max(req.temperature, 1e-5)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=req.max_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=req.top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated_ids = output_ids[0][prompt_tokens:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            completion_tokens = int(generated_ids.shape[-1])

            created = int(time.time())
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": created,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Qwen chat server.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--device-map",
        default="auto",
        help='Passed to transformers.from_pretrained (e.g. "auto", "cuda:0").',
    )
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["auto", "float16", "bfloat16"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app(
        model_id=args.model_id,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
