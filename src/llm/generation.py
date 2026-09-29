"""LLM backends used by the API and the evaluation script.

  * HFBackend        - Hugging Face model (+ optional LoRA/QLoRA adapter); needs a GPU (or lots of patience on CPU)
  * ExtractiveBackend - no LLM at all: returns the retrieved context. Lets the API run/test anywhere.
"""
from __future__ import annotations

import contextlib
import time
from pathlib import Path

from src import config


class ExtractiveBackend:
    name = "extractive (no LLM)"
    has_adapter = False

    def generate(self, messages: list[dict], max_new_tokens: int = 256, use_adapter: bool = True) -> dict:
        t0 = time.perf_counter()
        user = messages[-1]["content"]
        if user.startswith("Context:"):
            ctx, _, q = user.partition("\n\nQuestion:")
            text = "Relevant information found:\n" + ctx.replace("Context:", "", 1).strip()
        else:
            text = "No supporting context was found for this question."
        return {"text": text, "latency_s": time.perf_counter() - t0, "new_tokens": 0,
                "prompt_tokens": sum(len(m["content"]) for m in messages) // 4}


class HFBackend:
    def __init__(self, base_model: str = config.BASE_MODEL, adapter_dir: Path | None = config.ADAPTER_DIR,
                 load_in_4bit: bool = config.LOAD_IN_4BIT):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        cuda = torch.cuda.is_available()
        dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else (torch.float16 if cuda else torch.float32)
        self.tok = AutoTokenizer.from_pretrained(base_model)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kwargs = {"torch_dtype": dtype}
        if cuda and load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
            kwargs["device_map"] = {"": 0}
        model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
        if cuda and not load_in_4bit:
            model.to("cuda")
        self.has_adapter = bool(adapter_dir) and Path(adapter_dir).exists() and (Path(adapter_dir) / "adapter_config.json").exists()
        if self.has_adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(adapter_dir))
        model.eval()
        self.model = model
        self.name = f"{base_model}" + (" + LoRA adapter" if self.has_adapter else " (base, no adapter)")

    def generate(self, messages: list[dict], max_new_tokens: int = 256, use_adapter: bool = True) -> dict:
        torch = self.torch
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        ctx = self.model.disable_adapter() if (self.has_adapter and not use_adapter) else contextlib.nullcontext()
        t0 = time.perf_counter()
        with torch.no_grad(), ctx:
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        new = out[0][inputs["input_ids"].shape[1]:]
        return {"text": self.tok.decode(new, skip_special_tokens=True).strip(), "latency_s": dt, "new_tokens": int(len(new)),
                "prompt_tokens": int(inputs["input_ids"].shape[1])}


def get_backend():
    if config.LLM_BACKEND == "extractive":
        return ExtractiveBackend()
    return HFBackend(adapter_dir=config.ADAPTER_DIR if config.USE_ADAPTER else None)
