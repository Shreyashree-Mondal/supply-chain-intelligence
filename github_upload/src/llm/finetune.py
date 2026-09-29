"""Instruction fine-tuning with LoRA / QLoRA (Hugging Face PEFT + Trainer). Run this on the GPU machine.

    python -m src.llm.finetune --mode qlora                      # 4-bit base + LoRA adapters (least memory)
    python -m src.llm.finetune --mode lora                       # 16-bit base + LoRA adapters
    python -m src.llm.finetune --model Qwen/Qwen2.5-7B-Instruct --mode qlora --batch-size 2 --grad-accum 8

Loss is computed on the ANSWER tokens only (prompt tokens are masked with -100). The adapter is saved to
models/lora_adapter/ and is a few tens of MB; the base model is not modified.
"""
from __future__ import annotations

import argparse
import inspect
import json
import random
import time
from pathlib import Path

from src import config
from src.llm.prompts import BASIC_SYSTEM, PROMPT_VERSION, build_messages


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def encode_example(tok, e: dict, max_len: int):
    """Tokenise one example; returns None if it does not fit in max_len (we drop rather than truncate answers)."""
    prompt_msgs = build_messages(e["instruction"], e.get("context"), style="basic")   # the format the model is served with
    prompt = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    # Tokenise prompt and answer separately and concatenate: this matches inference exactly
    # (the model is given the tokenised prompt, then generates the answer tokens).
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    r_ids = tok(e["response"] + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]
    ids = p_ids + r_ids
    if len(ids) > max_len:
        return None
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [-100] * len(p_ids) + r_ids}


def build_features(tok, examples: list[dict], max_len: int):
    feats = [encode_example(tok, e, max_len) for e in examples]
    kept = [f for f in feats if f is not None]
    return kept, len(feats) - len(kept)


class ListDataset:
    def __init__(self, feats):
        self.feats = feats

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, i):
        return self.feats[i]


def make_collator(pad_id: int):
    import torch

    def collate(batch):
        m = max(len(b["input_ids"]) for b in batch)
        pad = lambda seq, v: seq + [v] * (m - len(seq))
        return {"input_ids": torch.tensor([pad(b["input_ids"], pad_id) for b in batch]),
                "attention_mask": torch.tensor([pad(b["attention_mask"], 0) for b in batch]),
                "labels": torch.tensor([pad(b["labels"], -100) for b in batch])}
    return collate


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.BASE_MODEL)
    ap.add_argument("--mode", choices=["qlora", "lora"], default="qlora")
    ap.add_argument("--train", default=str(config.INSTRUCTION_DIR / "train.jsonl"))
    ap.add_argument("--val", default=str(config.INSTRUCTION_DIR / "val.jsonl"))
    ap.add_argument("--out", default=str(config.ADAPTER_DIR))
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--no-grad-ckpt", action="store_true", help="faster but uses more memory")
    ap.add_argument("--max-examples", type=int, default=0, help="use only the first N training examples (quick dry run)")
    a = ap.parse_args(argv)

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,
                              TrainingArguments, set_seed)

    cuda = torch.cuda.is_available()
    if not cuda:
        if a.mode == "qlora":
            print("No CUDA GPU found: QLoRA (4-bit) needs one, so running plain LoRA on CPU instead.")
            a.mode = "lora"
        print("Running on CPU: use a small model (e.g. --model Qwen/Qwen2.5-0.5B-Instruct). This is slow.")
    set_seed(a.seed)
    bf16 = cuda and torch.cuda.is_bf16_supported()
    fp16 = cuda and not bf16
    dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)
    print(f"device: {torch.cuda.get_device_name(0) if cuda else 'CPU'} | dtype {dtype} | mode {a.mode} | model {a.model}")

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    train_feats, dropped_t = build_features(tok, read_jsonl(Path(a.train)), a.max_len)
    val_feats, dropped_v = build_features(tok, read_jsonl(Path(a.val)), a.max_len)
    print(f"train {len(train_feats)} (dropped {dropped_t} too long) | val {len(val_feats)} (dropped {dropped_v})")
    if a.max_examples:
        train_feats, val_feats = train_feats[:a.max_examples], val_feats[:max(8, a.max_examples // 4)]
        print(f"DRY RUN: using {len(train_feats)} train / {len(val_feats)} val examples")

    kwargs = {"torch_dtype": dtype}
    if a.mode == "qlora":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(a.model, **kwargs)
    if a.mode == "lora" and cuda:
        model.to("cuda")
    model.config.use_cache = False

    use_ckpt = cuda and not a.no_grad_ckpt      # gradient checkpointing only pays off when GPU memory is the limit
    if a.mode == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_ckpt,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    elif use_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    lcfg = LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=a.dropout, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    for p in model.parameters():            # adapters in fp32 so fp16 mixed precision can unscale gradients
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    steps_per_epoch = max(1, -(-len(train_feats) // (a.batch_size * a.grad_accum)))
    warmup_steps = max(1, int(0.03 * steps_per_epoch * a.epochs))      # works on every transformers version
    ta_kwargs = dict(
        output_dir=str(Path(a.out).parent / "checkpoints"), num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum, warmup_steps=warmup_steps, lr_scheduler_type="cosine",
        logging_steps=10, save_strategy="epoch", save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False, bf16=bf16, fp16=fp16, use_cpu=not cuda, dataloader_pin_memory=cuda,
        optim="paged_adamw_8bit" if a.mode == "qlora" else "adamw_torch", weight_decay=0.0,
        report_to="none", remove_unused_columns=False, seed=a.seed, dataloader_num_workers=0)
    # `evaluation_strategy` was renamed `eval_strategy` in recent transformers versions
    key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters else "evaluation_strategy"
    ta_kwargs[key] = "epoch"
    valid = inspect.signature(TrainingArguments.__init__).parameters       # newer versions rename/remove some options
    skipped = [k for k in ta_kwargs if k not in valid]
    if skipped:
        print("NOTE: this transformers version does not accept", skipped, "- skipping them")
    ta_kwargs = {k: v for k, v in ta_kwargs.items() if k in valid}

    trainer = Trainer(model=model, args=TrainingArguments(**ta_kwargs), train_dataset=ListDataset(train_feats),
                      eval_dataset=ListDataset(val_feats), data_collator=make_collator(tok.pad_token_id))
    t0 = time.time()
    trainer.train()
    minutes = (time.time() - t0) / 60

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out)
    tok.save_pretrained(out)
    meta = {"base_model": a.model, "mode": a.mode, "prompt_version": PROMPT_VERSION, "prompt_style": "basic", "system_prompt": BASIC_SYSTEM, "train_examples": len(train_feats), "val_examples": len(val_feats),
            "hyperparameters": {k: getattr(a, k) for k in ["epochs", "lr", "batch_size", "grad_accum", "max_len", "r", "alpha", "dropout", "seed"]},
            "train_minutes": round(minutes, 1), "final_eval": trainer.evaluate(), "log_history": trainer.state.log_history}
    (out / "training_summary.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"Saved adapter to {out} ({minutes:.1f} min). Final eval loss: {meta['final_eval'].get('eval_loss')}")


if __name__ == "__main__":
    main()
