import tempfile, time
from pathlib import Path
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from src import config
from src.llm.finetune import ListDataset, build_features, make_collator, read_jsonl
from src.llm.generation import HFBackend
from src.llm.prompts import build_messages

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
by_task = {}
for e in read_jsonl(config.INSTRUCTION_DIR / "train.jsonl"):
    by_task.setdefault(e["task_type"], e)
feats, dropped = build_features(tok, list(by_task.values()), 1024)
print(f"1) tokenised {len(feats)} examples (one per task type), dropped {dropped}; longest {max(len(f['input_ids']) for f in feats)} tokens")
f = feats[0]
ans = [t for t, l in zip(f["input_ids"], f["labels"]) if l != -100]
assert 0 < len(ans) < len(f["input_ids"]), "label masking looks wrong"
print(f"2) label masking ok: {len(ans)} answer tokens of {len(f['input_ids'])}; loss covers only: {tok.decode(ans)[:90]!r}")
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", target_modules=TARGETS))
model.print_trainable_parameters()
out = Path(tempfile.mkdtemp())
args = TrainingArguments(output_dir=str(out / "ckpt"), max_steps=3, per_device_train_batch_size=2, learning_rate=2e-4,
                         logging_steps=1, save_strategy="no", report_to="none", remove_unused_columns=False, dataloader_num_workers=0)
trainer = Trainer(model=model, args=args, train_dataset=ListDataset(feats), data_collator=make_collator(tok.pad_token_id))
t0 = time.time()
trainer.train()
losses = [round(h["loss"], 3) for h in trainer.state.log_history if "loss" in h]
assert losses and all(l == l and l < 1e3 for l in losses), f"bad losses {losses}"
print(f"3) trained 3 steps in {time.time() - t0:.0f}s on CPU, losses {losses}")
adapter = out / "adapter"
trainer.model.save_pretrained(adapter)
tok.save_pretrained(adapter)
assert (adapter / "adapter_config.json").exists()
print("4) adapter saved:", sorted(p.name for p in adapter.iterdir()))
backend = HFBackend(base_model=MODEL, adapter_dir=adapter, load_in_4bit=False)
assert backend.has_adapter
msgs = build_messages("What is safety stock?", style="basic")
tuned = backend.generate(msgs, max_new_tokens=25, use_adapter=True)
base = backend.generate(msgs, max_new_tokens=25, use_adapter=False)
print("5) adapter loads and generation works with it on and off:")
print("   with adapter    :", repr(tuned["text"][:90]), f"| {tuned['latency_s']:.1f}s, prompt tokens {tuned['prompt_tokens']}")
print("   adapter disabled:", repr(base["text"][:90]), f"| {base['latency_s']:.1f}s")
print("SMOKE TEST PASSED")
