# Supply Chain Intelligence + GenAI Copilot (DataCo)

End-to-end supply-chain analytics project on the public DataCo dataset, extended with a GenAI layer:
a late-delivery risk model, demand forecasting and inventory optimisation for every product, a RAG pipeline,
versioned prompt engineering, **LoRA instruction fine-tuning** (Hugging Face PEFT) of an open-source LLM, a FastAPI service,
and an evaluation that compares prompting vs RAG vs fine-tuning on held-out data.

> All experiments ran on CPU with a small model (Qwen2.5-0.5B-Instruct). The instruction dataset is synthetic (template-generated from computed values).
> See [Limitations](#limitations) before drawing conclusions from any number below.

## Results at a glance
| Area | Result |
|---|---|
| Late-delivery model (XGBoost, 5 order-time features, time-based split) | ROC-AUC **0.740**, accuracy 0.696, precision 0.834, recall 0.558 on 9,026 most-recent orders |
| Inventory optimisation | Safety stock, reorder point, EOQ and ABC class for **100 products** (55 active), up from 9 in the first EDA |
| What-if stress test (+10% demand, +20% lead time) | Reorder-point value $222k → $285k (**+28%**) across the 55 active products |
| Product-fact questions on held-out products (n=25) | Correct figure in **4%** of answers closed-book, **48%** base + RAG, **76%** base + RAG + engineered prompt, **92%** fine-tuned + RAG |
| Valid structured JSON with correct risk band (n=17) | 53% base → 77% engineered prompt → **100%** fine-tuned |
| Correctly declining when data is missing (n=10 each) | 0–50% base → **100%** fine-tuned |
| LLM-judge score, 20 paired questions (1–5 correctness) | base **1.70** vs fine-tuned system **4.60** |

## Pipeline
```
raw CSV ─► src.pipeline ─► late-delivery XGBoost + demand forecasts + inventory policy (all products)
                          ├─► knowledge base (concept notes + documents generated from the data) ─► RAG index (FAISS)
                          ├─► instruction dataset (JSONL) ─► LoRA fine-tune ─► adapter
                          └─► FastAPI: /ask  /predict-late-risk  /inventory/{id}  /inventory-scenario  /products
```
**Design principle:** the model is fine-tuned for terminology, format and behaviour (calculation steps, interpreting model output, JSON, declining when
data is missing). Facts and numbers are never learned: they come from RAG plus an exact product-record lookup.

## Data and modelling notes
- 180,519 order items, 118 products, 2015-01 to 2018-01. Post-outcome columns (delivery status, actual shipping days) are excluded from the risk model to avoid leakage.
- **Data-quality finding:** total monthly demand falls 76% in October 2017 (10,502 → 2,490) and 40 products stop selling in September 2017. The forecasting window is cut at
  2017-09; the 18 products first seen afterwards are excluded rather than forecast from 1–4 months of data.
- Each product gets the best forecaster its history supports (naive / 3-month moving average / exponential smoothing chosen by rolling backtest); products with short history
  use a pooled variability estimate and are marked Low confidence (9 High, 31 Medium, 15 Low among active products).
- Lead time: the dataset has outbound shipping days, not supplier lead times, so average shipping days (minimum 1) is used as the replenishment lead time. Configurable.

## GenAI layer
- **RAG:** `all-MiniLM-L6-v2` embeddings + FAISS over 142 chunks (concept notes, data summaries, one record per product) plus an exact product-name lookup, so figures come from the policy table, not from embeddings.
- **Prompt engineering** (`src/llm/prompts.py`, versioned): a training-format `basic` prompt and an `engineered` prompt (grounding rules, context-as-data rule, formula sheet,
  rule-based task routing, few-shot examples, output-format control, context budgeting). Few-shot examples use fictional entities and are tested for no overlap with any dataset split.
- **Instruction dataset:** 1,832 examples, 10 task types, split by product (held-out products never appear in training). Numbers are computed from real outputs and standard formulas; wording is template-generated.
- **LoRA fine-tuning:** Qwen2.5-0.5B-Instruct, r=16, alpha=32, 8.8M trainable parameters (1.75%), 3 epochs on 1,473 examples, 128 min on CPU; loss on answer tokens only.
  Validation loss 0.126 → 0.039 → 0.031.

## Evaluation
Held-out test examples (≤25 per task), answered under each setup. Primary metric per task: numeric answer correct (calculation), valid JSON with correct band (structured_json),
declined correctly (refusal tasks), ROUGE-L (concept_qa), fraction of required facts present (all others). RAG setups were run only on the three knowledge tasks.

| Task (n) | base | base + prompt | base + RAG | base + RAG + prompt | tuned | tuned + RAG |
|---|---|---|---|---|---|---|
| product_fact (25) | 0.04 | 0.00 | 0.48 | 0.76 | 0.04 | **0.92** |
| concept_qa (25) | 0.18 | 0.23 | 0.35 | 0.33 | 0.84 | 1.00 |
| structured_json (17) | 0.53 | 0.77 | – | – | **1.00** | – |
| risk_interpretation (17) | 0.47 | 0.97 | – | – | 0.97 | – |
| refusal_no_context (10) | 0.50 | 0.10 | – | – | **1.00** | – |
| refusal_wrong_context (10) | 0.00 | 0.00 | – | – | **1.00** | – |
| calculation (24) | 0.17 | 0.21 | – | – | 0.38 | – |

Cost per answer on CPU (mean): base 7.1 s / 129 prompt tokens; base + engineered prompt 3.8 s / 527 tokens; fine-tuned 4.0 s / 129 tokens; fine-tuned + RAG 3.2 s / 420 tokens.
The engineered prompt adds ~400 prompt tokens per request; the fine-tuned model does not.

**LLM-judge check.** 40 answers (20 questions × base vs best system) were scored 1–5 by an LLM judge (Claude) against the reference answers, blind to which system wrote each answer:
base 1.70 correctness / 2.95 groundedness; fine-tuned system 4.60 / 4.85. Correlation of automatic metrics with the judge's correctness (n=40, Spearman):
task-specific check 0.80, BERTScore 0.78, ROUGE-L 0.74. All three fail on wrong numbers inside otherwise well-formed answers: a safety-stock answer with wrong arithmetic (165 vs 186)
scored BERTScore ≈1.00 and ROUGE-L 0.90 but correctness 2 from the judge.

## What did not work (error analysis)
- **Arithmetic:** the fine-tuned model learns the calculation procedure but not exact multiplication (calculation 0.38). Exact numbers should come from code, which is how the API works.
- **Engineered prompt on missing data:** refusal accuracy fell from 0.50 to 0.10. The 0.5B base model ignored the "do not invent figures" rule, and the task router missed some phrasings.
  Fine-tuning fixed it (1.00). Prompting alone matched fine-tuning on risk interpretation (0.97) but not on refusals or JSON.
- **RAG context contamination:** in 2 of 25 tuned + RAG product answers the model copied the worked example from a retrieved concept note instead of the product record.
- **Risk-model probabilities are not calibrated:** First Class orders get ~0.99 because premium modes are almost always late in this data; the model's signal comes mostly from shipping mode.

## Limitations
- The instruction dataset is synthetic and train/test share templates, so scores measure this task suite, not general ability. Samples are small (4–25 per task); differences of a few examples are noise.
- Everything ran on CPU with a 0.5B model. **QLoRA (4-bit) was not run.** The code path exists but is untested on a GPU.
- The judge is an LLM, not a human, and was applied once; the sample is small. Judge-scored files are not included.
- Lead time is an assumption (shipping days), and 18 recently introduced products are excluded from the inventory policy.
- Not implemented: LangChain, semantic caching, model routing, Docker, cloud deployment.

## Run it
```bash
pip install -r requirements.txt                       # core (no GPU); pytest -q runs the test suite on synthetic data
python -m src.pipeline --forecast-end 2017-09         # model + forecasts + inventory (needs data/raw/DataCoSupplyChainDataset.csv)
python -m src.llm.knowledge_base
python -m src.llm.build_instruction_dataset
pip install -r requirements-llm.txt                   # torch, transformers, peft, sentence-transformers ...
python -m src.llm.rag build
python -m src.llm.finetune --model Qwen/Qwen2.5-0.5B-Instruct --mode lora --batch-size 2 --grad-accum 8 --out models/lora_qwen0.5b
SCP_BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct SCP_ADAPTER_DIR=models/lora_qwen0.5b python -m src.llm.evaluate --max-per-task 25
uvicorn src.api.main:app --port 8000                  # docs at /docs; SCP_LLM_BACKEND=extractive runs without any LLM
python -m src.llm.prompt_lab "What is the reorder point for <product name>?" --compare
```
With a GPU, `--mode qlora` uses a 4-bit base model. Details of the prompt design: [docs/prompt_engineering.md](docs/prompt_engineering.md).

## Repository layout
`src/pipeline.py` orchestration · `src/risk_models/` late-delivery model · `src/forecasting/` demand · `src/optimization/` inventory ·
`src/llm/` knowledge base, RAG, prompts, dataset builder, fine-tuning, evaluation · `src/api/` FastAPI · `tests/` pytest suite (runs in CI on synthetic data) · `notebooks/` original EDA · `docs/` design notes.
