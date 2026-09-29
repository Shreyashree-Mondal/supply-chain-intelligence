# Supply Chain Copilot (DataCo)

Extends the DataCo supply-chain analytics project with a GenAI layer: **instruction fine-tuning (LoRA / QLoRA via PEFT), RAG, and a REST API**,
on top of the existing late-delivery model, demand forecasts and inventory optimisation.

```
raw CSV ─► src.pipeline ─► late-delivery XGBoost + demand forecasts + inventory policy for EVERY product
                          │
                          ├─► knowledge base (concept notes + docs generated from your data) ─► RAG index (FAISS)
                          ├─► instruction dataset (JSONL) ─► LoRA/QLoRA fine-tune ─► adapter
                          └─► FastAPI:  /ask  /predict-late-risk  /inventory/{id}  /inventory-scenario  /products
```

**Prompt engineering** lives in one versioned module (`src/llm/prompts.py`): a training-compatible `basic` prompt and an `engineered` prompt
(grounding rules, injection hygiene, formula sheet, rule-based task routing, few-shot examples, output-format control, context budgeting).
The evaluation compares prompting vs RAG vs fine-tuning head-to-head. Details: `docs/prompt_engineering.md`.

**Design:** the model is fine-tuned for *terminology, format and behaviour* (calculations, interpreting model output, JSON,
declining when data is missing). **Facts and numbers are never learned** - they come from RAG plus an exact product-record lookup,
so the LLM does not have to recall (or invent) inventory figures.

## What changed vs the notebook
| | Notebook | This project |
|---|---|---|
| Inventory coverage | 9 continuing products | **all products** (each tagged Active/Inactive and High/Medium/Low data confidence) |
| Forecast | exponential smoothing (statsmodels) on 9 products | best of naive / 3-mo MA / exponential smoothing per product by rolling backtest; short-history products use pooled variability |
| Reorder point | step 2 used 1 month of demand, the scenarios used shipping-day lead time | one consistent formula: forecast x lead time + safety stock |
| Inventory risk tier | safety-stock share of ROP | forecast variability (CV); low-confidence products are never "Low" |
| ABC | all products | Active products only |

Numbers will therefore differ from your Power BI export. That is expected (see "Assumptions to sanity-check").

## Where to put things (repo root = your project folder that contains `data/`, `src/`, `notebooks/`, ...)
Copy the contents of this zip into your repo root. It **adds** files; nothing of yours is overwritten except that `src/*/` gets new files.
- `src/`, `tests/`, `docs/`, `.github/`, `requirements*.txt`, `pytest.ini`, `README.md` -> repo root
- Put the dataset at `data/raw/DataCoSupplyChainDataset.csv` (already there)
- Append `gitignore_additions.txt` to your `.gitignore`
- Your existing empty folders (`src/data_cleaning`, `nlp`, `feature_engineering`, ...) can stay as they are.

## Run order
All commands are run from the repo root.

```bash
# 0) once (any machine)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                   # 46 tests on synthetic data, ~5 s

# 1) data -> model + forecasts + inventory for all products   (CPU, ~1 min)
python -m src.pipeline
#    read the printed summary: product counts by status/confidence, and the last-6-months demand table.
#    if the last months show an anomalous drop (your notebook flagged Oct 2017), you can ignore them:
#    python -m src.pipeline --forecast-end 2017-09

# 2) knowledge base + instruction dataset                       (CPU)
python -m src.llm.knowledge_base
python -m src.llm.build_instruction_dataset
#    open data/processed/instructions/review_sample.csv and read ~60 rows; fix generators if something is wrong

# 3) on the GPU machine:  pip install -r requirements-llm.txt  (install the torch build for your CUDA first)
python -m src.llm.rag build                 # embeddings + FAISS   (use SCP_RAG_BACKEND=tfidf if the model can't be downloaded)
python -m src.llm.finetune --mode qlora     # QLoRA; ~20-40 min for ~1.5k examples on a 16-24 GB GPU
# python -m src.llm.finetune --mode lora    # 16-bit LoRA variant (needs more memory)
# python -m src.llm.finetune --model Qwen/Qwen2.5-7B-Instruct --mode qlora --batch-size 2 --grad-accum 8   # bigger model, 24 GB+

# 4) evaluate: base vs tuned vs base+RAG vs tuned+RAG, latency, human-review sheet
python -m src.llm.evaluate                  # -> reports/eval_summary.md, eval_results.csv, human_review.csv
#    fill correctness_1_5 / groundedness_1_5 in reports/human_review.csv, then:
python -m src.llm.evaluate --summarize-human reports/human_review.csv

#    see exactly what the model receives (no GPU needed):
python -m src.llm.prompt_lab "What is the reorder point for <a product name from your data>?" --compare

# 5) serve
uvicorn src.api.main:app --host 0.0.0.0 --port 8000      # docs at http://localhost:8000/docs
```
No GPU / no downloads? `SCP_LLM_BACKEND=extractive SCP_RAG_BACKEND=tfidf uvicorn src.api.main:app` serves everything with retrieval-only answers.
If the GPU node has no internet, download the model on a machine that does (`huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct`) and point `SCP_BASE_MODEL` at the folder.

Example calls:
```bash
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"What is the reorder point for Perfect Fitness Perfect Rip Deck?"}'
curl -X POST localhost:8000/predict-late-risk -H 'content-type: application/json' \
  -d '{"shipping_mode":"First Class","scheduled_days":1,"payment_type":"DEBIT","customer_segment":"Consumer","order_region":"Western Europe"}'
curl -X POST localhost:8000/inventory-scenario -H 'content-type: application/json' \
  -d '{"demand_change_pct":10,"lead_time_change_pct":20}'
```

## Configuration (`src/config.py`, env overrides)
`SCP_BASE_MODEL` (default Qwen/Qwen2.5-1.5B-Instruct - check its license on the model card before commercial use), `SCP_EMBED_MODEL`,
`SCP_RAG_BACKEND` (sbert|tfidf), `SCP_LLM_BACKEND` (hf|extractive), `SCP_USE_ADAPTER`, `SCP_LOAD_IN_4BIT`, `SCP_FIXED_LEAD_TIME_DAYS`, `SCP_RAW_CSV`.

## Assumptions to sanity-check
- **Lead time:** DataCo has *outbound shipping days*, not supplier lead times. By default each product's average shipping days (min 1) is used as its
  replenishment lead time (this is what your notebook's scenario section did). Set `SCP_FIXED_LEAD_TIME_DAYS=30` to use a fixed supplier lead time instead.
- Service level 95%, ordering cost $50, holding rate 20% (same as the notebook).
- "Inactive" = no demand in the last 3 months of data. These products still get figures, but the recommended action says to review before replenishing.
- Products with < 6 months of history get a provisional variability estimate (pooled CV) and `Data_Confidence = Low`.

## What is real and what is synthetic (say this in interviews / on the resume)
- The instruction dataset is **template-generated**: numbers come from your real outputs and standard formulas (so they are correct), but the wording is
  synthetic and lower-diversity than human-written data. A 60-row sample is exported for manual review. Product-based examples are split by *product*
  (held-out products never appear in training); concept examples hold out *paraphrases*, not concepts.
- Evaluation is small-scale (a few dozen examples per task). Report it as such.
- The engineered prompt's effect is measured by the evaluation, not assumed; report your own numbers (it can be small or negative on some tasks).
- Not implemented: LangChain, semantic caching, model routing, Docker, cloud deployment. Do not list them.

## Resume wording (only after you have run it and have your own numbers)
> Built a supply-chain GenAI copilot: QLoRA/LoRA instruction-tuning of an open-source LLM (Hugging Face PEFT) on a domain instruction dataset,
> RAG (sentence-transformer embeddings + FAISS), versioned prompt engineering (grounding rules, few-shot, task routing) evaluated against fine-tuning,
> and a FastAPI service exposing an XGBoost late-delivery model, demand forecasts and inventory
> optimisation (safety stock / reorder point / EOQ) for N products; evaluated base vs fine-tuned vs RAG configurations with automated metrics,
> latency measurements and a manually scored sample. Pytest suite + GitHub Actions CI.
