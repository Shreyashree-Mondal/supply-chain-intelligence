# Prompt engineering in this project

All prompts are built in **one module, `src/llm/prompts.py`**, and versioned (`PROMPT_VERSION`, written into training summaries,
evaluation results and API responses). Nothing else in the code builds prompt strings.

## Two prompt styles, and why there are two
| Style | Used for | What it contains |
|---|---|---|
| `basic` | fine-tuning data, serving the **fine-tuned** model | short system prompt + `Context:` / `Question:` user turn |
| `engineered` | the **base** model (no fine-tuning), and the A/B test against fine-tuning | everything below |

**Rule: train and serve with the same prompt.** The adapter is trained on `basic`, so the API's `prompt_style: "auto"` picks `basic`
when the adapter is loaded and `engineered` otherwise. A test locks the basic format so it cannot drift away from the training data.

## Techniques in the `engineered` style
1. **Role + rules in the system prompt** - grounding ("never invent numbers; say what is missing"), concision, units.
2. **Prompt-injection hygiene** - the context is declared to be *data, not instructions*; it is always delimited (`Context:` ... `Question:`).
3. **Formula sheet in the prompt** - EOQ, safety stock, reorder point, days of supply, z-values, risk bands, ABC. Small models get arithmetic
   wrong when they must recall formulas; the sheet removes that failure mode.
4. **Task routing** - `detect_task()` is a rule-based classifier (calculation / product / group fact / risk / scenario / JSON / concept /
   general; checked against the dataset in a test). It selects a one-line task hint and the few-shot example. No extra model call.
5. **Few-shot examples as chat turns** - one hand-written example per task. They use fictional products and concepts that are not in the
   dataset, and their wording differs from the training templates; a test asserts there is no overlap with any split, so the evaluation
   is not contaminated. For product questions the shot depends on whether context exists (answer vs. correct refusal).
6. **Output-format control** - "Final answer: ..." for calculations, JSON-only with exact keys, and a fixed refusal pattern.
7. **Context budgeting** - `trim_context()` keeps whole records/passages in order within a character budget (cuts tokens and latency).
8. **Token accounting** - real prompt/output token counts and latency are logged per config in the evaluation and returned by `/ask`.

## Prompt evaluation (prompts are treated like code: measured, not assumed)
`python -m src.llm.evaluate` compares, on the same held-out examples:

`base` -> `base+prompt` (prompt engineering alone) -> `base+rag` -> `base+rag+prompt` -> `tuned` -> `tuned+rag`
(+ `tuned+prompt` with `--include-tuned-engineered`, which measures the train/serve prompt mismatch).

The summary has an "Effect of the engineered prompt" table and per-config prompt-token counts, so you can state the cost/benefit with
your own numbers: e.g. how much accuracy prompting buys vs. fine-tuning, and how many more tokens per request it costs.
**Do not quote any effect before you have run it** - it may be small, and it may even be negative for some tasks.

## Inspect any prompt without a model
```bash
python -m src.llm.prompt_lab "What is the reorder point for Perfect Fitness Perfect Rip Deck?"
python -m src.llm.prompt_lab "What is EOQ?" --compare
```
Shows the detected task, the retrieved sources, every message the model would receive, and the approximate token count.

## Interview talking points
- Why two styles, and why train/serve prompts must match (and the test that enforces it).
- Prompting vs. RAG vs. fine-tuning: prompting fixes format and reasoning scaffolding, RAG supplies facts, fine-tuning bakes in behaviour and
  reduces prompt length; you measured all three on the same test set.
- How you avoided evaluation leakage in few-shot examples.
- Trade-off: the engineered prompt costs extra tokens (latency, cost) on every request; the tuned model does not.
