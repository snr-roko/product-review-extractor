# Product Review Extractor

Turns messy product-review text into validated, typed JSON using a small local LLM (Ollama), Pydantic schema validation, and (soon) a retry loop that feeds validation errors back to the model.

> **Status: work in progress.** The schema, dataset and prompt work are done. The extraction pipeline, evals and API are next. This README is updated as each part lands.

## What it does

Input: a free-text product review.

```
Paid GH₵450 for this Binatone blender at the market. It stopped working after two weeks
and the seller refused a refund. Never buying from them or this brand again.
```

Output: a JSON object that is guaranteed to match a schema, or a clear failure.

```json
{
  "product_name": "Binatone blender",
  "price": 450.0,
  "currency": "GHS",
  "sentiment": "angry",
  "repurchase_intent": "no",
  "intent_evidence": "Never buying from them or this brand again",
  "issues_mentioned": ["stopped working after two weeks", "seller refused refund"]
}
```

## Why this project

It applies three ideas from the LLM engineering course: prompt engineering, few-shot prompting, and structured outputs (schema validation and retry loops). The goal is to measure how much each technique helps, not just to make it work once.

## Schema

| Field | Type | Nullable | Meaning |
|---|---|---|---|
| `product_name` | string | no | The product the review is mainly about |
| `price` | float, >= 0 | yes | Price actually paid. Null if not stated |
| `currency` | string | yes | ISO code (USD, GHS, ...). Null if no price or no currency marker |
| `sentiment` | enum | no | `delighted`, `satisfied`, `neutral`, `disappointed`, `frustrated`, `angry` |
| `repurchase_intent` | enum | no | `yes`, `no`, `maybe` (hedged or conditional), `unclear` (review is silent) |
| `intent_evidence` | string | yes | Exact quote supporting the intent. Null when intent is `unclear` |
| `issues_mentioned` | list of strings | no | Problems the reviewer explicitly states. Empty list if none |

All fields are **required keys**; nullable ones must be present with a `null` value. This forces the model to state "not mentioned" explicitly instead of silently omitting a field. A model-level validator also enforces that `currency` is null whenever `price` is null.

Schema tests live in `tests/test_schemas.py`. A finding worth noting: Pydantic's default lax mode coerces `"348"` to `348.0`; strict mode rejects it.

## Dataset

`data/reviews.jsonl` holds 25 hand-labeled synthetic reviews (one JSON object per line: `id`, `text`, `expected`). They include deliberately hard cases: missing prices, prices in words and in cedis/naira, sarcasm, two products in one review, hedged purchase intent, generic product names, a prompt-injection attempt, and two inputs with no product at all (expected to be rejected).

Labeling rules:

- **Price** is what was actually paid (sale price over list price).
- **Currency** is an ISO code. It is null when the review gives a price with no currency marker; it is never guessed.
- **Several products:** the one the review is mainly about (the one kept or bought).
- **Sentiment** is the reviewer's overall verdict at the end of the review.
- **Intent** requires an explicit statement. "Best laptop ever" is `unclear`, not `yes`.
- **Evidence** is a verbatim snippet from the text.

## Prompt iteration notes

Model: `llama3.2:3b` via Ollama, temperature 0. Smoke test on three reviews, comparing prompt versions. These are observations on a tiny sample, not measured results; proper numbers come from the eval harness.

**v1 (keys and format only, no definitions)**

- Wrapped JSON in code fences, inconsistently, and sometimes added text after it.
- Returned currency as a symbol (`GH₵`) instead of an ISO code, which still passes a "non-empty string" schema.
- Used `maybe` as a catch-all when the review said nothing about buying again.
- Quoted the wrong sentence as intent evidence (the failure, not the intent to rebuy).
- Invented content in `issues_mentioned` that the review never stated.

**v2 (same structure, plus field definitions, `null` instead of `None`, "JSON only")**

- Clean JSON with no fences or chatter on all three reviews.
- Currency correctly returned as `GHS`.
- Evidence now quotes the repurchase statement.
- Still wrong: `maybe` with null evidence on a review that says nothing about repurchase, and a non-issue ("nothing special") listed as an issue.

Takeaway: definitions in the prompt fixed formatting and several meaning errors, but a schema can only catch what it is taught to check. Remaining errors need extra validation rules and few-shot examples.

Inference speed on CPU: roughly 5 to 13 seconds per review (the first call includes model loading).

## Setup

```bash
# 1. Install Ollama (https://ollama.com) and pull the model
ollama pull llama3.2:3b

# 2. Install dependencies
uv sync

# 3. Run the schema tests
uv run pytest
```

## Project layout

```
src/product_review_extractor/
    schemas.py          # Pydantic models
data/reviews.jsonl      # labeled synthetic reviews
tests/test_schemas.py   # schema tests
scratch.py              # prompt experiments (not part of the package)
```

## Roadmap

- [x] Pydantic schema and schema tests
- [x] Labeled dataset (25 reviews)
- [x] Zero-shot prompt with field definitions
- [ ] JSON cleanup and parsing step
- [ ] Extraction function with validation and retry loop
- [ ] Extra checks (evidence must be a quote from the review, intent/evidence consistency)
- [ ] Eval script: first-try validity, validity after retries, per-field accuracy
- [ ] Comparisons: retry vs no retry, zero-shot vs few-shot
- [ ] FastAPI endpoint (`POST /extract`)
- [ ] Dockerfile
- [ ] Results table and "what failed" section
