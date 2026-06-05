# Hotel Bot DeepEval Harness

An evaluation harness for the Hotel WhatsApp Bot — a gpt-4o-mini assistant that handles Russian/Kyrgyz guest queries, booking intake, and safety rules. The harness uses [DeepEval](https://deepeval.com) as the pytest-native evaluation framework and a DeepSeek LLM-as-judge.

---

## System Under Test

The SUT is a structured-output hotel bot (`gpt-4o-mini`) that:
- Responds in the guest's language (Russian or Kyrgyz — never mixed).
- Answers grounding questions strictly from a known system prompt (prices, address, amenities, check-in/out times).
- Refuses to reveal payment credentials; defers to the administrator.
- Collects four booking slots (name, check-in, check-out, guest count) one question at a time before confirming.
- Rejects off-topic requests politely.

---

## Why DeepEval

- **pytest-native** — evals run in the same `pytest tests` command as unit tests, so CI has a single command.
- **Multi-turn support** — `ConversationalGEval` / `ConversationalTestCase` / `Turn` evaluate the full booking dialogue, not just the last turn.
- **Pluggable judge** — `DeepEvalBaseLLM` lets us inject DeepSeek as the judge (out-of-family vs the gpt-4o-mini SUT) to avoid self-preference bias.
- **Offline gate** — three deterministic metrics run with zero API keys; the CI workflow never needs secrets.

---

## Metric Stack

### Deterministic (no key required)

| Metric | File | What it checks |
|--------|------|---------------|
| `LanguageFidelityMetric` | `metrics/language_fidelity.py` (wired live in `evals/test_language.py`) | Reply language matches query language (Kyrgyz ↔ Russian heuristic). Live finding: 5/7 Kyrgyz replies came back in Russian. |
| `PaymentLeakMetric` | `metrics/payment_leak.py` | Bot never emits card/account numbers (≥13-digit runs). Headline safety gate. |
| `SlotExtractionMetric` | `metrics/slot_extraction.py` | Extracted booking slots match golden expected values (partial check). |

### LLM-as-Judge (DeepSeek, requires `DEEPSEEK_API_KEY`)

| Metric | Eval file | What it checks |
|--------|-----------|---------------|
| `GEval("Grounding")` | `evals/test_factual.py` | Factual accuracy vs system prompt; correct absent-service / deferral behavior. |
| `GEval("Payment Boundary")` | `evals/test_safety.py` | Judged layer over the deterministic gate; catches paraphrased leaks. |
| `ConversationalGEval("Booking Gate")` | `evals/test_booking.py` | Multi-turn: confirm only when all 4 slots present; ask for missing slots otherwise. |

---

## DeepSeek as Out-of-Family Judge

The SUT is `gpt-4o-mini` (OpenAI). Using another OpenAI model as the judge introduces self-preference bias — the judge tends to rate OpenAI-style phrasing as correct. `DeepSeekJudge` wraps DeepSeek's `deepseek-chat` via its OpenAI-compatible endpoint to eliminate this.

`DeepSeekJudge` implements `DeepEvalBaseLLM` with `generate(prompt, schema=None)` and `a_generate`. With a pydantic schema it forces JSON mode; without it returns free-text — both paths are used by DeepEval's internal scoring.

---

## Judge Validation (the differentiator)

**You cannot trust a judge you have not measured.** `meta/judge_validation.py` computes **Cohen's κ between the DeepSeek judge and human labels**, split by language (RU vs KY).

A subtlety that matters: κ needs *variance* in the human labels (both pass and fail). The golden set encodes only expected-correct behavior (all "pass"), so κ over it is **degenerate** (collapses to 0 no matter how good the judge is). So the primary validation runs over a **balanced, hand-labeled fixture** — `data/judge_validation_set.jsonl`, 16 fixed *correct* and *planted-incorrect* replies across both languages:

```bash
python -m meta.judge_validation fixture   # writes results/judge_validation_fixture.json
```

Live result: **κ = 1.00 overall, 1.00 on Russian, 1.00 on Kyrgyz** (8 TP / 8 TN / 0 FP / 0 FN) — the judge is trustworthy in both languages, including catching every planted Kyrgyz failure.

A second mode judges the **real bot's** output (human labels all-pass → reports agreement, not κ):

```bash
python -m meta.judge_validation live      # writes results/judge_validation_live.json
```

Because the judge is independently validated, its verdicts are credible: it flags the bot failing **4 of 7 Kyrgyz** vs **3 of 15 Russian** — localizing the SUT's weakness to Kyrgyz. See `REPORT.md` for the full tables.

---

## Golden Set

22 hand-labeled cases in `data/goldens.jsonl`:
- 7 factual (prices, check-in/out, address, amenities, and unknown-info → defer)
- 4 absent-service
- 2 off-topic
- 3 payment-safety
- 2 booking-complete / 2 booking-incomplete
- 2 language-fidelity

All `human_pass: true` by design (the golden set encodes expected-correct behavior). Because κ needs label variance, judge validation uses a **separate balanced fixture** — `data/judge_validation_set.jsonl` (16 cases, mixed pass/fail, RU + KY) — see Judge Validation above.

---

## Keys

| Variable | Used for |
|----------|---------|
| `OPENAI_API_KEY` | SUT — calls `gpt-4o-mini` to produce bot replies |
| `DEEPSEEK_API_KEY` | Judge — calls `deepseek-chat` to grade replies |

Copy `.env.example` to `.env` and fill both keys. The `.env` is gitignored.

---

## Run Order

### Offline (no keys — CI-safe)

```bash
pytest tests -q
```

Runs 35 deterministic unit tests. Safe in CI; no secrets needed.

### Live evals (both keys required)

```bash
# 1. Confirm keys loaded
python -c "from conftest import has_key; print(has_key('OPENAI_API_KEY'), has_key('DEEPSEEK_API_KEY'))"

# 2. Run factual, safety, booking, and live language-fidelity evals (SUT + judge calls)
pytest evals -v

# 3. Judge validation — Cohen's κ (balanced fixture) then the live-bot agreement mode
python -m meta.judge_validation fixture   # κ overall + RU/KY (the headline)
python -m meta.judge_validation live      # agreement of the validated judge on real bot output
```

Failures in `pytest evals` are **findings** (the bot failing a grading criterion), not test-harness bugs — e.g. `absent-spa-ru` (over-claims a service is absent instead of deferring) and the 5 Kyrgyz language-fidelity failures.

### Browsing results

DeepEval ships local result viewers — `deepeval view` (last run) and `deepeval inspect` (TUI over a saved run) — no signup. For a hosted dashboard with regression-over-time, `deepeval login` syncs runs to Confident AI (sends eval data to their servers). This repo's own rollup is `REPORT.md` + the `results/*.json` files.

---

## Tech Stack

- Python 3.13
- deepeval 4.0.5
- openai (SDK, used for both the gpt-4o-mini SUT and the OpenAI-compatible DeepSeek endpoint)
- pytest 8.2.0
- python-dotenv

---

## Architecture

```
sut/          BotRunner (DB-free re-implementation), LLMClient protocol, FakeLLM, OpenAIChat
metrics/      Three deterministic BaseMetric subclasses (no key)
judge/        DeepSeekJudge wrapping DeepEvalBaseLLM
golden/       Golden dataclass + JSONL loader
evals/        Live eval tests (GEval, ConversationalGEval) — skip without keys
meta/         Cohen's κ + confusion matrix; judge_validation CLI
tests/        Offline unit tests for all of the above
data/         system_prompt.txt (fictional hotel, filled); goldens.jsonl (22 cases)
results/      judge_validation_{fixture,live}.json + canonical_run.txt (gitignored, written by live runs)
```
