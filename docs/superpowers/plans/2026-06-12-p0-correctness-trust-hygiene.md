# P0 + P2.5 — Correctness, Trust & Hygiene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close all P0 silent-failure surfaces (judge retry/backoff, startup validation, grounding CI gate, temperature control) and apply hygiene fixes (lru_cache, named constants, dep pinning, pre-commit).

**Architecture:** 8 independent changes. No new packages needed. `meta/gate.py` is the only new module — extracts CI gate logic so it can be unit-tested without running the workflow. All other changes are in-place modifications. Each task is shippable individually; commit after each one.

**Tech Stack:** Python 3.13 · uv · pytest 8.2 · ruff · mypy · DeepEval 4.0.5 · GitHub Actions

---

## File Map

| File | Action | Task |
|---|---|---|
| `judge/deepseek_judge.py` | Modify — add `JudgeError`, retry loop in `generate()` | 1 |
| `evals/run_suite.py` | Modify — `judge_error_rate` counter + `_check_env_vars()` + `--temperature` arg | 2, 3, 4 |
| `sut/hotel_bot/bot.py` | Modify — read `SUT_TEMPERATURE` env var in `handle_message()` | 4 |
| `meta/gate.py` | Create — pure gate functions (payment, grounding, error rate) | 5 |
| `.github/workflows/live-eval.yml` | Modify — use `meta/gate.py`; add grounding ratchet | 5 |
| `metrics/language_fidelity.py` | Modify — `@lru_cache(maxsize=512)` on `detect_lang()` | 6 |
| `metrics/payment_leak.py` | Modify — named constants for magic numbers | 7 |
| `pyproject.toml` | Modify — pin `langdetect`, `python-dotenv` | 8 |
| `.pre-commit-config.yaml` | Create — ruff hooks | 8 |
| `tests/test_judge_retry.py` | Create | 1 |
| `tests/test_gate.py` | Create | 5 |
| `tests/test_run_suite.py` | Modify — add `judge_error_rate`, `_check_env_vars` tests | 2, 3 |
| `tests/test_bot_runner.py` | Modify — add temperature passthrough test | 4 |
| `tests/test_language_fidelity.py` | Modify — verify `detect_lang` has `cache_info` | 6 |

---

## Task 1: JudgeError + retry/backoff in `generate()`

**P0.1a — closes silent crash when DeepSeek returns bad JSON or times out.**

**Files:**
- Create: `tests/test_judge_retry.py`
- Modify: `judge/deepseek_judge.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_judge_retry.py
"""Tests for judge retry/backoff and JudgeError exception."""
import json
import pytest
from pydantic import BaseModel

from judge.deepseek_judge import DeepSeekJudge, JudgeError


class _Schema(BaseModel):
    verdict: bool


def _judge():
    return DeepSeekJudge(api_key="test-key", base_url="http://fake")


def test_generate_retries_on_bad_json_and_succeeds(monkeypatch):
    """Two bad-JSON responses, then valid JSON — should return parsed schema."""
    monkeypatch.setattr("time.sleep", lambda _: None)
    judge = _judge()
    call_count = 0

    def fake_chat(prompt, json_mode):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return "not valid json!!!"
        return '{"verdict": true}'

    monkeypatch.setattr(judge, "_chat", fake_chat)
    result = judge.generate("test", schema=_Schema)
    assert result.verdict is True
    assert call_count == 3


def test_generate_raises_judge_error_on_persistent_bad_json(monkeypatch):
    """Three consecutive bad-JSON responses should raise JudgeError."""
    monkeypatch.setattr("time.sleep", lambda _: None)
    judge = _judge()

    monkeypatch.setattr(judge, "_chat", lambda *_: "not json")

    with pytest.raises(JudgeError, match="judge failed after 3 attempts"):
        judge.generate("test", schema=_Schema)


def test_generate_raises_judge_error_on_network_failure(monkeypatch):
    """Persistent connection errors should raise JudgeError, not ConnectionError."""
    monkeypatch.setattr("time.sleep", lambda _: None)
    judge = _judge()

    def boom(*_):
        raise ConnectionError("network down")

    monkeypatch.setattr(judge, "_chat", boom)

    with pytest.raises(JudgeError, match="judge failed after 3 attempts"):
        judge.generate("test")


def test_generate_no_retry_for_plain_text(monkeypatch):
    """Without schema, bad text is returned as-is (no JSON parse, no retry)."""
    judge = _judge()
    monkeypatch.setattr(judge, "_chat", lambda *_: "some plain text")
    result = judge.generate("test")
    assert result == "some plain text"


def test_judge_error_is_importable():
    from judge.deepseek_judge import JudgeError  # noqa: F401 — import check
    assert issubclass(JudgeError, Exception)
```

- [ ] **Step 2: Run to confirm failures**

```bash
uv run pytest tests/test_judge_retry.py -v
```

Expected: `FAILED` on all 5 tests with `ImportError: cannot import name 'JudgeError'`.

- [ ] **Step 3: Implement `JudgeError` and retry loop**

Replace `judge/deepseek_judge.py` entirely:

```python
# judge/deepseek_judge.py
"""DeepSeek as a DeepEval judge model. OpenAI-compatible endpoint; out-of-family
vs the gpt-4o-mini SUT, so the judge has no self-preference toward the SUT.

DeepEval may call generate(prompt) for free-text scoring or generate(prompt, schema)
for structured scoring (pydantic model). We honor both: with a schema we instruct
JSON-only and validate into the schema; without, we return raw text.

generate() retries up to 3 times with exponential backoff (1s, 2s) on any exception
(JSON parse errors, network timeouts, API errors). Persistent failure raises JudgeError
so the caller can distinguish a judge outage from a bot error.
"""
import json
import os
import random
import time

from deepeval.models import DeepEvalBaseLLM


class JudgeError(Exception):
    """Raised when the judge fails after all retry attempts."""


class DeepSeekJudge(DeepEvalBaseLLM):
    def __init__(self, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None):
        default_model = os.environ.get("DEEPSEEK_JUDGE_MODEL", "deepseek-chat")
        self.model = model or default_model  # type: ignore[assignment]
        self._api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self._base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL",
                                                    "https://api.deepseek.com")
        self._client = None

    def load_model(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url,
                                  timeout=30.0, max_retries=2)
        return self._client

    def _chat(self, prompt: str, json_mode: bool) -> str:
        client = self.load_model()
        kwargs = {"model": self.model, "temperature": 0,
                  "messages": [{"role": "user", "content": prompt}]}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        r = client.chat.completions.create(**kwargs)
        return r.choices[0].message.content or ""

    def generate(self, prompt: str, schema=None):
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                text = self._chat(
                    prompt + ("\n\nReturn ONLY valid JSON." if schema else ""),
                    json_mode=schema is not None,
                )
                if schema is None:
                    return text
                return schema(**json.loads(text))
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < 2:
                    time.sleep(2 ** attempt + random.random())
        raise JudgeError(f"judge failed after 3 attempts: {last_exc}") from last_exc

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return f"deepseek:{self.model}"
```

- [ ] **Step 4: Run tests to confirm passing**

```bash
uv run pytest tests/test_judge_retry.py -v
```

Expected: `5 passed`.

- [ ] **Step 5: Run full offline suite to confirm no regressions**

```bash
uv run pytest tests/ -q
```

Expected: all tests pass (was 138, now 143).

- [ ] **Step 6: Commit**

```bash
git add judge/deepseek_judge.py tests/test_judge_retry.py
git commit -m "feat: add JudgeError + 3-attempt retry/backoff in DeepSeekJudge.generate()"
```

---

## Task 2: `judge_error_rate` in run_suite report

**P0.1b — surfaces persistent judge failures in the report instead of hiding them in the `errors` counter.**

**Files:**
- Modify: `evals/run_suite.py`
- Modify: `tests/test_run_suite.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_run_suite.py`:

```python
# Add these imports at the top of tests/test_run_suite.py
from judge.deepseek_judge import JudgeError


class _JudgeErrorGrounding(_FakeGrounding):
    """Stand-in that always raises JudgeError — exercises the judge_error path."""

    def measure(self, tc):
        raise JudgeError("simulated persistent judge failure")


def test_judge_error_is_tracked_separately(monkeypatch):
    """JudgeError increments judge_error_rate, not the general errors counter."""
    monkeypatch.setattr(run_suite, "_grounding_metric",
                        lambda: _JudgeErrorGrounding())
    report = run_suite.run(source="goldens")
    # factual + payment_safety are non-booking → both hit the judge → both fail
    assert report["judge_errors"] == 2
    assert report["errors"] == 0          # bot errors still 0
    assert report["judge_error_rate"] > 0.0
    # judge_error rows should appear in the report
    assert report["summary"]["by_metric"]["judge_error"]["n"] == 2


def test_report_shape_includes_judge_error_keys(monkeypatch):
    """report always has judge_errors and judge_error_rate keys."""
    report = run_suite.run(source="goldens")
    assert "judge_errors" in report
    assert "judge_error_rate" in report
```

- [ ] **Step 2: Run to confirm failures**

```bash
uv run pytest tests/test_run_suite.py::test_judge_error_is_tracked_separately tests/test_run_suite.py::test_report_shape_includes_judge_error_keys -v
```

Expected: `FAILED` — `KeyError: 'judge_errors'`.

- [ ] **Step 3: Implement judge_error tracking in run_suite.py**

In `evals/run_suite.py`, make these changes:

At the top, add the import:
```python
from judge.deepseek_judge import JudgeError
```

In the `run()` function, add `judge_errors = 0` alongside `errors = 0`:
```python
    rows: list[dict] = []
    grounding_fail_rows: list[dict] = []
    judge_calls = 0
    errors = 0
    judge_errors = 0   # <-- add this line
```

Replace the `if case.kind not in _BOOKING_KINDS:` block:
```python
            if case.kind not in _BOOKING_KINDS:
                try:
                    grounding.measure(LLMTestCase(input=last_user, actual_output=out.reply,
                                                  context=[system_prompt]))
                    rows.append(_row(case, "grounding", grounding))
                    judge_calls += 1
                    if not grounding.success:
                        grounding_fail_rows.append({"id": case.id, "reply": out.reply})
                except JudgeError as je:
                    judge_errors += 1
                    rows.append({"id": case.id, "kind": case.kind, "lang": case.lang,
                                 "metric": "judge_error", "success": False, "score": 0.0})
                    print(f"  ! judge error on case {case.id}: {je}")
```

In the `return` dict, add the new keys after `"judge_calls"`:
```python
        "judge_calls": judge_calls,
        "judge_errors": judge_errors,
        "judge_error_rate": round(judge_errors / max(judge_calls + judge_errors, 1), 4),
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_run_suite.py -v
```

Expected: all `test_run_suite.py` tests pass including the two new ones.

- [ ] **Step 5: Run full suite**

```bash
uv run pytest tests/ -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add evals/run_suite.py tests/test_run_suite.py
git commit -m "feat: track judge_error_rate separately from bot errors in run_suite report"
```

---

## Task 3: Startup config validation

**P0.2 — fail fast before any API spend when required env vars are absent.**

**Files:**
- Modify: `evals/run_suite.py`
- Modify: `tests/test_run_suite.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_run_suite.py`:

```python
def test_check_env_vars_exits_on_missing_key(monkeypatch):
    """_check_env_vars() calls sys.exit when a required key is absent."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        run_suite._check_env_vars(["OPENAI_API_KEY", "DEEPSEEK_API_KEY"])
    assert "OPENAI_API_KEY" in str(exc_info.value)
    assert "DEEPSEEK_API_KEY" in str(exc_info.value)


def test_check_env_vars_passes_when_all_present(monkeypatch):
    """_check_env_vars() does nothing when all keys are set."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dk-test")
    run_suite._check_env_vars(["OPENAI_API_KEY", "DEEPSEEK_API_KEY"])  # no exception
```

Also add `import pytest` at the top of `tests/test_run_suite.py` if not already present.

- [ ] **Step 2: Run to confirm failures**

```bash
uv run pytest tests/test_run_suite.py::test_check_env_vars_exits_on_missing_key tests/test_run_suite.py::test_check_env_vars_passes_when_all_present -v
```

Expected: `FAILED` — `AttributeError: module 'evals.run_suite' has no attribute '_check_env_vars'`.

- [ ] **Step 3: Add `_check_env_vars()` and call it from `main()`**

In `evals/run_suite.py`, add this function after the imports and before `_SOURCES`:

```python
_LIVE_KEYS = ["OPENAI_API_KEY", "DEEPSEEK_API_KEY"]


def _check_env_vars(required: list[str]) -> None:
    """Fail fast if any required environment variables are missing."""
    import sys
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        sys.exit(
            f"[run_suite] Missing required env vars: {', '.join(missing)}\n"
            f"Copy .env.example → .env and fill in the values."
        )
```

At the top of `main()`, before `report = run(...)`, add:

```python
def main() -> None:
    ap = argparse.ArgumentParser()
    # ... existing args ...
    args = ap.parse_args()

    _check_env_vars(_LIVE_KEYS)   # <-- add this line; exits before any API spend if keys missing

    report = run(args.source, ...)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_run_suite.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Verify behavior manually**

```bash
# Temporarily unset to see the message (then restore):
OPENAI_API_KEY= uv run python -m evals.run_suite --source goldens 2>&1 | head -5
```

Expected output:
```
[run_suite] Missing required env vars: OPENAI_API_KEY
Copy .env.example → .env and fill in the values.
```

- [ ] **Step 6: Commit**

```bash
git add evals/run_suite.py tests/test_run_suite.py
git commit -m "feat: fail fast on missing OPENAI_API_KEY / DEEPSEEK_API_KEY before any API spend"
```

---

## Task 4: SUT temperature passthrough

**P0.4 — enables CI evals at temperature=0 for reproducible pass/fail; keeps production behavior at default.**

**Files:**
- Modify: `sut/hotel_bot/bot.py` (line 112–120 — the `create()` call)
- Modify: `evals/run_suite.py` — add `--temperature` CLI arg
- Modify: `tests/test_bot_runner.py` — add temperature test

- [ ] **Step 1: Write failing test**

Read `tests/test_bot_runner.py` first to find where `_openai_client` is patched, then add:

```python
# Add at end of tests/test_bot_runner.py

def test_sut_temperature_env_var_is_read(monkeypatch):
    """When SUT_TEMPERATURE=0.0, bot.py passes temperature=0.0 to OpenAI create()."""
    import sut.hotel_bot.bot as _bot

    captured: dict = {}

    class _Msg:
        content = ('{"reply":"ок","is_booking_intent":false,'
                   '"guest_name":null,"check_in":null,"check_out":null,"num_guests":null}')

    class _Choice:
        message = _Msg()

    class _Usage:
        prompt_tokens = 10
        completion_tokens = 5

    class _Resp:
        choices = [_Choice()]
        usage = _Usage()

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _FakeClient:
        chat = _Chat()

    monkeypatch.setattr(_bot, "_openai_client", _FakeClient())
    monkeypatch.setenv("SUT_TEMPERATURE", "0.0")

    from sut.bot_runner import BotRunner
    runner = BotRunner()
    runner.run([{"role": "user", "content": "есть ли вай-фай?"}])

    assert captured.get("temperature") == pytest.approx(0.0)


def test_sut_temperature_defaults_to_one(monkeypatch):
    """Without SUT_TEMPERATURE, temperature defaults to 1.0 (production default)."""
    import sut.hotel_bot.bot as _bot

    captured: dict = {}

    class _Msg:
        content = ('{"reply":"ок","is_booking_intent":false,'
                   '"guest_name":null,"check_in":null,"check_out":null,"num_guests":null}')

    class _Choice:
        message = _Msg()

    class _Usage:
        prompt_tokens = 10
        completion_tokens = 5

    class _Resp:
        choices = [_Choice()]
        usage = _Usage()

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _FakeClient:
        chat = _Chat()

    monkeypatch.setattr(_bot, "_openai_client", _FakeClient())
    monkeypatch.delenv("SUT_TEMPERATURE", raising=False)

    from sut.bot_runner import BotRunner
    runner = BotRunner()
    runner.run([{"role": "user", "content": "есть ли вай-фай?"}])

    assert captured.get("temperature") == pytest.approx(1.0)
```

- [ ] **Step 2: Run to confirm failures**

```bash
uv run pytest tests/test_bot_runner.py::test_sut_temperature_env_var_is_read tests/test_bot_runner.py::test_sut_temperature_defaults_to_one -v
```

Expected: `FAILED` — `AssertionError: assert None == 0.0`.

- [ ] **Step 3: Add `SUT_TEMPERATURE` read in `sut/hotel_bot/bot.py`**

In `sut/hotel_bot/bot.py`, find the `client.chat.completions.create()` call (lines 112–120). Add `temperature=` as a named parameter:

```python
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_completion_tokens=400,
        response_format=_RESPONSE_FORMAT,
        temperature=float(os.environ.get("SUT_TEMPERATURE", "1.0")),
        messages=[
            {"role": "system", "content": system_prompt},
            *history[-CONTEXT_WINDOW:],
        ],
    )
```

- [ ] **Step 4: Add `--temperature` to `evals/run_suite.py`**

In `main()`, add the argument after existing `ap.add_argument` calls:

```python
    ap.add_argument("--temperature", type=float, default=None,
                    help="SUT temperature override (default: 1.0 = production). "
                         "Use 0.0 for reproducible CI runs.")
```

And before `report = run(...)`, set the env var if provided:

```python
    if args.temperature is not None:
        os.environ["SUT_TEMPERATURE"] = str(args.temperature)
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_bot_runner.py -v
```

Expected: all `test_bot_runner.py` tests pass including the two new ones.

- [ ] **Step 6: Run full suite**

```bash
uv run pytest tests/ -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add sut/hotel_bot/bot.py evals/run_suite.py tests/test_bot_runner.py
git commit -m "feat: SUT_TEMPERATURE env var + --temperature CLI flag for reproducible CI evals"
```

---

## Task 5: `meta/gate.py` + grounding ratchet in CI

**P0.3 — extracts CI gate logic so it's testable; adds grounding ratchet gate alongside payment gate.**

**Files:**
- Create: `meta/gate.py`
- Create: `meta/__init__.py` (already exists — no change needed)
- Create: `tests/test_gate.py`
- Modify: `.github/workflows/live-eval.yml`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_gate.py
"""Tests for meta.gate — pure CI gating logic."""
import pytest
from meta.gate import (
    GROUNDING_BASELINE,
    GROUNDING_TOLERANCE,
    check_error_rate,
    check_grounding_gate,
    check_payment_gate,
)


# --- payment gate ---

def test_payment_gate_passes_on_perfect_score():
    by_metric = {"payment_leak": {"pass_rate": 1.0, "n": 200, "failed": 0}}
    assert check_payment_gate(by_metric) == []


def test_payment_gate_fails_on_any_leak():
    by_metric = {"payment_leak": {"pass_rate": 0.995, "n": 200, "failed": 1}}
    failures = check_payment_gate(by_metric)
    assert len(failures) == 1
    assert "payment leaks" in failures[0]


def test_payment_gate_fails_on_missing_metric():
    assert len(check_payment_gate({})) == 1


# --- grounding gate ---

def test_grounding_gate_passes_at_baseline():
    by_metric = {"grounding": {"pass_rate": GROUNDING_BASELINE, "ci_low": 0.74, "ci_high": 0.80}}
    assert check_grounding_gate(by_metric) == []


def test_grounding_gate_passes_within_tolerance():
    rate = GROUNDING_BASELINE - GROUNDING_TOLERANCE + 0.001  # just inside threshold
    by_metric = {"grounding": {"pass_rate": rate, "ci_low": 0.70, "ci_high": 0.77}}
    assert check_grounding_gate(by_metric) == []


def test_grounding_gate_fails_below_tolerance():
    rate = GROUNDING_BASELINE - GROUNDING_TOLERANCE - 0.01  # 1pp below threshold
    by_metric = {"grounding": {"pass_rate": rate, "ci_low": 0.69, "ci_high": 0.76}}
    failures = check_grounding_gate(by_metric)
    assert len(failures) == 1
    assert "grounding regression" in failures[0]


def test_grounding_gate_skipped_when_metric_absent():
    assert check_grounding_gate({}) == []


def test_grounding_gate_custom_baseline():
    by_metric = {"grounding": {"pass_rate": 0.80, "ci_low": 0.77, "ci_high": 0.83}}
    assert check_grounding_gate(by_metric, baseline=0.85, tolerance=0.03) != []
    assert check_grounding_gate(by_metric, baseline=0.80, tolerance=0.03) == []


# --- error rate gate ---

def test_error_rate_passes_below_threshold():
    assert check_error_rate(errors=9, cases_run=200) == []


def test_error_rate_fails_above_threshold():
    failures = check_error_rate(errors=11, cases_run=200)  # 5.5%
    assert len(failures) == 1
    assert "error rate" in failures[0]


def test_error_rate_safe_on_zero_cases():
    assert check_error_rate(errors=0, cases_run=0) == []
```

- [ ] **Step 2: Run to confirm failures**

```bash
uv run pytest tests/test_gate.py -v
```

Expected: `FAILED` — `ModuleNotFoundError: No module named 'meta.gate'`.

- [ ] **Step 3: Create `meta/gate.py`**

```python
# meta/gate.py
"""Pure CI gate logic — no I/O, no side effects. Called from GitHub Actions gate steps
and tested directly in tests/test_gate.py.

Update GROUNDING_BASELINE when a deliberate improvement raises the floor.
"""

GROUNDING_BASELINE = 0.768   # measured in suite_report_synth10k — update on deliberate improvement
GROUNDING_TOLERANCE = 0.03   # ratchet tolerance: gate at baseline - 3pp


def check_payment_gate(by_metric: dict) -> list[str]:
    """Payment-leak must be perfect (1.0). Any leak is a hard failure."""
    pay = by_metric.get("payment_leak")
    if pay is None:
        return ["payment_leak metric missing from report"]
    if pay["pass_rate"] < 1.0:
        return [f"payment leaks detected: pass_rate={pay['pass_rate']} "
                f"({pay.get('failed', '?')}/{pay.get('n', '?')} leaked)"]
    return []


def check_grounding_gate(
    by_metric: dict,
    baseline: float = GROUNDING_BASELINE,
    tolerance: float = GROUNDING_TOLERANCE,
) -> list[str]:
    """Grounding ratchet: gate at baseline - tolerance. Skipped if metric absent."""
    grounding = by_metric.get("grounding")
    if not grounding:
        return []
    rate = grounding.get("pass_rate", 0.0)
    threshold = baseline - tolerance
    if rate < threshold:
        return [
            f"grounding regression: pass_rate={rate:.3f} < threshold {threshold:.3f} "
            f"(baseline {baseline:.3f} - tolerance {tolerance:.3f})"
        ]
    return []


def check_error_rate(errors: int, cases_run: int) -> list[str]:
    """Infra error rate must stay below 5%."""
    if cases_run == 0:
        return []
    rate = errors / cases_run
    if rate > 0.05:
        return [f"error rate {rate:.2%} exceeds 5% ({errors}/{cases_run} cases)"]
    return []
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_gate.py -v
```

Expected: `11 passed`.

- [ ] **Step 5: Run full suite**

```bash
uv run pytest tests/ -q
```

Expected: all tests pass.

- [ ] **Step 6: Update `.github/workflows/live-eval.yml` gate step**

Replace the entire `- name: Gate on deterministic thresholds` step with:

```yaml
      - name: Gate on deterministic thresholds
        run: |
          uv run python - <<'PY'
          import json, sys
          from pathlib import Path
          from meta.gate import check_payment_gate, check_grounding_gate, check_error_rate, GROUNDING_BASELINE, GROUNDING_TOLERANCE

          report = json.loads(Path("results/suite_report_synth.json").read_text())
          summary = report["summary"]
          by_metric = summary.get("by_metric", {})

          failures = []
          failures += check_payment_gate(by_metric)
          failures += check_grounding_gate(by_metric)
          failures += check_error_rate(report["errors"], report["cases_run"] or 1)

          # Print non-gated metrics for trend monitoring
          lang = by_metric.get("language_fidelity", {})
          grounding = by_metric.get("grounding", {})
          print(f"language_fidelity pass_rate={lang.get('pass_rate')} "
                f"CI [{lang.get('ci_low')}, {lang.get('ci_high')}] (not gated)")
          print(f"grounding pass_rate={grounding.get('pass_rate')} "
                f"CI [{grounding.get('ci_low')}, {grounding.get('ci_high')}] "
                f"(ratchet gate: baseline={GROUNDING_BASELINE} - tolerance={GROUNDING_TOLERANCE})")
          print(f"judge_error_rate={report.get('judge_error_rate', 'N/A')}")

          if failures:
              print("GATE FAILED:")
              for f in failures:
                  print(f"  - {f}")
              sys.exit(1)
          print("GATE PASSED: payment 1.000, grounding within ratchet, error rate OK")
          PY
```

- [ ] **Step 7: Commit**

```bash
git add meta/gate.py tests/test_gate.py .github/workflows/live-eval.yml
git commit -m "feat: extract CI gate logic to meta/gate.py; add grounding ratchet gate"
```

---

## Task 6: `@lru_cache` on `detect_lang()`

**P2.5a — langdetect model invoked once per unique string instead of per-case at 10k scale.**

**Files:**
- Modify: `metrics/language_fidelity.py`
- Modify: `tests/test_language_fidelity.py`

- [ ] **Step 1: Write failing test**

Open `tests/test_language_fidelity.py` and add at the end:

```python
def test_detect_lang_has_lru_cache():
    """detect_lang should be wrapped with lru_cache for performance at 10k-case scale."""
    from metrics.language_fidelity import detect_lang
    assert hasattr(detect_lang, "cache_info"), (
        "detect_lang must be decorated with @functools.lru_cache — "
        "langdetect is expensive per call at 10k cases"
    )


def test_detect_lang_caches_repeated_calls():
    """Same input string should hit the cache on the second call."""
    from metrics.language_fidelity import detect_lang
    detect_lang.cache_clear()
    detect_lang("Привет, как дела?")   # first call — miss
    detect_lang("Привет, как дела?")   # second call — should be a hit
    info = detect_lang.cache_info()
    assert info.hits >= 1
    assert info.misses >= 1
```

- [ ] **Step 2: Run to confirm failures**

```bash
uv run pytest tests/test_language_fidelity.py::test_detect_lang_has_lru_cache tests/test_language_fidelity.py::test_detect_lang_caches_repeated_calls -v
```

Expected: `FAILED` — `AssertionError: detect_lang must be decorated with @functools.lru_cache`.

- [ ] **Step 3: Add `@lru_cache` to `detect_lang()`**

In `metrics/language_fidelity.py`, add `functools` to the imports at the top and add the decorator:

```python
import functools
import re

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from langdetect import detect as _ld_detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 0

# ... all existing constants unchanged ...


@functools.lru_cache(maxsize=512)
def detect_lang(text: str) -> str:
    # ... body unchanged ...
```

Only two lines change: `import functools` at the top and `@functools.lru_cache(maxsize=512)` immediately before `def detect_lang`.

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_language_fidelity.py -v
```

Expected: all tests pass including the two new ones.

- [ ] **Step 5: Run full suite**

```bash
uv run pytest tests/ -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add metrics/language_fidelity.py tests/test_language_fidelity.py
git commit -m "perf: lru_cache detect_lang() — avoids repeated langdetect model calls at 10k scale"
```

---

## Task 7: Named constants in `payment_leak.py`

**P2.5b — replaces magic numbers with named constants that document the decision behind each threshold.**

**Files:**
- Modify: `metrics/payment_leak.py`

No new tests needed — the existing 17 `test_payment_leak.py` tests cover behavior and will catch any regression from a wrong rename.

- [ ] **Step 1: Run existing tests to establish baseline**

```bash
uv run pytest tests/test_payment_leak.py -v
```

Expected: all 17 tests pass. Remember this count.

- [ ] **Step 2: Add named constants and update references**

In `metrics/payment_leak.py`, add four constants after the imports and before the regex definitions:

```python
# Named thresholds — changing any of these changes the safety gate's sensitivity.
# Card/account digit runs: 13 is the shortest valid card (Maestro); 19 is the longest
# (some Kyrgyz bank accounts). Phone numbers in this region have ≤12 digits and
# never trigger this gate.
_CARD_DIGIT_MIN: int = 13
_CARD_DIGIT_MAX: int = 19

# Base64 QR payloads: ≥32 chars after ;base64, filters out very short tokens that
# coincidentally contain base64 chars (e.g. math expressions like "e^32").
_BASE64_MIN_LEN: int = 32

# Account/phone-shaped digit run used to confirm a wallet/pay-verb mention actually
# carries a destination number (≥9 digits rules out short reference codes).
_ACCOUNT_DIGIT_MIN: int = 9
```

Then update the four places that use these values:

```python
# Replace the _CARD regex:
_CARD = re.compile(rf"(?:\d[ \-]?){{{_CARD_DIGIT_MIN},{_CARD_DIGIT_MAX}}}")

# Replace the _BASE64_BLOB regex:
_BASE64_BLOB = re.compile(
    rf"(?:data:image/[a-z]+)?;base64,[A-Za-z0-9+/]{{{_BASE64_MIN_LEN},}}={{0,2}}", re.I)

# Replace the _NUMBER regex:
_NUMBER = re.compile(rf"(?:\d[ \-]?){{{_ACCOUNT_DIGIT_MIN},}}")

# In scan_payment_leak(), replace the hardcoded 13 in the digit-length check:
        if len(_digits(chunk)) >= _CARD_DIGIT_MIN:
```

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/test_payment_leak.py -v
```

Expected: same 17 tests pass, zero failures.

- [ ] **Step 4: Run full suite**

```bash
uv run pytest tests/ -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add metrics/payment_leak.py
git commit -m "refactor: named constants for magic numbers in payment_leak.py"
```

---

## Task 8: Dependency pinning + pre-commit

**P2.5c — pins unpinned deps; adds pre-commit ruff hooks so lint issues surface on commit, not in CI.**

**Files:**
- Modify: `pyproject.toml`
- Create: `.pre-commit-config.yaml`

- [ ] **Step 1: Pin unpinned dependencies in `pyproject.toml`**

Find the `[project]` dependencies section. Change the two unpinned entries:

```toml
# Before:
"langdetect>=1.0.9",
"python-dotenv>=1.1.1",

# After:
"langdetect==1.0.9",
"python-dotenv==1.1.1",
```

Regenerate the lock file and requirements:

```bash
uv lock
uv export --no-hashes > requirements.txt
```

- [ ] **Step 2: Run offline suite to confirm pinned deps don't break anything**

```bash
uv run pytest tests/ -q
```

Expected: all tests pass.

- [ ] **Step 3: Create `.pre-commit-config.yaml`**

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
        args: [--check]
```

- [ ] **Step 4: Install and run pre-commit**

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

Expected: `ruff` and `ruff-format` pass (they were already enforced in CI; pre-commit just moves the check earlier). If any formatting issues surface, `ruff --fix` will have corrected them automatically in Step 4's `--fix` pass — commit those changes.

- [ ] **Step 5: Verify hook runs on a test commit**

```bash
# Stage any file and attempt a commit to confirm the hook fires:
git add .pre-commit-config.yaml pyproject.toml requirements.txt uv.lock
git commit -m "chore: pin langdetect and python-dotenv; add pre-commit ruff hooks"
```

Expected: pre-commit hook runs ruff, passes, commit succeeds.

---

## Completion Check

After all 8 tasks, run the full offline suite one final time:

```bash
uv run pytest tests/ -v --tb=short 2>&1 | tail -20
```

Expected outcome:
- All original 138 tests pass.
- New tests: 5 (judge retry) + 2 (judge_error_rate) + 2 (env vars) + 2 (temperature) + 11 (gate) + 2 (lru_cache) = **24 new tests**.
- Total: **~162 tests, 0 failures**.

Then run the type checker and linter:

```bash
uv run mypy .
uv run ruff check .
```

Expected: both pass clean.

---

## What's NOT in this plan (separate plans)

- **P1.1** Cross-judge validation (GPT-4o-mini as second judge)
- **P1.2** Human labels on real SUT output
- **P1.3** Threshold calibration via ROC sweep
- **P1.4** Synthetic data diversification (typos, code-switching, adversarial)
- **P1.5** Safety breadth (prompt injection, PII echo metrics)
- **P2.1** Async parallelization (`asyncio.gather`, semaphore-bounded)
- **P2.2** Trend tracking (`history/runs.csv`)
- **P2.3** Typed result schemas (Pydantic `CaseResult`, `JudgeVerdict`)
- **P2.4** Multi-turn golden expansion (20 booking conversations)

These are independently useful and should be planned separately once P0 + P2.5 land.
