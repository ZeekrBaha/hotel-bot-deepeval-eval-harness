# metrics/language_fidelity.py
"""Deterministic language-fidelity metric: reply language must match query language.

No LLM. Heuristic over Cyrillic letter sets and common Kyrgyz words. Its blind
spots (short strings, mixed code) are precisely what the meta judge-validation
step quantifies.

Deviation from plan: added _KY_WORDS as a secondary signal for Kyrgyz texts that
lack distinctive Kyrgyz Cyrillic letters (e.g. "Баасы канча?" contains 'ы' but
no ң/ө/ү). Without this, common Kyrgyz phrases fall through to the Russian
default, causing test_metric_fails_on_language_mismatch to fail.
"""
import re

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

_KY_ONLY = set("ңөү")          # Kyrgyz Cyrillic letters not used in Russian
_RU_ONLY = set("ыэъщ")         # common in Russian, rare/absent in Kyrgyz
_CYRILLIC = lambda c: "Ѐ" <= c <= "ӿ"
# Common Kyrgyz words that don't appear in Russian; used as secondary signal
# when distinctive letters are absent.
_KY_WORDS = {"канча", "баасы", "бармы", "жок", "рахмат", "кечиресиз",
             "атыңыз", "атым", "брондоо", "бронь", "дарегиңер"}


def detect_lang(text: str) -> str:
    low = text.lower()
    if any(c in _KY_ONLY for c in low):
        return "ky"
    words = set(re.findall(r"[а-яёңөүa-z]+", low))
    if words & _KY_WORDS:
        return "ky"
    if any(c in _RU_ONLY for c in low):
        return "ru"
    if any(_CYRILLIC(c) for c in low):
        return "ru"  # Cyrillic with no distinguishing letters -> default Russian
    return "unknown"


class LanguageFidelityMetric(BaseMetric):
    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.async_mode = False

    def measure(self, test_case: LLMTestCase) -> float:
        q = detect_lang(test_case.input)
        a = detect_lang(test_case.actual_output)
        if q == "unknown" or a == "unknown":
            self.score, self.success = 1.0, True
            self.reason = f"query={q}, reply={a}: language not enforceable"
            return self.score
        self.success = q == a
        self.score = 1.0 if self.success else 0.0
        self.reason = f"query={q}, reply={a}: " + ("match" if self.success else "MISMATCH")
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Language Fidelity"
