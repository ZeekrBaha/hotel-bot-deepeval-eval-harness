# tests/test_language_fidelity.py
from metrics.language_fidelity import detect_lang, LanguageFidelityMetric
from deepeval.test_case import LLMTestCase


def test_detect_kyrgyz_by_special_letters():
    assert detect_lang("Баасы канча, бөлмө бармы?") == "ky"


def test_detect_russian():
    assert detect_lang("Сколько стоит номер?") == "ru"


def test_detect_unknown_for_non_cyrillic():
    assert detect_lang("hello there") == "unknown"


def test_metric_passes_when_languages_match():
    tc = LLMTestCase(input="Сколько стоит номер?", actual_output="Стандарт 2500 сом за ночь.")
    m = LanguageFidelityMetric()
    m.measure(tc)
    assert m.success is True
    assert m.score == 1.0


def test_metric_fails_on_language_mismatch():
    # Kyrgyz question answered in Russian -> violation
    tc = LLMTestCase(input="Баасы канча?", actual_output="Стандартный номер стоит 2500 сом.")
    m = LanguageFidelityMetric()
    m.measure(tc)
    assert m.success is False
    assert m.score == 0.0
    assert "ky" in m.reason and "ru" in m.reason


def test_metric_passes_when_query_language_unknown():
    tc = LLMTestCase(input="ok", actual_output="Здравствуйте!")
    m = LanguageFidelityMetric()
    m.measure(tc)
    assert m.success is True  # cannot demand a language we couldn't detect
