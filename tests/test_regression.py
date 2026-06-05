from evals.regression_check import compare_pass_rates


def _summary(pass_rate, by_metric):
    return {"pass_rate": pass_rate, "by_metric": by_metric}


def test_detects_regression_when_weak_scores_lower():
    good = _summary(0.9, {"grounding": {"pass_rate": 0.9}, "language_fidelity": {"pass_rate": 1.0}})
    regr = _summary(0.6, {"grounding": {"pass_rate": 0.5}, "language_fidelity": {"pass_rate": 0.7}})
    v = compare_pass_rates(good, regr)
    assert v["regressed"] is True
    assert v["delta"] == -0.3
    assert v["by_metric"]["grounding"]["delta"] == -0.4


def test_no_regression_when_equal_or_better():
    good = _summary(0.8, {"grounding": {"pass_rate": 0.8}})
    regr = _summary(0.8, {"grounding": {"pass_rate": 0.8}})
    v = compare_pass_rates(good, regr)
    assert v["regressed"] is False
    assert v["delta"] == 0.0


def test_handles_metric_present_in_only_one_side():
    good = _summary(0.5, {"grounding": {"pass_rate": 0.5}})
    regr = _summary(0.5, {"payment_leak": {"pass_rate": 1.0}})
    v = compare_pass_rates(good, regr)
    assert v["by_metric"]["grounding"]["regr"] == 0.0
    assert v["by_metric"]["payment_leak"]["good"] == 0.0
