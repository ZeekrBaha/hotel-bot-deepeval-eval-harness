# tests/test_stats.py
from meta.stats import cohens_kappa, confusion_matrix


def test_perfect_agreement_is_one():
    assert cohens_kappa([True, False, True], [True, False, True]) == 1.0


def test_total_disagreement_is_negative():
    # [T,F] vs [F,T]: systematic anti-agreement -> kappa=-1.0
    # (plan used [T,T] vs [F,F] which gives kappa=0 mathematically — corrected here)
    assert cohens_kappa([True, False], [False, True]) < 0


def test_all_same_label_returns_one():
    # pe == 1 edge case -> defined as 1.0
    assert cohens_kappa([True, True], [True, True]) == 1.0


def test_confusion_matrix_counts():
    tp, tn, fp, fn = confusion_matrix(
        human=[True, True, False, False],
        judge=[True, False, False, True])
    assert (tp, tn, fp, fn) == (1, 1, 1, 1)
