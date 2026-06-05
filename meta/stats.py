# meta/stats.py
"""Pure agreement statistics for judge validation. No dependencies."""


def cohens_kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa_true = sum(a) / n
    pb_true = sum(b) / n
    pe = pa_true * pb_true + (1 - pa_true) * (1 - pb_true)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def confusion_matrix(human: list[bool], judge: list[bool]) -> tuple[int, int, int, int]:
    tp = sum(1 for h, j in zip(human, judge) if h and j)
    tn = sum(1 for h, j in zip(human, judge) if not h and not j)
    fp = sum(1 for h, j in zip(human, judge) if not h and j)
    fn = sum(1 for h, j in zip(human, judge) if h and not j)
    return tp, tn, fp, fn
