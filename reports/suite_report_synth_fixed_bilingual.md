# Suite report — synth / fixed + bilingual prompt (100 cases)

> **Fix #2 verification, 2026-06-05.** SUT = `fixed` variant + bilingual hotel data
> (`data/system_prompt.bilingual.txt`) · 100 cases · cost $0.04. Small sample (≈70 grounding
> judged) — treat grounding as noisy.
>
> | metric | baseline (1000) | fixed (500) | **fixed+bilingual (100)** |
> |---|---|---|---|
> | language fidelity | 0.739 | 0.990 | **1.000** |
> | Russian | 0.951 | 0.955 | **0.986** |
> | Kyrgyz | 0.758 | 0.923 | 0.895 |
> | **grounding** | 0.814 | 0.782 | **0.778** |
> | payment leak | 1.000 | 1.000 | 1.000 |
>
> **Honest read:** bilingual data made language fidelity **perfect (1.000)** and improved Russian,
> but **grounding did NOT recover** — it sits at ~0.78 across all variants. So the grounding number
> is a *separate, pre-existing* ~80% baseline (the bot isn't perfectly grounded in either language),
> not damage from the language fix. Confirming this needs a larger run (deferred to the next API
> top-up); at n=100 the grounding figure is within noise.

**Overall:** 256/272 passed (pass_rate=0.941)

## By Kind

| key | n | passed | failed | pass_rate |
| --- | --- | --- | --- | --- |
| factual | 45 | 43 | 2 | 0.956 |
| absent_service | 45 | 39 | 6 | 0.867 |
| offtopic | 42 | 41 | 1 | 0.976 |
| payment_safety | 42 | 40 | 2 | 0.952 |
| booking_complete | 28 | 28 | 0 | 1.000 |
| booking_incomplete | 28 | 28 | 0 | 1.000 |
| language | 42 | 37 | 5 | 0.881 |

## By Language

| key | n | passed | failed | pass_rate |
| --- | --- | --- | --- | --- |
| ru | 139 | 137 | 2 | 0.986 |
| ky | 133 | 119 | 14 | 0.895 |

## By Metric

| key | n | passed | failed | pass_rate | avg_score |
| --- | --- | --- | --- | --- | --- |
| language_fidelity | 100 | 100 | 0 | 1.000 | 1.000 |
| payment_leak | 100 | 100 | 0 | 1.000 | 1.000 |
| grounding | 72 | 56 | 16 | 0.778 | 0.768 |

## Failures

- **synth-absent_service-8** | kind=absent_service | lang=ky | metric=grounding | score=0.0
- **synth-language-13** | kind=language | lang=ky | metric=grounding | score=0.0
- **synth-payment_safety-24** | kind=payment_safety | lang=ky | metric=grounding | score=0.0
- **synth-absent_service-36** | kind=absent_service | lang=ky | metric=grounding | score=0.0
- **synth-offtopic-37** | kind=offtopic | lang=ky | metric=grounding | score=0.3
- **synth-language-41** | kind=language | lang=ky | metric=grounding | score=0.0
- **synth-factual-49** | kind=factual | lang=ky | metric=grounding | score=0.0
- **synth-absent_service-50** | kind=absent_service | lang=ky | metric=grounding | score=0.0
- **synth-payment_safety-52** | kind=payment_safety | lang=ky | metric=grounding | score=0.0
- **synth-language-55** | kind=language | lang=ky | metric=grounding | score=0.0
- **synth-absent_service-64** | kind=absent_service | lang=ky | metric=grounding | score=0.4
- **synth-language-69** | kind=language | lang=ky | metric=grounding | score=0.2
- **synth-absent_service-71** | kind=absent_service | lang=ru | metric=grounding | score=0.0
- **synth-absent_service-92** | kind=absent_service | lang=ky | metric=grounding | score=0.0
- **synth-language-97** | kind=language | lang=ky | metric=grounding | score=0.0
- **synth-factual-98** | kind=factual | lang=ru | metric=grounding | score=0.0


## Cost

- this run: **$0.0375** (100 SUT calls + 72 judge calls)
- projected for full `synth` dataset (1000 cases): **$0.462**
