# Root-cause analysis: the bot answers Kyrgyz queries in Russian

> Found by the harness (the 1000-case suite): Kyrgyz pass-rate 0.76 vs Russian 0.95;
> **83% of all 394 failures are Kyrgyz**, dominated by language fidelity (261/1000 replies
> in the wrong language). This doc explains *why* and proves it with two experiments.

## Verdict

**It is a prompt bug, not a model-capability limit.** `gpt-4o-mini` *can* answer in Kyrgyz —
including facts (prices, address, room types). It just isn't reliably triggered to.

## Experiment 1 — the failure pattern (28 Kyrgyz queries through the SUT)

| Flips to Russian | Stays Kyrgyz |
|---|---|
| Prices, address, room types, breakfast time — anything that **pulls DATA** from the prompt | Refusals ("Кечиресиз, мен отель боюнча…"), booking ack ("Рахмат, Арзыгул!"), "бизде QR жок" — **canned phrases** |

The split is along *data vs template*, not along query difficulty.

## Experiment 2 — intervention (force "reply in Kyrgyz")

Same queries that flipped, but with an explicit system directive `гость пишет на КЫРГЫЗСКОМ,
отвечай ТОЛЬКО на кыргызском`:

| query | baseline reply lang | forced-KY reply lang |
|---|---|---|
| Баасы канча? (price) | mixed/ru | **ky** — "Стандарт номер 2500 сом/түн…" |
| Дарегиңер кайда? (address) | ru | **ky** — "Биздин дарегибиз: Бишкек ш., Ибраимова 42" |
| Бөлмө кандай? (room types) | ru | **ky** — "Отелибизде эки түрдөгү бөлмө бар…" |
| Бассейн барбы? (absent) | ru | **ky** — "Кечиресиз, бизде бассейн жок." |
| Эртең мененки тамак качан? (breakfast) | ru | **ky** — "Эртең мененки тамак 8:00дөн 10:00гө чейин." |

**5/5 factual answers became Kyrgyz when explicitly told.** → "can't" is ruled out.

## Mechanism (two compounding causes)

1. **Language detection is delegated to the model, and it's unreliable.** The prompt says
   "if the guest writes Kyrgyz → answer Kyrgyz", but leaves *detecting* the language to the
   model. Kyrgyz and Russian share Cyrillic and the queries carry Russian loanwords
   ("номер", "бронь"); `gpt-4o-mini` (weak on low-resource Kyrgyz) frequently decides the
   conversation is Russian and answers Russian.

2. **The prompt is overwhelmingly Russian, and the hotel DATA is Russian-only.** Instructions,
   prices, address, amenities, the date stamp — all Russian. Only the canned bot phrases
   (refusal, greeting, booking ack) are given bilingually. So:
   - a templated answer has a Kyrgyz template to copy → comes out Kyrgyz ✅
   - a **factual** answer has no Kyrgyz template → the model emits the Russian data verbatim ❌

That is exactly why the failures concentrate on Kyrgyz *factual* queries.

## Fixes (by leverage)

1. **Detect the language in code, not in the model.** The harness already has `detect_lang`.
   Inject an explicit per-turn directive ("the guest writes Kyrgyz, answer only in Kyrgyz")
   into the system prompt. Deterministic language routing — cheapest, strongest fix.
   *Implemented as the `fixed` SUT variant (`sut/hotel_bot/bot_fixed.py`); verified below.*
2. **Make the hotel DATA bilingual** in the prompt so factual answers have a Kyrgyz anchor.
3. Move the language rule to the top and repeat it; add Kyrgyz few-shot examples of *factual* answers.

## Verification (close the loop)

Fix #1 was implemented as a SUT variant and re-measured with the same harness on 500 cases.
See `reports/suite_report_fixed.md` for the numbers (baseline vs fixed language-fidelity pass-rate).
