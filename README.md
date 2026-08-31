# PII Redaction Benchmark (v5)

Reference implementation of **PII_Redaction_Semantic_ve_System_Benchmark_Plani_v5**.

Two independent benchmarks over the same frozen 1000-record dataset — Stanza/BERTurk
and Regex/GLiNER are not comparable in the same scope, so there is **no single
leaderboard** (plan §1):

| Benchmark | Question | Label scope | Units under test | Output |
|---|---|---|---|---|
| **Semantic NER** (§2.1, §5) | Which model detects Turkish PERSON / LOCATION best? | `{PERSON, LOCATION}` | `stanza`, `berturk`, `gliner_tr`, `gliner_edge`, `gliner_stream` | `results/semantic_leaderboard.csv` |
| **System-Level PII** (§2.2, §6) | How well does a production detector combo cover the full taxonomy? | 13 labels (§4) | `regex_only`, `regex_gliner_tr`, `regex_gliner_edge`, `regex_gliner_stream` | `results/system_leaderboard.csv` |

Section references (`§`) point at the v5 plan.

---

## Locked contract (plan §31)

| Aspect | Decision |
|---|---|
| Model input | only `sample["text"]` — `entities` / `entity_count` / `canonical_value` / `notes` / `dataset_type` never reach a model (§3, §13) |
| Span unit | Python `str` code-point index, `[start, end)`, on the **unmodified** original text — no lowercase / casefold / whitespace-collapse / NFC / NFD before evaluation (§14) |
| Primary match | `pred.label == gold.label` **and** `intersection_length > 0` (IoU ≥ 0.5 is *not* used) (§8) |
| Alignment | optimal 1:1 bipartite: max TP count → max total IoU → higher score → smaller `start` (§10). One prediction never matches two gold entities. |
| **Primary aggregate** | **relaxed micro** P/R/F1 (§11) |
| Secondary | relaxed macro, exact micro, exact macro, macro-supported (`support ≥ 30`); `support < 30` flagged `low_support` (§11) |
| `ORGANIZATION` | not in the dataset ground truth → out of scope, excluded from every metric (§4) |
| Latency | `adapter.predict(text)` wall time only, `batch_size = 1`, `time.perf_counter_ns()`, warm-up discarded, `torch.cuda.synchronize()` around each timed call on CUDA (§15). Excludes model/dataset load, evaluation, tokenization→safe text, metric/IO. |
| Redaction | tokenization / pseudonymization; token key `(label, NFC(surface).strip())`; **no** coreference / entity linking / alias resolution (§16) |
| `canonical_value` | never used — not in inference, matching, P/R/F1, coverage or leakage; evaluator scores `text[start:end]` (§13) |
| Taxonomy | model labels normalized to canonical labels, mapping frozen before the run; risky generic maps (`national id → TCKN`, `bank account → IBAN`) are **not** applied automatically (§12) |
| Thresholds | locked before the run, `threshold_source` reported; the frozen test set is never used for training / threshold / hyperparameter / prompt tuning (§23, §30) |

**Semantic decision order (§27):** relaxed micro recall → exact micro F1 → PERSON recall → LOCATION recall → relaxed macro F1 → P95 latency → model size.

**System decision order (§28):** relaxed micro recall → PII leakage rate → exact micro F1 → macro-supported recall → precision → P95 latency → tokenization coverage → RAM/VRAM → model size → cold start.

---

## Layout (plan §19)

```
common/         dataset IO, label taxonomy + normalization
adapters/       base, regex + validators, stanza, berturk, gliner_{tr,edge,stream},
                _gliner_base, composite_adapter
evaluation/     spans.py (geometry) · alignment.py (bipartite matcher)
                metrics.py (shared engine) · semantic_metrics.py · system_metrics.py
                entity_metrics.py · leakage.py
benchmarks/     latency.py · memory.py · throughput.py · model_size.py · cold_start.py
scripts/        validate_dataset.py
                run_semantic.py / run_all_semantic.py
                run_system.py   / run_all_system.py
                _runner.py (shared inference driver) · leaderboard.py
                run_model.py / run_all.py  (legacy: single model, full taxonomy)
config/         models.yaml (semantic + shared) · systems.yaml · patterns.yaml
data/           taxonomy.json · pii_benchmark_merged_fixed.json (not committed)
results/
  raw/semantic/<model>_predictions.json        metrics/semantic/<model>_metrics.json
  raw/system/<system>_predictions.json         metrics/system/<system>_metrics.json
  semantic_leaderboard.csv                     system_leaderboard.csv
```

Deviations from the §19 tree: added `common/` and `tokenization/` (helpers / the
redaction layer, which §16 describes but does not place); `scripts/_runner.py`
and `scripts/leaderboard.py` (shared plumbing); the legacy `run_model.py` /
`run_all.py` are kept as an auxiliary full-taxonomy single-model view.

---

## Setup

<<<<<<< HEAD
```bash
pip install -r requirements.txt            # core harness — CPU, no downloads
pip install -r requirements-models.txt     # CUDA machine only: torch (+cuda), transformers, gliner, stanza
```

`config/models.yaml` sets `device: cuda` for every ML model. If PyTorch is a
CPU-only build the GLiNER `.to("cuda")` call fails fast — install a CUDA wheel
(`pip install torch --index-url https://download.pytorch.org/whl/cu128`) and
verify `python -c "import torch; print(torch.cuda.is_available())"`.
=======
Core harness (evaluation + regex baseline + scripts + tests) — CPU only, no
downloads:

```bash
pip install -r requirements.txt
```

Heavy models (BERTurk / GLiNER ×3 / Stanza) — **CUDA machine only**:

```bash
pip install -r requirements-models.txt
```

The harness code is identical on CPU and CUDA; only `device:` in
`config/models.yaml` changes.
>>>>>>> e59785055f5985db95f4735d0c15d37f3a6cdd53

---

## Run

```bash
<<<<<<< HEAD
# freeze-check the dataset (span invariant, id range, §3 distributions)
python -m scripts.validate_dataset

# Semantic NER benchmark
python -m scripts.run_all_semantic
python -m scripts.run_semantic --model gliner_tr
python -m scripts.run_semantic --model stanza --limit 50 --device cpu   # smoke test

# System-Level PII benchmark
python -m scripts.run_all_system
python -m scripts.run_system --system regex_gliner_tr
python -m scripts.run_system --system regex_only --limit 50             # smoke test (no ML)

python -m pytest
```

`regex_only` needs no ML dependencies and reproduces the regex baseline.

=======
# 1. freeze-check the dataset (span invariant, id range, §8.4/§8.2 distributions)
python -m scripts.validate_dataset

# 2. one model
python -m scripts.run_model --model regex
python -m scripts.run_model --model berturk --device cuda
python -m scripts.run_model --model regex --limit 50   # smoke test

# 3. every model in plan order (§18), refresh the leaderboard
python -m scripts.run_all
python -m scripts.run_all --only regex,berturk --continue-on-error

# tests
python -m pytest
```

>>>>>>> e59785055f5985db95f4735d0c15d37f3a6cdd53
---

## Reading the results

<<<<<<< HEAD
**`results/metrics/semantic/<model>_metrics.json`** — `semantic.relaxed_micro` (primary),
`semantic.exact_micro`, `semantic.person` / `semantic.location`
(P/R/F1 + `exact_f1`), `semantic.dataset_type`, `latency_ms`, `cold_start`,
`resources` (`model_size_mb`, `ram_load_delta_mb`, `vram_mb`), `throughput`.

**`results/metrics/system/<system>_metrics.json`** — `system.relaxed_micro` (primary),
`system.macro_supported`, `system.entity_level.{relaxed,exact}` (per-label rows
with `support` / `low_support`), `system.leakage` (`pii_leakage_rate` → target 0,
`tokenization_coverage`; wrong-label redaction counts as *not covered*),
`system.hard_negative` (`all_negative_samples` + `hard_negative_category`:
`false_positive_rate`, `clean_pass_rate`, `predicted_labels`),
`system.dataset_type`, latency / cold_start / resources.

**`results/raw/**/*_predictions.json`** — every span each model/system produced,
per sample, with `latency_ms`.

**`results/{semantic,system}_leaderboard.csv`** — one row per model / system
(plan §5.3 / §6 columns), sorted by name; upserted on every run.

---

## Adapters & systems

| model key | source | notes |
|---|---|---|
| `regex` | `config/patterns.yaml` | deterministic, CPU; structured detector for every system. `*_ID` / plate / DOB patterns enabled from observed format |
| `stanza` | `stanza` tr NER | semantic only |
| `berturk` | `savasy/bert-base-turkish-ner-cased` | semantic only; HF token-classification, word aggregation |
| `gliner_tr` | `omeryentur/gliner-pam-pii-large` | zero-shot; prompt = `[person, location]` (semantic) / 8 labels incl. structured-gap (system) |
| `gliner_edge` | `knowledgator/gliner-pii-edge-v1.0` | " |
| `gliner_stream` | `knowledgator/gliner-stream-pii-v1.0` | streaming disabled — plain full-text inference (§6.3) |

### System architecture — parallel detectors + label-domain-priority merge

```
        TEXT
     ┌───┴────┐
     ▼        ▼
   REGEX    GLiNER
     │        │
 structured  PERSON/LOCATION/ADDRESS  +  structured-gap (CUSTOMER_ID, POLICY_ID,
     │        │                          CLAIM_ID, VEHICLE_PLATE, DATE_OF_BIRTH)
     └───┬────┘
         ▼   MERGE
```

`config/systems.yaml`:
* `structured_labels` → `regex` (authoritative).
* `semantic_labels` (`PERSON`, `LOCATION`, `ADDRESS`) + `semantic_gap_labels`
  (structured labels regex cannot express reliably) → the chosen GLiNER variant.
  GLiNER's prompt uses frozen, specific phrasings (§12) via
  `common.taxonomy.GLINER_PROMPT`.
* `regex_only` has no semantic detector.

**Merge (label-domain priority):** regex owns the character territory of every
span it produced — any overlapping GLiNER span is dropped there (a GLiNER
`PERSON` over a regex `CUSTOMER_ID` disappears). GLiNER spans in the gaps regex
left are kept. Identical `(label, start, end)` from both sides → higher score.
A label with no detector at all makes every matching gold entity an FN — the
intended *system coverage* measurement (§7).

### Regex baseline notes (patterns authored from structure, never fitted to metrics, §30)

* `CUSTOMER_ID` / `POLICY_ID` / `CLAIM_ID` / `VEHICLE_PLATE` / `DATE_OF_BIRTH`
  patterns are enabled in `config/patterns.yaml` — literal prefix + fixed
  digit-group layout (`CUST-`/`MUS-`/`POL-`/`HSR-…`). Spelled-out / STT-like
  variants are out of regex scope by design.
* `IBAN` recall is bounded by the share of spec-valid 26-char values; over-length
  and spelled-out STT-like values are correctly rejected by the mod-97 validator.
* Hard-negative false positives here are mostly adversarial "looks like an ID /
  plate / phone but the sentence says it isn't" records — regex has no context.
=======
* `results/metrics/<model>_metrics.json`
  * `primary` — the three numbers that decide ranking.
  * `secondary` — macro / exact / macro-supported views.
  * `entity_level.relaxed` / `.exact` — per-label rows with `support` and
    `low_support` (§10.9). Never read a `low_support` row as strong evidence.
  * `dataset_type` — per-category relaxed/exact P/R/F1 (§10.12).
  * `hard_negative` — clean-pass rate + false-positive spans on the 150
    no-PII records (§10.5).
  * `leakage` — `pii_leakage_rate` (target 0, §10.10) and
    `tokenization_coverage` (§10.11); wrong-label redaction counts as *not
    covered* and is reported separately.
  * `latency_ms`, `throughput`, `cold_start`, `resources`.
* `results/raw/<model>_predictions.json` — every span the model produced
  (`predictions`), plus the in-scope subset actually scored
  (`predictions_in_scope`) and the discarded out-of-scope spans.
* `results/leaderboard.csv` — one row per model (§16.5), sorted by model name.

---

## Adapters

| model key | adapter | source | notes |
|---|---|---|---|
| `regex` | `RegexAdapter` | `config/patterns.yaml` | deterministic, CPU, validated spans only (score = 1.0) |
| `stanza` | `StanzaAdapter` | `stanza` tr NER | PERSON / LOCATION / ORG(→out of scope) |
| `berturk` | `BERTurkAdapter` | `savasy/bert-base-turkish-ner-cased` | HF token-classification, word aggregation |
| `gliner_tr` | `GlinerTrAdapter` | `omeryentur/gliner-pam-pii-large` | zero-shot, labels + threshold locked in config |
| `gliner_edge` | `GlinerEdgeAdapter` | `knowledgator/gliner-pii-edge-v1.0` | " |
| `gliner_stream` | `GlinerStreamAdapter` | `knowledgator/gliner-stream-pii-v1.0` | streaming disabled, plain text inference (§5.6) |

Adding a model: new adapter file + register in `adapters/__init__.py` + entry in
`config/models.yaml`. Evaluation code is untouched (§17).

### Regex baseline — known, expected limitations

Patterns are authored from PII-type **specifications** (TCKN checksum, TR IBAN
mod-97, Luhn, TR plate format), never fitted to the benchmark records (§8.4.3).
Consequences visible in results:

* `IBAN` recall is bounded by the share of spec-valid 26-char values; malformed
  (over-length) and spelled-out STT-like values are correctly rejected by the
  mod-97 validator.
* `CUSTOMER_ID` / `POLICY_ID` / `CLAIM_ID` are **disabled** in
  `config/patterns.yaml` — they need a documented institution format spec.
  Enable and fill `regex`, then record the spec source with the results.
* `PERSON` / `LOCATION` / `ADDRESS` are out of scope for regex by design (§5.1).
>>>>>>> e59785055f5985db95f4735d0c15d37f3a6cdd53
