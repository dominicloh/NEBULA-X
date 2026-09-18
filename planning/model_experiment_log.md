# MODEL EXPERIMENT LOG

The purpose of this log is to record what was changed, why it was changed, and whether it improved the team’s ability to solve the two PS3 task types. Do not claim final model quality without verified labels or a documented evaluation rule.

## Evaluation plan

### Door task

Door is a temporal segment detection and binary classification problem. Final scoring uses IoU-weighted F1 across predicted and true temporal segments. This means the evaluation depends on both the predicted cycle timing and the predicted class label.

### Rail Corrugation task

Rail Corrugation is a file-level multi-class classification problem. Final scoring uses Macro F1 across file predictions.

## Door experiment summary table

| ID | Date/time | Model | Features | Key parameters | Validation method | Result | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [DOOR EXP ID] | [DATE/TIME] | [MODEL] | [FEATURES] | [PARAMETERS] | [METHOD] | [RESULT] | [KEEP/MODIFY/REJECT] |
| DOOR-003 | 2026-09-19 | Dummy most frequent | 29 detected-cycle features | Default | Repeated stratified 5-fold x 10 | Macro F1 0.421 +/- 0.000; abnormal recall 0 | Baseline only |
| DOOR-004 | 2026-09-19 | StandardScaler + balanced Logistic Regression | Same 29 features | seed 42 | Same 50 folds | Macro F1 0.9988 +/- 0.0085; abnormal recall 0.9967 | Strong alternate |
| DOOR-005 | 2026-09-19 | Balanced Random Forest | Same 29 features | 300 trees, seed 42 | Same 50 folds | Macro F1 1.000 +/- 0.000; abnormal recall 1.000 | Selected |

Stage 3 used 18,036 Train readings and automatically detected 110 cycles, each
matched exactly once to the official answer boundaries (IoU 1.000). The 29
features are cycle duration plus seven statistics for each of motor current,
motor voltage, back EMF, and door leaf position; see `src/door/features.py`.
Neither absolute time, cycle order, nor answer fields enter the model.

For the selected Random Forest, a separate fixed stratified five-fold
out-of-fold run gave macro F1 1.000, balanced accuracy 1.000, and Normal /
Abnormal resistance precision, recall, and F1 all 1.000. Confusion matrix
(rows true, columns predicted; Normal then Abnormal resistance):
`[[80, 0], [0, 30]]`. With the automatically detected exact boundaries,
the Info Kit's same-label greedy IoU-weighted F1 on these Train out-of-fold
predictions was 1.000. This is a Train estimate, not hidden Test accuracy.

The chronological stress test trained on the first 88 cycles (65 Normal,
23 abnormal) and tested on the last 22 (15 Normal, 7 abnormal): macro F1
1.000, abnormal recall 1.000. A shuffled-label fixed five-fold check gave
macro F1 0.560, below the unshuffled 1.000. These checks do not establish
generalisation to other doors or recordings. The final model was refit on all
110 Train cycles. Test inference produced 38 detected cycles: 28 predicted
Normal and 10 predicted Abnormal resistance. The Test class distribution is
not evidence of accuracy. Full aggregate metrics are in
`output/door/classification/summary.json`.

## Rail experiment summary table

| ID | Date/time | Model | Features | Key parameters | Validation method | Result | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RAIL-000 | 2026-09-18 | DummyClassifier(strategy="most_frequent") | 74 file-level features (see below) | none (predicts "Normal" every time) | Repeated 5-fold stratified CV, 50 evaluated folds total (5 folds x 10 repeats) + a separate fixed 5-fold stratified CV for out-of-fold (OOF) predictions | Macro F1 0.308 ± 0.002; Side I/II recall = 0 | REJECT (floor only) |
| RAIL-001 | 2026-09-18 | Class-weighted LogisticRegression + StandardScaler | 74 file-level features | `class_weight="balanced"`, `max_iter=5000`, `random_state=42` | Repeated 5-fold stratified CV, 50 evaluated folds total (5 folds x 10 repeats) + a separate fixed 5-fold stratified CV for out-of-fold (OOF) predictions | Macro F1 0.779 ± 0.091 (repeated); 0.807 OOF; detects both Side I (recall 0.71) and Side II (recall 0.83) | **KEEP — current best baseline** |
| RAIL-002 | 2026-09-18 | Class-weighted RandomForestClassifier | 74 file-level features | 300 trees, `class_weight="balanced"`, `random_state=42` | same as above | Macro F1 0.605 ± 0.082 (repeated); 0.588 OOF; Side I recall = **0** (misses the fault entirely) | REJECT for now (fails minority-class check) |
| RAIL-003 | 2026-09-18 | Class-weighted ExtraTreesClassifier | 74 file-level features | 300 trees, `class_weight="balanced"`, `random_state=42` | same as above | Macro F1 0.595 ± 0.071 (repeated); 0.588 OOF; Side I recall = **0** (misses the fault entirely) | REJECT for now (fails minority-class check) |

## Door planned experiments

### Experiment 0 — Simple segmentation baseline

Potential approaches:
- Candidate cycle detection from signal transitions
- Fixed window or threshold-based segmentation
- Reference to the Door Info Kit rules once confirmed

### Experiment 1 — Rule-based candidate segmentation

Use the official Door rules after reading the Info Kit.

### Experiment 2 — Segment-feature binary classifier

Possible features include:
- Cycle duration
- Mean current
- Peak current
- Current variability
- Current integral or energy
- Voltage summary
- Back-EMF summary
- Position progression
- Time spent in movement phases
- Local peaks
- Relationship between current and position

### Experiment 3 — Class-weighted classifier

Use this when the class distribution is imbalanced and the model needs explicit handling.

### Experiment 4 — Segmentation plus classification evaluation

Evaluate segment timing and class labels together using IoU-weighted F1.

## Rail planned experiments

### Experiment 0 — File-level baseline (DONE — see RAIL-000)

`DummyClassifier(strategy="most_frequent")` run through the same CV pipeline as every
other candidate, to confirm any real model is actually beating the "always predict
Normal" floor. Macro F1 0.308 (matches the Info Kit's worked example of ~0.33 for an
always-Normal model).

### Experiment 1 — Vibration and shock feature model (DONE — see `src/rail_corrugation/features.py`)

Implemented in `extract_rail_features()`. Each of the 272 Train CSVs (10,000 rows x
129 columns) is reduced to **74 features**:
- Per-channel time/frequency stats (mean, std, RMS, abs peak, peak-to-peak, kurtosis,
  crest factor, spectral energy, dominant frequency, spectral centroid, 3 broad
  frequency-band energy ratios) computed for all 128 vibration/shock channels, then
  **aggregated into 4 groups** (vibration x Side I, vibration x Side II, shock x Side I,
  shock x Side II) rather than kept as 128 x 13 raw columns — keeps the feature count
  small relative to the 272-file sample size.
- Per-side car-level summaries (max/std of per-car RMS across the 8 cars), "where
  useful" per the brief, without exploding to one column per car.
- Direct **Side I vs. Side II comparison features**: difference, safe-epsilon ratio,
  max side energy, vibration RMS difference, shock RMS difference, spectral energy
  difference — these exist because the label itself is defined by comparing the two
  sides, so the model should see that comparison directly rather than having to
  re-derive it from 64 separate channels.
- `Rotating speed` summaries: mean, std, toggle count, and an estimated km/h derived
  from the confirmed 90-tooth/0.85 m wheel-diameter sensor description in the Info Kit.
- **No filename, file index, or processing-order-derived value is ever included** —
  verified in code (features are computed purely from signal columns) and by removing
  the filename before the feature matrix reaches any model.

Column-name parsing (`parse_signal_column`, `validate_rail_columns`) is regex-validated
against the exact confirmed header text and raises `ValueError` on any unrecognised
column rather than silently skipping or guessing a mapping.

### Experiment 2 — Class-weighted or balanced classifier (DONE — see RAIL-001/002/003)

All three real candidates (Logistic Regression, Random Forest, Extra Trees) use
`class_weight="balanced"` — no resampling (e.g. SMOTE) was used for this baseline, to
avoid leakage risk and an unnecessary new dependency (see `model.py` docstring).
Logistic Regression is the only candidate that actually detects both minority classes;
both tree ensembles collapse Side I recall to 0 despite class weighting — worth deeper
investigation before relying on tree models here.

### Experiment 3 — Macro F1 evaluation (DONE — see Results below)

Validation method: **repeated five-fold stratified cross-validation, 50 evaluated folds
in total** (`RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)` = 5
folds x 10 repeats = 50 fold fits), used to measure macro F1's mean and spread given
only 14 Side I files. Separately, a single fixed `StratifiedKFold(n_splits=5,
shuffle=True, random_state=42)` run once via `cross_val_predict` gives one reproducible
out-of-fold (OOF) prediction per file per model — this is what the per-class
precision/recall/F1, balanced accuracy, and confusion matrix are computed from (the
repeated CV above gives a distribution of scores, not one prediction per file, so it
cannot directly produce a confusion matrix). Full outputs saved under
`output/rail_corrugation/baseline/`.

#### Results (2026-09-18 run, `python src/rail_corrugation/train.py`)

| Model | Mean Macro F1 (repeated CV) | Std | OOF Macro F1 | Balanced accuracy (OOF) | Side I recall | Side II recall | Detects both minority classes? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DummyMostFrequent | 0.308 | 0.002 | 0.308 | 0.333 | 0.00 | 0.00 | No |
| **LogisticRegression** | **0.779** | 0.091 | **0.807** | 0.833 | 0.71 | 0.83 | **Yes** |
| RandomForest | 0.605 | 0.082 | 0.588 | 0.569 | 0.00 | 0.71 | No |
| ExtraTrees | 0.595 | 0.071 | 0.588 | 0.588 | 0.00 | 0.71 | No |

Class-imbalance handling: `class_weight="balanced"` only (no SMOTE/resampling this
stage). Limitations: only 14 Side I examples in the entire training set means the OOF
confusion matrix's Side I numbers (10 correct / 4 missed out of the 14 true Side I
files, i.e. recall 10/14 = 0.71) come from a very small sample — a single reshuffled
split could move Side I recall by one or two files. Side I **precision** (0.56) means
something different and is worth stating precisely rather than just as a decimal: the
model predicted "Side I" for **18 files total in the out-of-fold run — 10 of those were
correct and 8 were false positives** (7 true-Normal files and 1 true-Side-II file
misclassified as Side I; see `output/rail_corrugation/baseline/confusion_matrix.png`),
i.e. precision = 10/18 = 0.56. Both the recall and precision numbers come from the same
14-vs-18 file counts, confirmed directly from the stored confusion matrix and
`model_comparison.csv`, not recomputed from the rounded decimals. The repeated-CV std
of ~0.09 on macro F1 for Logistic Regression reflects the same small-sample
uncertainty and should be quoted alongside the mean, not the mean alone. Random Forest
and Extra Trees achieving *higher* macro F1 than Logistic Regression on some individual
folds but *lower* on average, while missing Side I recall entirely in the OOF run, is
exactly the "don't trust accuracy/majority-driven metrics alone" trap the brief warns
about — both were rejected for that reason despite non-trivial macro F1 numbers.

## Final-model decision

| Subsystem | Candidate direction | Selected? | Notes |
| --- | --- | --- | --- |
| Door | Simple segmentation baseline | [YES/NO] | Not preselected until Info Kit review |
| Door | Rule-based candidate segmentation | [YES/NO] | [TO FILL] |
| Door | Segment-feature binary classifier | [YES/NO] | [TO FILL] |
| Rail | File-level feature baseline | YES | 74-feature extractor in `src/rail_corrugation/features.py`; see Experiment 1 |
| Rail | Multi-class classifier (Logistic Regression, class-weighted) | YES (baseline only, not final) | Best of 4 candidates on macro F1 and the only one detecting both fault classes; see RAIL-001. Not yet used to generate test predictions — that is a later stage. |
| Rail | Multi-class classifier (Random Forest / Extra Trees, class-weighted) | NO (for now) | Both miss Side I entirely (recall 0) in OOF evaluation despite class weighting; see RAIL-002/003. Worth revisiting with different features/tuning before ruling out permanently. |

## Reproducibility

*(Rail Corrugation Stage 2 baseline — not yet a locked final model; this documents the current validation baseline.)*

- Random seed: `42` (`RANDOM_STATE` in `src/rail_corrugation/model.py`, used for every model and both CV schemes)
- Python version: 3.13.9
- Library versions: pandas 2.3.3, numpy 2.2.6, scikit-learn 1.8.0, matplotlib 3.10.8, seaborn 0.13.2 (all already present in the project environment; no new dependencies added)
- Final feature list: 74 features, listed in full in `output/rail_corrugation/baseline/feature_names.txt` (generated by `src/rail_corrugation/features.py::extract_rail_features`)
- Final parameters (current best baseline, Logistic Regression): `class_weight="balanced"`, `max_iter=5000`, `random_state=42`, features scaled with `StandardScaler` fit inside each CV fold only
- Training-data description: 272 Rail Corrugation Train files (234 Normal / 24 Side II / 14 Side I), 10,000 rows x 129 columns each, 10,000 Hz sampling, 1 second/file — see `planning/rail_corrugation_data_audit.md` for the full Stage 1 audit
- Final model location: not yet saved to disk (no `.joblib`/`.pkl` produced this stage — this stage only cross-validates candidates; a persisted model belongs to a later "finalise baseline" stage)
- Exact run command: `python src/rail_corrugation/train.py` (outputs under `output/rail_corrugation/baseline/`; add `--n-repeats N` to change the repeated-CV cost/precision trade-off)

---

Keep this log simple, factual, and easy to update during the project timeline. The main goal is understanding what changed, why it changed, and which evaluation rule governs the final result.
