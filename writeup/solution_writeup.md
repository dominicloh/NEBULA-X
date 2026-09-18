# NEBULA X — Train Condition Monitoring

## 1. Shared problem and user

The project addresses a shared requirement for rail condition-monitoring engineers: to process subsystem sensor data quickly, identify recordings that require attention, understand the evidence behind the decision, and produce consistent outputs for further investigation. The primary user is the rail condition-monitoring or reliability engineer.

## 2. Door subsystem

### Data
The Door task uses a single continuous test stream named `Test.csv`, and the pipeline must segment and classify operating cycles over time. The exact schema and timestamp rule remain pending confirmation from the Door Info Kit.

### Segmentation
The Door pipeline is designed to segment a continuous stream into predicted open/close cycles before classifying each segment. The interface is in place, but the real cycle detection method must be confirmed once the Info Kit is read.

### Features
Potential Door features include cycle duration, current statistics, voltage summaries, back-EMF summaries, position progression and movement-phase timing. These are candidate features only until the Info Kit confirms the signal set and cycle rules.

### Model
The Door model is a binary classifier for `Normal` vs `Abnormal resistance`. The final evaluation must use IoU-weighted F1 across temporal segments, so segment timing and class label both matter.

### IoU-weighted F1
This is the official scoring rule for the Door task. A door submission is only complete when both the predicted boundaries and predicted label match the task specification and validation rules.

### Results
No final Door results are claimed here. The project intentionally leaves this section pending until the Info Kit and trained pipeline are available.

## 3. Rail Corrugation subsystem

### Problem
Rail corrugation is a periodic wear pattern on the rail running surface, detected here from axle-box vibration and shock sensors. Each file is one 1-second recording (10,000 samples at 10,000 Hz) from 8 cars x 8 axle-box positions (128 vibration+shock channels) plus a raw speed-sensor channel, and must be classified as `Normal`, `Side I`, or `Side II`. Full confirmed facts are in `planning/rail_corrugation_data_audit.md`.

### Dataset
272 labelled training files (234 Normal / 24 Side II / 14 Side I -- a heavily imbalanced 86% / 8.8% / 5.1% split) and 68 unlabelled Test files, both with the identical 129-column schema. No missing, infinite, or duplicate values were found in either split (Stage 1 audit).

### Side mapping
Per the Info Kit, axle-box positions 1, 3, 5, 7 are on the Side I rail and positions 2, 4, 6, 8 are on the Side II rail (same rule for every car). `Normal` means both sides are clear; `Side I`/`Side II` means corrugation is present on that side only. This mapping directly shapes the feature design below.

### Feature approach
Each file is reduced to 74 deterministic features (`src/rail_corrugation/features.py`), never a per-row prediction, because the label is defined at the file level:
- Time- and frequency-domain statistics (mean, std, RMS, peak, peak-to-peak, kurtosis, crest factor, spectral energy, dominant frequency, spectral centroid, 3 broad frequency-band energy ratios) computed per channel, then **aggregated into 4 groups** -- vibration x Side I, vibration x Side II, shock x Side I, shock x Side II -- rather than kept as 128 raw per-channel columns, to avoid an oversized feature space relative to only 272 training files.
- Per-side car-level summaries (max/std of per-car RMS across the 8 cars).
- Direct **Side I vs. Side II comparison features** (difference, safe-epsilon ratio, max side energy, vibration/shock RMS difference, spectral-energy difference) -- included specifically because the label is a *comparison* between the two sides, so the model should see that comparison directly rather than re-deriving it from 64 separate channels.
- `Rotating speed` summaries, including an estimated train speed (km/h) decoded from the raw sensor pulses using the confirmed 90-tooth/0.85 m wheel geometry.
- Nothing derived from the filename or processing order is ever included as a feature.

### Class imbalance
With only 14 Side I examples, a random train/validation split could put zero Side I files in a fold. The model comparison therefore used **stratified** cross-validation throughout, and `class_weight="balanced"` (not SMOTE/resampling, to avoid leakage risk and an unnecessary dependency for this baseline -- see `planning/model_experiment_log.md`).

### Model selection: Logistic Regression
Four candidates were compared under identical repeated 5-fold stratified CV (50 folds total): `DummyClassifier` (floor, macro F1 0.308), class-weighted Logistic Regression + `StandardScaler`, class-weighted Random Forest, and class-weighted Extra Trees. **Logistic Regression was selected** -- not because it had a slightly higher score, but because it was the *only* candidate that detected both minority classes at all in out-of-fold predictions; both tree ensembles collapsed Side I recall to 0 despite class weighting. Selection was made primarily on macro F1 while explicitly checking minority-class detection, not on accuracy.

### Macro F1 validation estimate
Repeated 5-fold stratified CV, 50 evaluated folds total: **mean macro F1 0.779 ± 0.091**. A single fixed 5-fold stratified split (for one reproducible confusion matrix): **out-of-fold macro F1 0.807**, Side I recall 10/14 (0.71), Side II recall 20/24 (0.83), Side I precision 10/18 (0.56 -- 10 correct out of 18 files predicted Side I, 8 false positives). **These are cross-validation estimates on the 272 labelled training files only** -- the hidden Test-file labels are not available to the team, so no accuracy claim is made about the actual 68 Test-file predictions.

### Explainability
The deployed model is linear, so each prediction's top contributing features are computed directly as `standardised feature value x coefficient for the predicted class` and shown with a plain-language description (e.g. "Side I minus Side II vibration RMS difference") in the Streamlit app's per-file explanation panel (`app/rail_view.py`). This explains what influenced the model's output; it does not independently identify a physical root cause, and the app states this explicitly next to every explanation.

### Limitations
- Only 14 Side I training examples means the recall/precision estimates above have real fold-to-fold variability (reflected in the ±0.091 macro F1 std); a single reshuffled split could move Side I recall by one or two files.
- Random Forest and Extra Trees missing Side I entirely despite class weighting was not root-caused further in this pass.
- Frequency-band boundaries (0-500/500-2000/2000+ Hz) used in feature engineering are a plain modelling choice, not an organiser-confirmed standard.
- No claim is made about performance on the actual hidden Test set.

## 4. Shared app

The app is a single Streamlit workflow that provides a subsystem selector. The user can switch between Door and Rail Corrugation, upload the relevant data, inspect the output and download the correct prediction CSV file. The app is designed to keep the pipelines separate while sharing one interface.

## 5. Explainability

Each subsystem is designed to present evidence supporting the prediction, such as signal plots, segment timings, relevant channels and file-level metadata. The app does not claim a precise mechanical root cause from the sensor data alone.

## 6. Submission outputs

The expected package contains both attempted subsystem outputs in a single ZIP archive at the root:

```text
predictions.zip
├── door_predictions.csv
└── rail_predictions.csv
```

The archives contain no raw data and no extra folders. Each CSV must follow the official subsystem schema exactly.

**Current status:** `rail_predictions.csv` exists and has passed all validation checks (`src/rail_corrugation/validate_predictions.py`). `door_predictions.csv` is not yet available -- Door is still a scaffold (see Section 2) -- so `predictions.zip` has not been created yet; `scripts/package_predictions.py` refuses to build it until both required files exist.

## 7. Limitations

The project intentionally avoids unsupported claims. Door segmentation and classification depend on the Info Kit and trained pipeline; Rail Corrugation also depends on its own confirmed schema and model; and any final submission must be validated against the official competition rules rather than assumptions.

## 8. Team contributions

The Door subsystem owner is responsible for the Door data understanding and initial model, the Rail owner is responsible for the Rail data understanding and initial model, the app owner manages the shared visual interface, and the integration lead ensures both pipelines use consistent interfaces and validation.
