# NEBULA X — Train Condition Monitoring

## 1. Shared problem and user

The project addresses a shared requirement for rail condition-monitoring engineers: to process subsystem sensor data quickly, identify recordings that require attention, understand the evidence behind the decision, and produce consistent outputs for further investigation. The primary user is the rail condition-monitoring or reliability engineer.

## 2. Door subsystem

### Data
The Door task uses a single continuous Test stream and 17 confirmed input columns. Train has 18,036 readings and 110 labelled cycles (80 Normal, 30 Abnormal resistance); Test has 6,253 unlabelled readings.

### Segmentation
The frozen Train-derived detector splits only when consecutive source timestamps differ by more than 100 ms. It preserves original timestamp text and source order. On Train, 110 detected cycles matched all 110 official cycles one-to-one with mean, median, and minimum IoU 1.000 and zero boundary error. On Test, it found 38 ordered, non-overlapping cycles.

### Features
The final 29 features are duration plus mean, standard deviation, minimum, maximum, range, squared-signal energy, and first-to-last slope for each of four channels: motor current, voltage, back EMF, and leaf position. The feature matrix excludes labels, answer-file columns, absolute time, and cycle index. Features are extracted from automatically detected boundaries on both Train and Test.

### Model
Train-only model selection compared a most-frequent Dummy (mean macro F1 0.421), scaled balanced Logistic Regression (0.9988 +/- 0.0085), and balanced Random Forest (1.000 +/- 0.000) using repeated stratified five-fold CV over 10 repeats. All learned scaling is inside the Logistic Regression Pipeline. Random Forest was selected and refit on all 110 detected and exactly matched Train cycles. A separate fixed five-fold out-of-fold run gave Normal and Abnormal resistance precision, recall, and F1 of 1.000, with confusion matrix `[[80, 0], [0, 30]]`; macro F1 and balanced accuracy were 1.000. A chronological 88/22-cycle holdout gave macro F1 1.000. Shuffled-label out-of-fold macro F1 fell to 0.560. Small sample size and shared-stream dependence limit these estimates.

### IoU-weighted F1
The official Info Kit matches same-label overlapping segments greedily by descending IoU, then computes the harmonic mean of soft recall and precision from summed IoU credit. Using automatic Train boundaries and out-of-fold labels, this score was 1.000. Segmentation IoU and classification macro F1 are reported separately above to show their contributions. Hidden Test scoring is unavailable.

### Results
The frozen model at `models/door_model.joblib` generated 38 Test predictions (28 Normal, 10 Abnormal resistance). `predictions/door_predictions.csv` contains only `start_time,end_time,prediction` and passed validation against the detector's exact Test boundaries. The Test class distribution does not measure accuracy. The Door Streamlit page validates uploads, displays cycle-level confidence and signal evidence, and downloads the validated submission. Model confidence is a review cue, not a safety threshold; signal evidence does not prove mechanical root cause.

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

**Current status:** Both `rail_predictions.csv` and `door_predictions.csv` exist and pass their subsystem validators. Packaging and final submission remain separate team tasks.

## 7. Limitations

The project reports Train validation and structural Test checks separately. Hidden Test labels are unavailable, and real-world use requires validation across doors, routes, and recording conditions.

## 8. Team contributions

The Door subsystem owner is responsible for the Door data understanding and initial model, the Rail owner is responsible for the Rail data understanding and initial model, the app owner manages the shared visual interface, and the integration lead ensures both pipelines use consistent interfaces and validation.
