# NEBULA X — Train Condition Monitoring

Our system provides one accessible condition-monitoring app for two rail subsystems. It analyses door motor-cycle signals to detect and classify normal or abnormal-resistance operating cycles, and analyses axle-box vibration and shock files to classify rail condition as Normal, Side I or Side II corrugation. Users can upload data, review the prediction and supporting signal evidence, and download competition-ready prediction files.

## Project facts

- Event: NEBULA X Hackathon — “The Living Railway: Future of Mobility”
- Track: PS3 — Predictive Fault Detection
- Primary user: Rail condition-monitoring or reliability engineer
- Main user need: Help rail engineers process subsystem sensor data quickly, identify which recordings require attention, understand the evidence behind each result and download consistent outputs for further investigation.
- Value proposition: Turn complex rail sensor time series into clear, explainable and downloadable condition-monitoring results.
- GitHub repository: https://github.com/dominicloh/NEBULA-X
- Shared app: Streamlit-based workflow for Door and Rail Corrugation
- Output files: `door_predictions.csv` and `rail_predictions.csv`
- Exact mechanical root cause is not claimed from the signal data alone.

## Shared problem and user

The project supports rail engineers who need to review subsystem sensor data quickly and understand when a recording deserves attention. They need clear evidence, consistent outputs and a single workflow that can handle multiple rail monitoring tasks without adding complexity.

The primary user is the rail condition-monitoring or reliability engineer.

## Comparison of the two attempted subsystems

| Subsystem | Input | Task | Output | Metric |
| --- | --- | --- | --- | --- |
| Door | Continuous motor/position stream | Segment detection + binary classification | One row per predicted cycle | IoU-weighted F1 |
| Rail Corrugation | Vibration/shock file | Three-class file classification | One row per file | Macro F1 |

These two tasks require separate models and separate validation rules, but they are delivered through one shared application.

## System workflow

```text
Select subsystem
        ↓
Door: upload continuous CSV → detect cycles → classify each segment → inspect evidence → download door_predictions.csv
Rail Corrugation: upload file(s) → extract file features → classify each file → inspect evidence → download rail_predictions.csv
```

## Current implementation status

- **Rail Corrugation is functional end to end**: a frozen, validated baseline model, a reusable feature/prediction pipeline, an official `rail_predictions.csv`, and a dedicated Streamlit page (`app/rail_view.py`) with validation, an engineer review queue, explainability and signal evidence. See **Rail Corrugation status** below for the confirmed numbers.
- **Door has a working scaffold, not yet a working pipeline.** The official Door Info Kit has been fully read and its exact schema, timestamp format, class balance and scoring rule are confirmed (see `planning/door_handoff.md`). Data loading, feature extraction, model comparison and output validation are implemented and runnable now; **cycle segmentation (`src/door/segment.py::detect_cycles`) is the one piece still to be implemented** by the Door teammate. No Door prediction results are claimed anywhere in this repository.
- The shared app (`app/streamlit_app.py`) provides one subsystem selector; Door and Rail Corrugation stay separate at the module level (`src/door/`, `src/rail_corrugation/`, `app/door_view.py`, `app/rail_view.py`) so one subsystem's work never blocks the other's.
- Prediction exports are validated against the official schemas (`src/common/validation.py`, `src/rail_corrugation/validate_predictions.py`, `src/door/validate_predictions.py`) rather than fabricated.

## Rail Corrugation status

| Item | Value |
| --- | --- |
| Feature extractor | 74 deterministic file-level features (`src/rail_corrugation/features.py`) -- see `planning/rail_corrugation_data_audit.md` and `planning/model_experiment_log.md` |
| Selected model | `StandardScaler` + class-weighted `LogisticRegression` (chosen over Random Forest / Extra Trees baselines -- see `planning/model_experiment_log.md`) |
| Validation method | Repeated 5-fold stratified cross-validation, 50 evaluated folds total, on the 272 labelled training files only |
| Validation estimate | Mean macro F1 0.779 ± 0.091 (repeated CV); out-of-fold macro F1 0.807 (fixed 5-fold) |
| Frozen model artifact | `models/rail_corrugation_model.joblib` (~10 KB; trained once, never retrained by the app) |
| Official prediction output | `predictions/rail_predictions.csv` (`file_id,prediction`, one row per official Test file, all validation checks passed) |

## Door status

Full handoff guide (confirmed dataset facts, exact schemas, safe first approach, open questions): **[planning/door_handoff.md](planning/door_handoff.md)**.

| Item | Value |
| --- | --- |
| Confirmed dataset | `Train.csv` (18,036 rows), `Test.csv` (6,253 rows), `Train_Segments_Answer.csv` (110 labelled segments: 80 Normal / 30 Abnormal resistance) -- see `planning/door_handoff.md` |
| Implemented now | Data loading + validation (`src/door/preprocess.py`), feature extraction from any valid segments (`src/door/features.py`), baseline model comparison using the *official* segments (`src/door/train.py`), output validation (`src/door/validate_predictions.py`), read-only inspection (`src/door/inspect_data.py`) |
| Classification-only validation result | Classification-only cross-validation using official ground-truth cycle boundaries achieved 1.000 macro F1. This is a proxy result and not the official end-to-end IoU-weighted F1. Automatic cycle segmentation remains unfinished. |
| Not implemented yet | Cycle segmentation (`src/door/segment.py::detect_cycles`) -- required before `predict.py` can produce real predictions from `Test.csv`. `predict.py` never uses the official ground-truth segments for prediction; it only ever calls `detect_cycles` on the uploaded stream. |
| Official metric | IoU-weighted F1 over predicted segments (not plain classification accuracy) -- see the Door Info Kit and `planning/door_handoff.md` Section 5 |
| Official prediction output | Not yet available -- `door_predictions.csv` cannot be produced until segmentation is implemented |

**These are cross-validation estimates on the 272 labelled training files only.** The hidden Test-file labels are not available to the team, so no accuracy claim is made about the actual 68 Test-file predictions -- only that the output file matches the required schema exactly.

## Repository structure

```text
nebula-x-bogie-early-warning/
├── README.md
├── TEAM_WORKFLOW.md
├── SUBMISSION_CHECKLIST.md
├── requirements.txt
├── .gitignore
├── app/
│   ├── streamlit_app.py       # shared navigation/branding; delegates to rail_view / door_view
│   ├── rail_view.py           # all Rail Corrugation UI, validation and explainability
│   └── door_view.py           # Door UI: upload validation, signal preview, pipeline status
├── data/
│   └── input/
│       └── README.md
├── docs/
│   ├── index.html
│   ├── app.js
│   ├── styles.css
│   └── data/
│       └── dashboard_data.json
├── models/
│   ├── rail_corrugation_model.joblib   # frozen, tracked (~10 KB) -- see .gitignore
│   └── door_model.joblib               # classifier checkpoint only (segmentation not done yet); gitignored
├── planning/
│   ├── data_review_template.md
│   ├── door_handoff.md
│   ├── hackathon.playbook.md
│   ├── model_experiment_log.md
│   ├── rail_corrugation_data_audit.md
│   ├── submission_links.md
│   └── video_pitch_template.md
├── predictions/
│   ├── README.md
│   └── rail_predictions.csv    # official Rail output; door_predictions.csv pending
├── scripts/
│   ├── package_predictions.py
│   └── validate_predictions.py
├── src/
│   ├── __init__.py
│   ├── export_results.py
│   ├── run_pipeline.py
│   ├── common/
│   │   ├── __init__.py
│   │   └── validation.py
│   ├── door/                   # working scaffold -- see Door status; segmentation still pending
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── inspect_data.py
│   │   ├── preprocess.py
│   │   ├── segment.py          # detect_cycles() -- the one piece still to implement
│   │   ├── features.py
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── validate_predictions.py
│   └── rail_corrugation/       # functional: inspection, features, training, prediction
│       ├── __init__.py
│       ├── inspect_data.py
│       ├── plot_samples.py
│       ├── preprocess.py
│       ├── features.py
│       ├── model.py
│       ├── train.py
│       ├── predict.py
│       └── validate_predictions.py
├── tests/
│   ├── test_rail_predict.py
│   ├── test_rail_app.py
│   └── test_door_scaffold.py
├── output/
│   ├── rail_corrugation/        # baseline metrics/plots, EDA plots (not raw data)
│   └── door/                    # classifier comparison outputs (not raw data)
└── writeup/
    └── solution_writeup.md
```

## Installation

Clone the repository:

```bat
git clone https://github.com/dominicloh/NEBULA-X.git
cd NEBULA-X
```

Windows Command Prompt:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`requirements.txt` pins `scikit-learn==1.8.0` and `joblib==1.5.3` exactly, matching the versions used to train and save `models/rail_corrugation_model.joblib` -- a mismatched scikit-learn version can fail to load a pickled model correctly.

## Running the shared Streamlit app locally

From the project root:

```bat
python -m streamlit run app/streamlit_app.py
```

This opens the shared app in your browser (default `http://localhost:8501`). Select **Rail Corrugation** to upload one or more CSV files (or a ZIP of CSVs) and see validation results, the engineer review queue, per-file explanations and signal evidence; select **Door** to see its current scaffold state.

**Hosted Streamlit app:** [LINK TO BE ADDED once deployed](#) -- not yet deployed; a teammate needs to publish this (e.g. Streamlit Community Cloud) and update this link.

## Input data

Official CSV files should be placed in `data/input/` only after the team confirms the organisers' usage restrictions and file requirements. Raw data is excluded from Git by default (`.gitignore`); do not commit organiser CSVs.

For Rail Corrugation, the confirmed dataset facts (schema, sampling rate, side mapping, class distribution) are recorded in `planning/rail_corrugation_data_audit.md` -- read that before changing any Rail-specific assumption.

## Running the Rail Corrugation pipeline directly (without the app)

```bat
python src\rail_corrugation\inspect_data.py
python src\rail_corrugation\train.py --finalize
python src\rail_corrugation\predict.py "<path to organiser Test folder>"
python src\rail_corrugation\validate_predictions.py
```

See `planning/model_experiment_log.md` for what each step produces and why.

## Running the Door pipeline directly (without the app)

```bat
python src\door\inspect_data.py
python src\door\train.py --finalize
python src\door\predict.py "<path to organiser Test.csv>"
python src\door\validate_predictions.py --predictions predictions\door_predictions.csv
```

`train.py` already works today (it trains against the *official* labelled segments). `predict.py` will raise a clear `NotImplementedError` until `src/door/segment.py::detect_cycles` is implemented -- see `planning/door_handoff.md` for exactly what's left and in what order to build it.

## Running the dashboard locally

```bat
python -m http.server 8000 --directory docs
```

Then open:

```text
http://localhost:8000/
```

Public static dashboard (separate from the Streamlit app above): https://dominicloh.github.io/NEBULA-X/

## Submission outputs

| Requirement | Location |
| --- | --- |
| GitHub repository | [https://github.com/dominicloh/NEBULA-X](https://github.com/dominicloh/NEBULA-X) |
| Hosted Streamlit app | [LINK TO BE ADDED](#) -- placeholder until deployed |
| Hosted static dashboard | [https://dominicloh.github.io/NEBULA-X/](https://dominicloh.github.io/NEBULA-X/) |
| Rail prediction output | [predictions/rail_predictions.csv](predictions/rail_predictions.csv) |
| Door prediction output | Not yet available -- Door pipeline is still a scaffold |
| Solution write-up | [writeup/solution_writeup.md](writeup/solution_writeup.md) |
| Video pitch | [LINK TO BE ADDED](#) |

## Team

- Dominic Loh — Ngee Ann Polytechnic
- Ryan Koh Zhixiang
- Muhamed Aydin
- Chloe How Wanyu

## Limitations

- A model prediction does not independently confirm a physical rail defect or any other fault; it supports engineering review and prioritisation.
- The Rail Corrugation validation numbers above are cross-validation estimates on 272 labelled training files, not a measurement against the hidden Test-file labels (unavailable to the team).
- The Rail training set is heavily imbalanced (234 Normal / 24 Side II / 14 Side I); the Side I minority class has the widest validation uncertainty (see `planning/model_experiment_log.md`).
- Door segmentation and classification are not implemented yet -- no Door result of any kind should be assumed.
- Production use of either subsystem requires further validation with verified labels and operational testing.

## Disclaimer

This is a hackathon decision-support prototype. It does not replace engineering judgement, inspection procedures or railway safety requirements. This prototype supports engineering review. A prediction does not independently confirm rail corrugation or replace railway inspection and safety procedures.
