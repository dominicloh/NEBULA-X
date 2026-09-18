# NEBULA X â€” Train Condition Monitoring

Our system provides one accessible condition-monitoring app for two rail subsystems. It analyses door motor-cycle signals to detect and classify normal or abnormal-resistance operating cycles, and analyses axle-box vibration and shock files to classify rail condition as Normal, Side I or Side II corrugation. Users can upload data, review the prediction and supporting signal evidence, and download competition-ready prediction files.

## Project facts

- Event: NEBULA X Hackathon â€” â€œThe Living Railway: Future of Mobilityâ€
- Track: PS3 â€” Predictive Fault Detection
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
        â†“
Door: upload continuous CSV â†’ detect cycles â†’ classify each segment â†’ inspect evidence â†’ download door_predictions.csv
Rail Corrugation: upload file(s) â†’ extract file features â†’ classify each file â†’ inspect evidence â†’ download rail_predictions.csv
```

## Current implementation status

- **Rail Corrugation is functional end to end**: a frozen, validated baseline model, a reusable feature/prediction pipeline, an official `rail_predictions.csv`, and a dedicated Streamlit page (`app/rail_view.py`) with validation, an engineer review queue, explainability and signal evidence. See **Rail Corrugation status** below for the confirmed numbers.
- **Door is functional end to end locally.** The frozen 100 ms detector, Train-validated classifier, official 38-cycle Test output, and Streamlit review workflow are implemented. Hidden Test labels are unavailable, so Test accuracy is unknown.
- The shared app (`app/streamlit_app.py`) provides one subsystem selector; Door and Rail Corrugation stay separate at the module level (`src/door/`, `src/rail_corrugation/`, `app/door_view.py`, `app/rail_view.py`) so one subsystem's work never blocks the other's.
- Prediction exports are validated against the official schemas (`src/common/validation.py`, `src/rail_corrugation/validate_predictions.py`, `src/door/validate_predictions.py`) rather than fabricated.

## Rail Corrugation status

| Item | Value |
| --- | --- |
| Feature extractor | 74 deterministic file-level features (`src/rail_corrugation/features.py`) -- see `planning/rail_corrugation_data_audit.md` and `planning/model_experiment_log.md` |
| Selected model | `StandardScaler` + class-weighted `LogisticRegression` (chosen over Random Forest / Extra Trees baselines -- see `planning/model_experiment_log.md`) |
| Validation method | Repeated 5-fold stratified cross-validation, 50 evaluated folds total, on the 272 labelled training files only |
| Validation estimate | Mean macro F1 0.779 Â± 0.091 (repeated CV); out-of-fold macro F1 0.807 (fixed 5-fold) |
| Frozen model artifact | `models/rail_corrugation_model.joblib` (~10 KB; trained once, never retrained by the app) |
| Official prediction output | `predictions/rail_predictions.csv` (`file_id,prediction`, one row per official Test file, all validation checks passed) |

## Door status

Full handoff guide (confirmed dataset facts, exact schemas, safe first approach, open questions): **[planning/door_handoff.md](planning/door_handoff.md)**.

| Item | Value |
| --- | --- |
| Confirmed dataset | `Train.csv` (18,036 rows), `Test.csv` (6,253 rows), `Train_Segments_Answer.csv` (110 labelled segments: 80 Normal / 30 Abnormal resistance) -- see `planning/door_handoff.md` |
| Implemented now | Order-preserving validation, 100 ms automatic segmentation, 29 sensor features per detected cycle, Train-only model selection, frozen model, official validator, and Door Streamlit review |
| Train validation | 110/110 exact detected/official matches; Random Forest repeated 5-fold x 10 macro F1 1.000 +/- 0.000; fixed out-of-fold macro F1 1.000, abnormal recall 1.000; chronological 22-cycle holdout macro F1 1.000 |
| Leakage check | Shuffled-label out-of-fold macro F1 0.560; no absolute timestamp, answer columns, row index, or cycle number in features |
| Official metric | IoU-weighted F1 over predicted segments (not plain classification accuracy) -- see the Door Info Kit and `planning/door_handoff.md` Section 5 |
| Official prediction output | `predictions/door_predictions.csv`: 38 Test cycles (28 Normal, 10 Abnormal resistance), exact detector boundaries, official validation passed; Test accuracy unknown |

**These are cross-validation estimates on the 272 labelled training files only.** The hidden Test-file labels are not available to the team, so no accuracy claim is made about the actual 68 Test-file predictions -- only that the output file matches the required schema exactly.

## Repository structure

```text
nebula-x-bogie-early-warning/
â”œâ”€â”€ README.md
â”œâ”€â”€ TEAM_WORKFLOW.md
â”œâ”€â”€ SUBMISSION_CHECKLIST.md
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ .gitignore
â”œâ”€â”€ app/
â”‚   â”œâ”€â”€ streamlit_app.py       # shared navigation/branding; delegates to rail_view / door_view
â”‚   â”œâ”€â”€ rail_view.py           # all Rail Corrugation UI, validation and explainability
â”‚   â””â”€â”€ door_view.py           # Door UI: upload validation, signal preview, pipeline status
â”œâ”€â”€ data/
â”‚   â””â”€â”€ input/
â”‚       â””â”€â”€ README.md
â”œâ”€â”€ docs/
â”‚   â”œâ”€â”€ index.html
â”‚   â”œâ”€â”€ app.js
â”‚   â”œâ”€â”€ styles.css
â”‚   â””â”€â”€ data/
â”‚       â””â”€â”€ dashboard_data.json
â”œâ”€â”€ models/
â”‚   â”œâ”€â”€ rail_corrugation_model.joblib   # frozen, tracked (~10 KB) -- see .gitignore
â”‚   â””â”€â”€ door_model.joblib               # frozen Train-selected Door classifier
â”œâ”€â”€ planning/
â”‚   â”œâ”€â”€ data_review_template.md
â”‚   â”œâ”€â”€ door_handoff.md
â”‚   â”œâ”€â”€ hackathon.playbook.md
â”‚   â”œâ”€â”€ model_experiment_log.md
â”‚   â”œâ”€â”€ rail_corrugation_data_audit.md
â”‚   â”œâ”€â”€ submission_links.md
â”‚   â””â”€â”€ video_pitch_template.md
â”œâ”€â”€ predictions/
â”‚   â”œâ”€â”€ README.md
â”‚   â””â”€â”€ rail_predictions.csv    # official Rail output; Door output also available
â”œâ”€â”€ scripts/
â”‚   â”œâ”€â”€ package_predictions.py
â”‚   â””â”€â”€ validate_predictions.py
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ __init__.py
â”‚   â”œâ”€â”€ export_results.py
â”‚   â”œâ”€â”€ run_pipeline.py
â”‚   â”œâ”€â”€ common/
â”‚   â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚   â””â”€â”€ validation.py
â”‚   â”œâ”€â”€ door/                   # validated Door segmentation and classification
â”‚   â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚   â”œâ”€â”€ config.py
â”‚   â”‚   â”œâ”€â”€ inspect_data.py
â”‚   â”‚   â”œâ”€â”€ preprocess.py
â”‚   â”‚   â”œâ”€â”€ segment.py          # frozen 100 ms detect_cycles()
â”‚   â”‚   â”œâ”€â”€ features.py
â”‚   â”‚   â”œâ”€â”€ model.py
â”‚   â”‚   â”œâ”€â”€ train.py
â”‚   â”‚   â”œâ”€â”€ predict.py
â”‚   â”‚   â””â”€â”€ validate_predictions.py
â”‚   â””â”€â”€ rail_corrugation/       # functional: inspection, features, training, prediction
â”‚       â”œâ”€â”€ __init__.py
â”‚       â”œâ”€â”€ inspect_data.py
â”‚       â”œâ”€â”€ plot_samples.py
â”‚       â”œâ”€â”€ preprocess.py
â”‚       â”œâ”€â”€ features.py
â”‚       â”œâ”€â”€ model.py
â”‚       â”œâ”€â”€ train.py
â”‚       â”œâ”€â”€ predict.py
â”‚       â””â”€â”€ validate_predictions.py
â”œâ”€â”€ tests/
â”‚   â”œâ”€â”€ test_rail_predict.py
â”‚   â”œâ”€â”€ test_rail_app.py
â”‚   â””â”€â”€ test_door_scaffold.py
â”œâ”€â”€ output/
â”‚   â”œâ”€â”€ rail_corrugation/        # baseline metrics/plots, EDA plots (not raw data)
â”‚   â””â”€â”€ door/                    # classifier comparison outputs (not raw data)
â””â”€â”€ writeup/
    â””â”€â”€ solution_writeup.md
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

This opens the shared app in your browser (default `http://localhost:8501`). Select **Rail Corrugation** for file classification or **Door** for continuous-stream cycle detection, classification, evidence review, and validated CSV download.

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

`train.py` matches automatically detected Train cycles one-to-one to official labels, evaluates three candidate classifiers, and saves a final model with `--finalize`. `predict.py` uses only the frozen detector and model for Test inference; the separate validator confirms the exact detected boundaries.

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
| Door prediction output | [predictions/door_predictions.csv](predictions/door_predictions.csv) |
| Solution write-up | [writeup/solution_writeup.md](writeup/solution_writeup.md) |
| Video pitch | [LINK TO BE ADDED](#) |

## Team

- Dominic Loh â€” Ngee Ann Polytechnic
- Ryan Koh Zhixiang
- Muhamed Aydin
- Chloe How Wanyu

## Limitations

- A model prediction does not independently confirm a physical rail defect or any other fault; it supports engineering review and prioritisation.
- The Rail Corrugation validation numbers above are cross-validation estimates on 272 labelled training files, not a measurement against the hidden Test-file labels (unavailable to the team).
- The Rail training set is heavily imbalanced (234 Normal / 24 Side II / 14 Side I); the Side I minority class has the widest validation uncertainty (see `planning/model_experiment_log.md`).
- Door Train estimates are unusually high on only 110 cycles; hidden Test accuracy is unknown, and the 100 ms gap rule requires validation on recordings without the same gaps.
- Production use of either subsystem requires further validation with verified labels and operational testing.

## Disclaimer

This is a hackathon decision-support prototype. It does not replace engineering judgement, inspection procedures or railway safety requirements. This prototype supports engineering review. A prediction does not independently confirm rail corrugation or replace railway inspection and safety procedures.
