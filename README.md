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

This repository is intentionally scaffolded to be safe and explicit:

- The app supports a shared subsystem selector.
- Door and Rail Corrugation are kept separate in their own pipeline modules.
- The app stops with clear “model not ready” messages until the required Info Kit details and trained models are available.
- The prediction exports are validated against the official schemas without fabricating results.

## Repository structure

```text
nebula-x-bogie-early-warning/
├── README.md
├── TEAM_WORKFLOW.md
├── SUBMISSION_CHECKLIST.md
├── requirements.txt
├── .gitignore
├── app/
│   └── streamlit_app.py
├── data/
│   └── input/
│       └── README.md
├── docs/
│   ├── index.html
│   ├── app.js
│   ├── styles.css
│   └── data/
│       └── dashboard_data.json
├── planning/
│   ├── data_review_template.md
│   ├── hackathon.playbook.md
│   ├── model_experiment_log.md
│   ├── submission_links.md
│   └── video_pitch_template.md
├── predictions/
│   ├── README.md
│   └── [official prediction outputs]
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
│   ├── door/
│   │   ├── __init__.py
│   │   ├── preprocess.py
│   │   ├── segment.py
│   │   ├── features.py
│   │   ├── model.py
│   │   └── predict.py
│   └── rail_corrugation/
│       ├── __init__.py
│       ├── preprocess.py
│       ├── features.py
│       ├── model.py
│       └── predict.py
├── writeup/
│   └── solution_writeup.md
└── .github/
```

## Key notes

- Door uses a single continuous test stream named `Test.csv`.
- Door is temporal segment detection and binary classification.
- Rail Corrugation is file-level multi-class classification.
- Neither subsystem should claim a final model or result until the relevant Info Kit and data checks are complete.
- The shared app provides both workflows from one place; the subsystem-specific model logic stays separate.

## Quick links

- Prediction outputs: [predictions/README.md](predictions/README.md)
- Shared app: [app/streamlit_app.py](app/streamlit_app.py)
- Solution write-up: [writeup/solution_writeup.md](writeup/solution_writeup.md)
- Submission planning: [planning/hackathon.playbook.md](planning/hackathon.playbook.md)
│   ├── app.js
│   ├── index.html
│   ├── styles.css
│   └── data/
│       └── dashboard_data.json
├── notebooks/
│   └── README.md
├── planning/
│   ├── data_review_template.md
│   ├── hackathon.playbook.md
│   ├── model_experiment_log.md
│   ├── submission_links.md
│   └── video_pitch_template.md
├── predictions/
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── export_results.py
│   ├── features.py
│   ├── model.py
│   ├── preprocess.py
│   └── run_pipeline.py
├── writeup/
│   └── solution_writeup.md
└── requirements.txt
```

## 17. Installation

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

## 18. Input data

Official CSV files should be placed in `data/input/` only after the team confirms the organisers’ usage restrictions and file requirements.

Important points:

- Raw data is excluded from Git by default.
- The team must first confirm filenames, column meanings, timestamps, units, identifiers and output requirements.
- Do not imply that arbitrary CSV schemas work automatically.
- The actual prediction format must be confirmed against the official specification.

## 19. Running the pipeline

The project currently provides an inspection-first pipeline scaffold. Review [src/run_pipeline.py](src/run_pipeline.py) before running it, because the script stops safely until the real schema is known.

```bat
python src\run_pipeline.py --input data\input --dashboard-output docs\data\dashboard_data.json --predictions-output predictions\sample_predictions.csv
```

The intended stages are:

1. Load inputs
2. Validate schema
3. Clean data
4. Build features
5. Generate predictions
6. Group events
7. Calculate severity
8. Export official predictions
9. Export dashboard JSON

The workflow is intentionally conservative until the official specification is reviewed.

## 20. Running the dashboard locally

```bat
python -m http.server 8000 --directory docs
```

Then open:

```text
http://localhost:8000/
```

Public dashboard: https://dominicloh.github.io/NEBULA-X/

## 21. Submission outputs

| Requirement | Location |
| --- | --- |
| GitHub repository | [https://github.com/dominicloh/NEBULA-X](https://github.com/dominicloh/NEBULA-X) |
| Hosted prototype | [https://dominicloh.github.io/NEBULA-X/](https://dominicloh.github.io/NEBULA-X/) |
| Prediction outputs | [predictions/](predictions/) |
| Solution write-up | [writeup/solution_writeup.md](writeup/solution_writeup.md) |
| Video pitch | [LINK TO BE ADDED](#) |

## 22. Team

- Dominic Loh — Ngee Ann Polytechnic
- Ryan Koh Zhixiang
- Muhamed Aydin
- Chloe How Wanyu

## 23. Limitations

- Anomalies do not independently confirm exact faults.
- Performance depends on data quality and inspection of the official specification.
- Operating conditions may affect temperature behaviour.
- Model outputs require engineering interpretation.
- Temperature does not capture every possible bogie fault.
- Production use requires further validation with verified labels and operational testing.

## 24. Disclaimer

This is a hackathon decision-support prototype. It does not replace engineering judgement, inspection procedures or railway safety requirements.
