# NEBULA X — Bogie Early Warning

An explainable early-warning system that turns bogie-temperature telemetry into prioritised maintenance alerts.

## Quick links

- Hosted dashboard: https://dominicloh.github.io/NEBULA-X/
- Prediction folder: [predictions/](predictions/)
- Solution write-up: [writeup/solution_writeup.md](writeup/solution_writeup.md)

## Project facts

- Event: NEBULA X Hackathon — “The Living Railway: Future of Mobility”
- Track: PS3 — Predictive Fault Detection
- Main focus: bogie-temperature anomaly detection
- Primary user: rail condition-monitoring or reliability engineer
- GitHub repository: https://github.com/dominicloh/NEBULA-X
- Hosted prototype: https://dominicloh.github.io/NEBULA-X/
- Team:
  - Dominic Loh — Ngee Ann Polytechnic
  - Ryan Koh Zhixiang
  - Muhamed Aydin
  - Chloe How Wanyu
- The dashboard is hosted from the `/docs` folder using GitHub Pages.
- Python performs data processing and anomaly detection.
- JSON connects the Python pipeline to the dashboard.
- Exact root-cause diagnosis and repair recommendations are out of scope.

## 1. Project summary

Our system analyses bogie-temperature telemetry to identify current and developing abnormal heating patterns. It groups abnormal readings into meaningful events, ranks affected train components by severity, and presents the supporting evidence through an interactive dashboard for rail condition-monitoring engineers.

This project supports engineering review by turning raw telemetry into prioritised evidence. The core questions for the engineer are:

- Which component requires attention?
- Is it abnormal now or gradually worsening?
- How serious and persistent is it?
- What evidence caused the alert?
- Which issue should be investigated first?

## 2. Challenge context

Trains generate large amounts of telemetry, and important changes may be hidden among normal readings. The team selected bogie-temperature anomaly detection because temperature behaviour can provide evidence that a component requires investigation, even when the exact root cause cannot be confirmed from telemetry alone.

This work is intended to support prioritisation and investigation. It does not claim that every bogie fault produces a temperature change or that temperature alone identifies the exact mechanical issue.

## 3. Problem statement

Engineers have large quantities of data and limited maintenance windows. A single unusual reading may not represent a meaningful fault, and operating or environmental conditions may affect temperature. Engineers therefore need to understand whether behaviour is temporary, persistent, isolated, repeated, stable or worsening, and they need enough supporting evidence to decide where to investigate first.

The goal is not to replace engineering judgement, but to help teams focus their attention where the evidence is strongest.

## 4. Target user

The primary user is a rail condition-monitoring or reliability engineer.

The system is designed to help that user:

- Identify current abnormalities
- Recognise developing patterns
- Locate affected components
- Prioritise inspection activity
- Compare current and historical behaviour
- See supporting evidence behind each alert
- Communicate findings to maintenance teams clearly

## 5. Proposed solution

The project uses a practical, evidence-first workflow:

1. Data validation and cleaning
2. Time-series feature engineering
3. Unsupervised anomaly detection
4. Event grouping
5. Severity prioritisation
6. Evidence-based explanations
7. Interactive dashboard

This approach highlights behaviour for investigation and does not independently confirm an exact root cause.

## 6. How it works

```text
Official bogie CSV files
        ↓
Data inspection and validation
        ↓
Cleaning and feature engineering
        ↓
Baseline comparison and anomaly detection
        ↓
Row-level predictions
        ↓
Event grouping and severity prioritisation
        ↓
Official prediction CSVs + dashboard JSON
        ↓
Interactive hosted dashboard
```

## 7. Current and developing abnormalities

### Current abnormality

The component is behaving unusually at the present time.

### Developing abnormality

The component is gradually moving away from its established behaviour and may be worsening even before the condition becomes highly abnormal.

These categories help prioritise engineering review. They are not official safety classifications, and they do not replace formal engineering assessment.

## 8. Planned temperature features

Candidate features include:

- Current temperature
- Rolling mean
- Rolling variability
- Temperature difference
- Rate of change
- Deviation from component baseline
- Difference between comparable components
- Duration
- Consecutive abnormal readings
- Operating context if available

The final feature set depends on the actual supplied data and the organiser specification.

## 9. Model approach

The initial planned model is Isolation Forest, chosen as a flexible unsupervised option for abnormal-temperature detection in time-series data.

A simple statistical baseline should be used for comparison. Additional models may be explored only if time and data support them. The final selection will consider detection performance, false alerts, explainability, runtime and robustness.

```text
Final model: [TO UPDATE AFTER MODEL EVALUATION]
Final features: [TO UPDATE AFTER DATA INSPECTION]
Final parameters: [TO UPDATE AFTER MODEL EVALUATION]
```

## 10. Event grouping

Consecutive row-level predictions are grouped into events to make the output operationally useful. A single abnormal reading may be noisy or isolated, while a grouped event provides a more meaningful indication of persistence and severity.

Potential event details include:

- Start and end time
- Duration
- Highest temperature
- Maximum deviation
- Maximum anomaly score
- Severity
- Explanation

## 11. Severity and prioritisation

The project distinguishes between two separate concepts:

- Anomaly detection: “Is this behaviour unusual?”
- Severity ranking: “How urgently should this event be reviewed?”

Possible evidence for prioritisation may include:

- Anomaly score
- Temperature deviation
- Rate of increase
- Duration
- Persistence
- Repeated abnormal behaviour
- Comparison with similar components

Severity categories are prioritisation aids and are not official railway thresholds.

## 12. Dashboard

The dashboard helps the engineer move from raw telemetry to prioritised evidence and then to a maintenance decision.

Raw telemetry → prioritised evidence → maintenance decision

Planned dashboard content includes:

- Trains requiring attention
- Current critical anomalies
- Developing warnings
- Highest-priority component
- Prioritised alert list
- Fleet or component overview
- Selected temperature trend
- Baseline
- Alert start marker
- Explanation panel
- Filters

Hosted dashboard: https://dominicloh.github.io/NEBULA-X/

## 13. What makes our solution different

- Per-component behaviour where possible
- Developing warnings alongside current abnormalities
- Event-level alerts instead of isolated row-level noise
- Maintenance prioritisation for limited engineering windows
- Evidence-based explanations that support human review
- Human-in-the-loop engineering decisions

This is not unique simply because a model is used; the value comes from combining context, event grouping, prioritisation and explainability.

## 14. Evaluation

The approach will be evaluated according to the official dataset and organiser specification. If verified labels are available, the team may consider the following measures:

- Precision
- Recall
- F1 score
- False alerts
- Verified events detected
- Event-level recall
- Early-warning lead time

If verified labels are unavailable, evaluation may rely on manual event inspection, deviation from baseline, persistence, rate of change, comparison with a statistical baseline and honest reporting of limitations.

No accuracy claim will be made unless the data supports it.

| Measure | Result |
| --- | --- |
| Final model | [TO UPDATE] |
| Verified events detected | [TO UPDATE IF LABELS EXIST] |
| Precision | [TO UPDATE IF SUPPORTED] |
| Recall | [TO UPDATE IF SUPPORTED] |
| F1 score | [TO UPDATE IF SUPPORTED] |
| Early-warning lead time | [TO UPDATE IF SUPPORTED] |

## 15. Technology stack

- Python: data processing and ML workflow
- pandas: data manipulation and inspection
- NumPy: numerical operations and time-series preparation
- scikit-learn: anomaly detection and model comparison
- Matplotlib: plotting and exploratory analysis
- Seaborn: visual exploration and reporting
- HTML: dashboard structure
- CSS: dashboard styling and responsive layout
- JavaScript: dashboard interactions and rendering
- Chart.js: trend charts and visualisation
- JSON: structured connection between Python output and the dashboard
- GitHub Pages: public dashboard hosting

## 16. Repository structure

```text
nebula-x-bogie-early-warning/
├── README.md
├── TEAM_WORKFLOW.md
├── SUBMISSION_CHECKLIST.md
├── requirements.txt
├── .gitignore
├── data/
│   └── input/
│       └── README.md
├── docs/
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
