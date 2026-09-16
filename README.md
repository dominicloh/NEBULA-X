# NEBULA X Bogie Early Warning

## Project summary
Our system analyses bogie-temperature telemetry to identify current and developing abnormal heating patterns. It groups abnormal readings into events, ranks affected train components by severity, and presents the supporting evidence through an interactive dashboard for rail condition-monitoring engineers.

## Challenge context
This repository is being prepared for the NEBULA X Hackathon, track PS3 — Predictive Fault Detection. The focus is bogie-temperature anomaly detection for maintenance prioritisation and engineering investigation.

## Problem statement
Bogie temperature behaviour can indicate developing thermal issues before a component reaches a critical failure state. The challenge is to detect abnormal heating, distinguish current and gradually developing problems, and provide interpretable evidence to support investigation without claiming exact mechanical fault identification.

## Target user
The target user is a rail condition-monitoring or reliability engineer who needs early visibility of risky thermal changes, the affected component hierarchy, and the supporting evidence needed to decide where to inspect next.

## Proposed solution
The proposed solution combines exploratory data analysis, feature engineering, anomaly detection, and an evidence-led dashboard. It aims to surface anomalies in bogie temperature patterns, cluster them into operationally meaningful events, and prioritise the components needing engineering attention.

## Main features
- Early detection of abnormal bogie temperature behaviour
- Support for current and developing thermal anomalies
- Ranking of affected train components by urgency
- Evidence-based alert explanations
- Interactive dashboard for engineers and reviewers
- Reusable Python pipeline structure for future development

## Planned ML pipeline
1. Inspect and validate the available CSV inputs.
2. Clean and standardise time-series data.
3. Engineer features such as rolling averages, rolling variability, temperature deltas, rate of change, and component-level baselines.
4. Train and compare initial anomaly detection methods.
5. Produce a ranked list of abnormal events and component severity.
6. Export results for the dashboard and formal submission files.

## Repository structure
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
├── notebooks/
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── features.py
│   ├── model.py
│   ├── export_results.py
│   └── run_pipeline.py
├── predictions/
│   └── README.md
├── docs/
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   └── data/
│       └── dashboard_data.json
├── writeup/
│   └── solution_writeup.md
└──
```

## Installation
From the project root, create a virtual environment if needed and install the dependencies:

```bat
cd nebula-x-bogie-early-warning
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running the pipeline
This repository currently provides an inspection-first pipeline scaffold. The actual data schema will be confirmed after reading the organiser specification.

```bat
cd nebula-x-bogie-early-warning
.venv\Scripts\activate
python src\run_pipeline.py --input data\input --dashboard-output docs\data\dashboard_data.json --predictions-output predictions\sample_predictions.csv
```

The script is intentionally designed to stop safely until the schema is understood and a valid configuration is prepared. This is not a final production pipeline yet.

## Dashboard instructions
The static dashboard is designed to run from the `docs/` folder and is intended for GitHub Pages hosting:

```bat
cd nebula-x-bogie-early-warning
python -m http.server 8000 --directory docs
```
Then open:

```text
http://localhost:8000/
```

## Placeholder prototype link
TBD — add the hosted dashboard URL once available.

## Placeholder video-pitch link
TBD — add the public video link once available.

## Team members
- Dominic Loh
- Ryan Koh Zhixiang
- Muhamed Aydin
- Chloe How Wanyu

## Limitations and disclaimer
- The exact data fields and the official prediction format will be updated after reading `Problem_Statement_3_Specifications.md`.
- An anomaly does not confirm an exact mechanical fault.
- This system supports engineering investigation and does not replace engineering judgement.
- Accuracy should not be claimed unless verified labels are provided.
- The official `*_predictions.csv` format must follow the organisers’ specification exactly.
- This repository is a hackathon scaffold and should be treated as a working prototype until the real dataset schema is confirmed.

---

This project is intentionally structured to be easy for a small team to build quickly during the hackathon while keeping the data pipeline, ML logic, dashboard, and submission expectations clearly separated.
