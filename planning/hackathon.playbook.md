# HACKATHON PLAYBOOK

## Event information

- Event name: NEBULA X Hackathon — “The Living Railway: Future of Mobility”
- Track: PS3 — Predictive Fault Detection
- Event dates: 18–20 September 2026
- Submission deadline: 19 September 2026 at 4:00 pm Singapore time
- Submission counter opens: 2:30 pm on 19 September 2026
- Submission location: EA Atrium, outside LT7A
- Physical sign-in requirement: At least one team member must physically sign in for submission

## Team

- Chloe How Wanyu
- Dominic Loh
- Muhamed Aydin
- Ryan Koh Zhixiang

## Role assignments

| Responsibility | Main owner | Backup | Status |
| --- | --- | --- | --- |
| Data inspection and cleaning | Muhamed Aydin | Dominic Loh
| Features and ML | Muhamed Aydin | Dominic Loh
| Dashboard | Ryan Koh Zhixiang | Dominic Loh | Scaffold ready |
| Integration and exports | Chloe How Wanyu | Scaffold ready |
| README and write-up | All | Template ready |
| Video pitch | All | Template ready |
| Submission checker | All | Checklist ready |
| Physical submission | All | Not assigned |

## Project scope

### Problem statement
Bogie-temperature telemetry can contain early signs of abnormal heating, but the signal is usually mixed with normal operating variation. The challenge is to identify current and developing abnormal behaviour and prioritise components that merit engineering attention without claiming a confirmed root cause.

### Primary user
Rail condition-monitoring or reliability engineer.

### Main user need
“Identify which train component requires attention, understand whether its condition is worsening, and see enough evidence to decide whether it should be investigated.”

### Proposed solution
Our system analyses bogie-temperature telemetry to identify current and developing abnormal heating patterns. It groups abnormal readings into meaningful events, ranks affected train components by severity, and presents the supporting evidence through an interactive dashboard for rail condition-monitoring engineers.

### Value proposition
“Turn large amounts of bogie-temperature telemetry into prioritised, explainable maintenance alerts.”

## Minimum viable product

- [ ] Read and inspect the official data files
- [ ] Clean and validate required fields
- [ ] Engineer suitable temperature features
- [ ] Produce anomaly predictions
- [ ] Identify current and developing abnormal behaviour where supported by data
- [ ] Group consecutive predictions into events
- [ ] Rank events for review
- [ ] Produce the required `*_predictions.csv` files
- [ ] Produce `dashboard_data.json`
- [ ] Display prioritised alerts and supporting evidence
- [ ] Host the dashboard publicly
- [ ] Complete README, write-up and video

## Out of scope

- Door anomaly detection
- Track-condition monitoring
- Exact root-cause diagnosis
- Automated repair recommendations
- Official safety certification
- Mobile application
- Claims unsupported by the supplied data

## Proposed technical flow

```text
Official bogie CSVs
→ inspection and validation
→ cleaning
→ feature engineering
→ baseline and anomaly model
→ row-level predictions
→ event grouping and prioritisation
→ official prediction CSVs
→ dashboard JSON
→ hosted dashboard
```

## Planned technical approach

This section is provisional and will be updated after the official dataset and specification are reviewed.

- Per-component baselines if component identifiers are available
- Current temperature
- Rolling mean
- Rolling standard deviation
- Temperature difference
- Rate of temperature change
- Deviation from baseline
- Paired-component comparison if suitable sensors exist
- Persistence of abnormal behaviour
- Isolation Forest as the initial model
- Simple statistical baseline for comparison
- Another model only if time and data allow

The final feature set and model choice depend on the actual dataset and organiser specification.

## Dashboard minimum content

- Four useful KPI cards
- Prioritised alert list
- Selected-alert temperature trend
- Normal or historical baseline
- Alert start marker
- Explanation panel
- Fleet or component status overview
- Filters for available train, component, severity and time fields

The dashboard should use progressive disclosure and avoid unnecessary charts or decorative elements.

## Definition of done

- [ ] Official CSV format confirmed
- [ ] Data inspection completed
- [ ] Baseline model completed
- [ ] Main model completed
- [ ] Official predictions exported
- [ ] Prediction files validated
- [ ] Dashboard JSON exported
- [ ] Dashboard tested locally
- [ ] Dashboard tested through GitHub Pages
- [ ] README completed
- [ ] Write-up completed
- [ ] Pitch recorded
- [ ] Submission links tested
- [ ] Backup created
- [ ] Physical submission completed

## Timeline

### Friday

- Read specification
- Inspect data
- Confirm scope and roles
- Build the first end-to-end version
- Complete first integration before the team rests

### Saturday

- Finalise features and model
- Generate predictions
- Integrate dashboard
- Complete documentation
- Record pitch
- Freeze development by 2:15 pm
- Submit after the counter opens at 2:30 pm and before 4:00 pm

### Sunday, if selected

- Prepare live demo
- Review likely technical questions
- Ensure every member understands the full solution

Do not invent final presentation duration.

## Decision log

| Decision | Status | Notes |
| --- | --- | --- |
| Bogie temperature selected as the main focus | Confirmed | Primary objective remains anomaly detection and prioritisation |
| Dashboard will be a web application | Confirmed | Source is in `docs/` and is intended for GitHub Pages |
| Python will perform processing and ML | Confirmed | The repository uses a Python-based workflow |
| JSON will connect Python outputs to the dashboard | Confirmed | Dashboard data is stored under `docs/data/dashboard_data.json` |
| GitHub Pages will host the dashboard | Confirmed | Public hosted dashboard is expected to use the repository Pages configuration |
| Exact root-cause recommendations remain out of scope | Confirmed | We do not claim exact mechanical fault diagnosis |
| [Additional decision] | [Status] | [Notes] |

## Team check-in

Use the following five questions in every short stand-up.

- What did you finish?
- What are you doing next?
- Are you blocked?
- Do you need anything from another teammate?
- Does the end-to-end pipeline still work?

---

This playbook should be updated throughout the event as the team confirms the official specification, the data structure, and the final delivery plan.
