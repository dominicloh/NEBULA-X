# HACKATHON PLAYBOOK

## Event information

- Event name: NEBULA X Hackathon — “The Living Railway: Future of Mobility”
- Track: PS3 — Predictive Fault Detection
- Event dates: As advised by the organiser
- Submission deadline: As advised by the organiser
- Submission counter opening: As advised by the organiser
- Submission location: As advised by the organiser

## Team

- Dominic Loh
- Ryan Koh Zhixiang
- Muhamed Aydin
- Chloe How Wanyu

## Role assignments

| In charge of | Responsibility | Main owner | Backup
| --- | --- | --- | --- |
| Door data, segmentation and model | Door subsystem owner | Muhamed Aydin | Dominic Loh
| Rail data, features and model | Rail subsystem owner | Dominic Loh | Muhamed Aydin
| Shared app and visualisation | App owner | Ryan Koh Zhixiang | Dominic Loh
| Integration, validation, write-up and submission | Integration / team lead | Chloe How Wanyu | Ryan Koh Zhixiang
| README and documentation | All | In progress |
| Video pitch | All |

## Decision log

The team selected Rail Corrugation and Door as its two official PS3 subsystems after the official specifications were released.

## Project scope

### Problem statement
The team supports rail engineers who need to review subsystem sensor data quickly and understand whether a recording requires attention. The challenge is to interpret time-series signals from two different task types while keeping the evidence accessible and the outputs consistent.

### Primary user
Rail condition-monitoring or reliability engineer.

### Main user need
“Help rail engineers process subsystem sensor data quickly, identify which recordings require attention, understand the evidence behind each result and download consistent outputs for further investigation.”

### Proposed solution
Our system provides one accessible condition-monitoring app for two rail subsystems. It analyses door motor-cycle signals to detect and classify normal or abnormal-resistance operating cycles, and analyses axle-box vibration and shock files to classify rail condition as Normal, Side I or Side II corrugation.

### Value proposition
“Turn complex rail sensor time series into clear, explainable and downloadable condition-monitoring results.”

## Minimum viable product

- [ ] Read the official Door and Rail Corrugation specifications
- [ ] Confirm the Door stream schema and required fields
- [ ] Confirm the Rail file schema and labels
- [ ] Keep the Door and Rail pipelines separate
- [ ] Produce a shared app with a subsystem selector
- [ ] Implement safe fail states when the model is not ready
- [ ] Validate each official prediction CSV schema
- [ ] Package the outputs into `predictions.zip`
- [ ] Produce the required `door_predictions.csv` and `rail_predictions.csv`
- [ ] Provide evidence views for each subsystem
- [ ] Complete README, write-up and video

## Out of scope

- Fabricating model results or timestamps
- Claiming an exact mechanical root cause from signal data alone
- Unsupported ACV or SHM labels unless clearly marked as not attempted
- Unverified final model claims before the relevant Info Kit is read
- Raw dataset leakage in the package or app

## Proposed technical flow

```text
Door Info Kit / Rail Info Kit
→ schema confirmation
→ cleaning and validation
→ subsystem-specific feature engineering
→ subsystem-specific model training or safe placeholder state
→ official prediction export
→ app presentation and evidence review
→ combined package validation and ZIP submission bundle
```

## Dashboard and app minimum content

- Shared app title and explanation
- Subsystem selector for Door or Rail Corrugation
- Upload area and analysis trigger
- Clear validation errors and safe non-fake message states
- Prediction results and evidence for each subsystem
- Download button for the correct output file
- Methodology and disclaimer

## Definition of done

- [ ] Door Info Kit reviewed
- [ ] Rail Corrugation Info Kit reviewed
- [ ] Door stream schema confirmed
- [ ] Rail file schema confirmed
- [ ] Door segmentation works or fails safely with a clear message
- [ ] Door classifier works or fails safely with a clear message
- [ ] Rail classifier works or fails safely with a clear message
- [ ] IoU-weighted F1 planned for Door evaluation
- [ ] Macro F1 planned for Rail evaluation
- [ ] `door_predictions.csv` validates correctly
- [ ] `rail_predictions.csv` validates correctly
- [ ] `predictions.zip` contains both CSVs at its root
- [ ] Shared app works without code interaction
- [ ] README, write-up and video updated
- [ ] Submission links checked

## Timeline

### Phase 1

- Read the Info Kits and confirm schemas
- Inspect the data and note missing or uncertain fields
- Finalise the subsystem-specific module boundaries

### Phase 2

- Implement subsystem data validation and feature interfaces
- Build the shared app workflow
- Review the safe fail states and packaging checks

### Phase 3

- Train or validate the final model for each subsystem
- Export the official predictions
- Package and test the ZIP submission bundle
- Finalise the documentation and pitch

## Team check-in

Use the following five questions in every short stand-up.

- What did you finish?
- What are you doing next?
- Are you blocked?
- Do you need anything from another teammate?
- Does the end-to-end pipeline still work for both subsystems?

---

This playbook should be updated as the team confirms the final official specification, data structure and delivery plan.
