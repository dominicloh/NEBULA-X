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

## Rail experiment summary table

| ID | Date/time | Model | Features | Key parameters | Validation method | Result | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [RAIL EXP ID] | [DATE/TIME] | [MODEL] | [FEATURES] | [PARAMETERS] | [METHOD] | [RESULT] | [KEEP/MODIFY/REJECT] |

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

### Experiment 0 — File-level baseline

A simple file-level feature baseline for Normal vs Side I/II classification.

### Experiment 1 — Vibration and shock feature model

Use the agreed feature set from the Rail Info Kit.

### Experiment 2 — Class-weighted or balanced classifier

Use when class imbalance is present.

### Experiment 3 — Macro F1 evaluation

Report the official competition metric for the Rail task.

## Final-model decision

| Subsystem | Candidate direction | Selected? | Notes |
| --- | --- | --- | --- |
| Door | Simple segmentation baseline | [YES/NO] | Not preselected until Info Kit review |
| Door | Rule-based candidate segmentation | [YES/NO] | [TO FILL] |
| Door | Segment-feature binary classifier | [YES/NO] | [TO FILL] |
| Rail | File-level feature baseline | [YES/NO] | [TO FILL] |
| Rail | Multi-class classifier | [YES/NO] | [TO FILL] |

## Reproducibility

- Random seed: [TO FILL AFTER MODEL EVALUATION]
- Python version: [TO FILL AFTER MODEL EVALUATION]
- Library versions: [TO FILL AFTER MODEL EVALUATION]
- Final feature list: [TO FILL AFTER MODEL EVALUATION]
- Final parameters: [TO FILL AFTER MODEL EVALUATION]
- Training-data description: [TO FILL AFTER DATA INSPECTION]
- Final model location: [TO FILL AFTER MODEL EVALUATION]
- Exact run command: [TO FILL AFTER MODEL EVALUATION]

---

Keep this log simple, factual, and easy to update during the project timeline. The main goal is understanding what changed, why it changed, and which evaluation rule governs the final result.
