# MODEL EXPERIMENT LOG

The purpose of this log is to record what was changed, why it was changed, and whether it improved the team’s ability to detect abnormal bogie-temperature behaviour. Do not claim final model quality without verified labels or a documented evaluation rule.

## Evaluation plan

### If verified labels are available

Possible evaluation measures include:

- Precision
- Recall
- F1 score
- Confusion matrix
- False alerts
- Verified fault events detected
- Event-level recall
- Early-warning lead time

Accuracy alone may be misleading because anomalies are usually rare. The exact measure should follow the official specification and the supplied labels.

### If verified labels are unavailable

Use the following practical checks instead:

- Manual inspection of detected events
- Deviation from historical or component baseline
- Persistence
- Rate of change
- Agreement between signals
- Comparison with a statistical baseline
- Transparent limitation statement

## Experiment summary table

| ID | Date/time | Model | Features | Key parameters | Validation method | Result | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [EXP ID] | [DATE/TIME] | [MODEL] | [FEATURES] | [PARAMETERS] | [METHOD] | [RESULT] | [KEEP/MODIFY/REJECT] |

## Detailed experiment template

### Experiment ID
[EXP ID]

### Objective
[TO FILL AFTER MODEL EVALUATION]

### Data used
[TO FILL AFTER DATA INSPECTION]

### Preprocessing
[TO FILL AFTER DATA INSPECTION]

### Features
- [FEATURE 1]
- [FEATURE 2]
- [FEATURE 3]

### Model
[MODEL NAME]

### Parameters
- [PARAMETER NAME]: [VALUE OR UNKNOWN]

### Training method
[TO FILL AFTER MODEL EVALUATION]

### Validation method
[TO FILL AFTER MODEL EVALUATION]

### Results
- [RESULT SUMMARY]

### Example events
- [EVENT DESCRIPTION]

### False positives
- [TO FILL AFTER MODEL EVALUATION]

### Problems
- [TO FILL AFTER MODEL EVALUATION]

### Interpretation
[TO FILL AFTER MODEL EVALUATION]

### Keep, modify or reject
[KEEP / MODIFY / REJECT]

### Next action
[TO FILL AFTER MODEL EVALUATION]

## Planned experiments

### Experiment 0 — Statistical baseline

Use a simple statistical baseline to compare against the anomaly-detection model. Possible signals include:

- Component rolling baseline
- Deviation from rolling mean
- Rate of change
- Persistence of abnormal readings

Parameters and results: [TO FILL AFTER MODEL EVALUATION]

### Experiment 1 — Isolation Forest with temperature only

This is a simple reference experiment, not necessarily the final model. It helps the team understand whether the raw temperature signal alone is enough for a useful early warning.

Parameters and results: [TO FILL AFTER MODEL EVALUATION]

### Experiment 2 — Isolation Forest with rolling behaviour

Possible features include:

- Temperature
- Rolling mean
- Rolling standard deviation
- Temperature change
- Rate of change
- Deviation from baseline

This should be tested before deeper modelling work because the data likely contains short-term thermal patterns that matter for early warning.

Parameters and results: [TO FILL AFTER MODEL EVALUATION]

### Experiment 3 — Contextual or component-comparison model

Only if identifiers and contextual fields are available.

Possible directions:

- Compare a component against its own recent history
- Compare similar components within the same train or operating context
- Combine component baseline and persistence behaviour

Parameters and results: [TO FILL AFTER MODEL EVALUATION]

### Optional model comparison

Only if time allows:

- Local Outlier Factor
- One-Class SVM

These experiments are optional. They should not be treated as required for the hackathon MVP unless the data and time support them.

## Final-model decision

| Criterion | Statistical baseline | Isolation Forest | Optional comparison |
| --- | --- | --- | --- |
| Detection performance | [UNKNOWN] | [UNKNOWN] | [UNKNOWN] |
| False alerts | [UNKNOWN] | [UNKNOWN] | [UNKNOWN] |
| Explainability | [UNKNOWN] | [UNKNOWN] | [UNKNOWN] |
| Runtime | [UNKNOWN] | [UNKNOWN] | [UNKNOWN] |
| Robustness | [UNKNOWN] | [UNKNOWN] | [UNKNOWN] |
| Selected? | [YES/NO] | [YES/NO] | [YES/NO] |

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

Keep this log simple, factual, and easy to update during the short hackathon timeline. The main goal is understanding what changed and why, not writing a long narrative.
