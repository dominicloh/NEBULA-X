# DATA REVIEW TEMPLATE

This worksheet should be completed in the first hour or two after the official data is released. The goal is to confirm what the organiser expects before any final model or submission work is considered reliable.

## Specification review

- [ ] Read `Problem_Statement_3_Specifications.md`
- [ ] Identify required prediction filenames
- [ ] Identify required prediction columns
- [ ] Identify column order
- [ ] Identify timestamp format
- [ ] Identify accepted prediction values
- [ ] Identify whether every input row needs a prediction
- [ ] Identify whether files must remain separate
- [ ] Identify evaluation criteria
- [ ] Record unanswered questions for organisers

## Dataset inventory

| File | Purpose | Rows | Columns | Size | Notes |
| --- | --- | --- | --- | --- | --- |
| [FILE NAME] | [PURPOSE] | [ROW COUNT] | [COLUMN COUNT] | [SIZE] | [NOTES] |

## Data dictionary

| Column | Meaning | Data type | Unit | Missing values | Used by model? | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| [COLUMN NAME] | [MEANING] | [TYPE] | [UNIT] | [COUNT/STATUS] | [YES/NO] | [NOTES] |

## Identifier mapping

Only record what is actually available in the data. Do not assume all fields exist.

- Timestamp: [TO FILL AFTER DATA INSPECTION]
- Train: [TO FILL AFTER DATA INSPECTION]
- Bogie: [TO FILL AFTER DATA INSPECTION]
- Axle/bearing/component: [TO FILL AFTER DATA INSPECTION]
- Sensor: [TO FILL AFTER DATA INSPECTION]
- Temperature: [TO FILL AFTER DATA INSPECTION]
- Speed: [TO FILL AFTER DATA INSPECTION]
- Operating state: [TO FILL AFTER DATA INSPECTION]
- Ambient/environmental value: [TO FILL AFTER DATA INSPECTION]
- Fault label: [TO FILL AFTER DATA INSPECTION]
- Confirmed fault time: [TO FILL AFTER DATA INSPECTION]
- Maintenance record: [TO FILL AFTER DATA INSPECTION]

## Time-series structure

- Dataset start and end: [TO FILL AFTER DATA INSPECTION]
- Sampling frequency: [TO FILL AFTER DATA INSPECTION]
- Regular sampling: [YES/NO/UNKNOWN]
- Number of trains: [TO FILL AFTER DATA INSPECTION]
- Number of bogies/components: [TO FILL AFTER DATA INSPECTION]
- Number of sensors: [TO FILL AFTER DATA INSPECTION]
- Missing time periods: [TO FILL AFTER DATA INSPECTION]
- Duplicate timestamps: [TO FILL AFTER DATA INSPECTION]
- Time zone: [TO FILL AFTER DATA INSPECTION]

## Data-quality checks

- [ ] Missing values
- [ ] Duplicate rows
- [ ] Invalid timestamps
- [ ] Non-numeric sensor readings
- [ ] Constant columns
- [ ] Impossible values
- [ ] Large gaps
- [ ] Irregular sampling
- [ ] Sudden isolated spikes
- [ ] Unequal sensor coverage
- [ ] Leakage from future information
- [ ] Class imbalance

## Initial plots

- [ ] Temperature against time
- [ ] Temperature distribution
- [ ] Temperature by component
- [ ] Comparison between similar components
- [ ] Rolling mean and variability
- [ ] Rate of change
- [ ] Data around verified fault events, if available

## Initial findings

### Normal behaviour
- [TO FILL AFTER DATA INSPECTION]

### Suspected anomalous behaviour
- [TO FILL AFTER DATA INSPECTION]

### Operating-context effects
- [TO FILL AFTER DATA INSPECTION]

### Possible useful features
- [TO FILL AFTER DATA INSPECTION]

### Potential data leakage
- [TO FILL AFTER DATA INSPECTION]

### Open questions
- [TO FILL AFTER DATA INSPECTION]

### Initial modelling decision
- [TO FILL AFTER DATA INSPECTION]

## Official output requirements

| Item | Value |
| --- | --- |
| Output directory | `predictions/` |
| File naming rule | [TO CONFIRM FROM SPECIFICATION] |
| Required columns | [TO CONFIRM FROM SPECIFICATION] |
| Column order | [TO CONFIRM FROM SPECIFICATION] |
| Row count expectation | [TO CONFIRM FROM SPECIFICATION] |
| Timestamp format | [TO CONFIRM FROM SPECIFICATION] |
| Prediction format | [TO CONFIRM FROM SPECIFICATION] |
| Missing-value policy | [TO CONFIRM FROM SPECIFICATION] |
| Validation completed by | [OWNER TO BE ASSIGNED] |
| Final confirmation status | [PENDING / CONFIRMED] |

---

Keep this template short and operational. It should be used as a working document during the first inspection session, then updated as the team confirms the final output contract.
