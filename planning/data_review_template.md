# DATA REVIEW TEMPLATE

This worksheet should be completed after the Door and Rail Corrugation specifications are reviewed. The goal is to confirm what the organiser expects before final model or submission work is treated as reliable.

## Specification review

- [ ] Read the Door Info Kit and the Rail Corrugation Info Kit
- [ ] Identify required prediction filenames for each subsystem
- [ ] Identify required prediction columns and order
- [ ] Identify timestamp format and accepted prediction values
- [ ] Identify whether files remain separate
- [ ] Identify evaluation criteria for each subsystem
- [ ] Record unanswered questions for organisers

## Door continuous-stream inspection

### Stream and timing
- Stream name: [TO CONFIRM FROM DOOR INFO KIT]
- Timestamp format: [TO CONFIRM FROM DOOR INFO KIT]
- Stream duration: [TO FILL AFTER DATA INSPECTION]
- Sampling interval: [TO FILL AFTER DATA INSPECTION]
- Missing periods: [TO FILL AFTER DATA INSPECTION]
- Duplicate timestamps: [TO FILL AFTER DATA INSPECTION]

### Signal fields
- Signal columns: [TO CONFIRM FROM DOOR INFO KIT]
- Motor current: [TO FILL AFTER DATA INSPECTION]
- Voltage: [TO FILL AFTER DATA INSPECTION]
- Back-EMF: [TO FILL AFTER DATA INSPECTION]
- Door position: [TO FILL AFTER DATA INSPECTION]
- Missing values: [TO FILL AFTER DATA INSPECTION]

### Cycle and label review
- Position states: [TO CONFIRM FROM DOOR INFO KIT]
- Reference segment labels: [TO CONFIRM FROM DOOR INFO KIT]
- Normal vs abnormal-resistance distribution: [TO FILL AFTER DATA INSPECTION]
- Cycle durations: [TO FILL AFTER DATA INSPECTION]
- Potential leakage between related cycles: [TO FILL AFTER DATA INSPECTION]

## Rail file-level inspection

### File inventory
| File | Purpose | Rows | Columns | Notes |
| --- | --- | --- | --- | --- |
| [FILE NAME] | [PURPOSE] | [ROW COUNT] | [COLUMN COUNT] | [NOTES] |

### File content review
- File-level labels available: [YES/NO/UNKNOWN]
- Signal columns: [TO CONFIRM FROM RAIL INFO KIT]
- Sampling characteristics: [TO FILL AFTER DATA INSPECTION]
- Missing values: [TO FILL AFTER DATA INSPECTION]
- Class distribution: [TO FILL AFTER DATA INSPECTION]
- Leakage-safe split plan: [TO FILL AFTER DATA INSPECTION]

## Data-quality checks

- [ ] Missing values
- [ ] Duplicate timestamps or duplicate rows
- [ ] Invalid timestamps
- [ ] Non-numeric signal readings
- [ ] Large gaps or irregular sampling
- [ ] Impossible position or motion states
- [ ] Leakage from future information
- [ ] Class imbalance

## Official output requirements

| Item | Value |
| --- | --- |
| Door output filename | `door_predictions.csv` |
| Door required columns | `start_time,end_time,prediction` |
| Door label set | `Normal`, `Abnormal resistance` |
| Rail output filename | `rail_predictions.csv` |
| Rail required columns | `file_id,prediction` |
| Rail label set | `Normal`, `Side I`, `Side II` |
| ZIP root contents | `door_predictions.csv`, `rail_predictions.csv` |
| Timestamp format | [TO CONFIRM FROM DOOR INFO KIT] |
| Final confirmation status | [PENDING / CONFIRMED] |

---

Keep this template short and operational. It should be updated as the team confirms the final output contract for both subsystems.
