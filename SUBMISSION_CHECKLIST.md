# Submission Checklist

## Final submission review
- [ ] Public GitHub repository
- [ ] Complete README and project documentation
- [ ] Shared Streamlit app works without code interaction
- [ ] App tested for both Door and Rail Corrugation workflows
- [ ] Video pitch between 2 and 3 minutes
- [ ] Video accessible without requesting permission
- [ ] Official `door_predictions.csv` file present
- [ ] Official `rail_predictions.csv` file present
- [ ] Both outputs inside `predictions.zip` at the archive root
- [ ] Correct column names and order for both CSVs
- [ ] Correct timestamp and prediction formats
- [ ] Correct number of prediction rows
- [ ] No accidental pandas index column
- [ ] No missing required predictions
- [ ] Short solution write-up
- [ ] Explanation of solution uniqueness
- [ ] Technology stack
- [ ] No API keys or secrets
- [ ] Backup copy of links and prediction files
- [ ] At least one member assigned to physical submission
- [ ] Final submission before the organiser deadline

> Warning: The prediction files must follow the official specification exactly for each subsystem. Do not guess the required schema or field names. Validate both file types before submission.

## Quick pre-flight checks
- [ ] The two CSV files are inside `predictions.zip` directly at its root.
- [ ] `door_predictions.csv` has columns `start_time,end_time,prediction`.
- [ ] `rail_predictions.csv` has columns `file_id,prediction`.
- [ ] Door labels are only `Normal` and `Abnormal resistance`.
- [ ] Rail labels are only `Normal`, `Side I` and `Side II`.
- [ ] Timestamp format is compliant with the Door Info Kit once confirmed.
- [ ] No raw data is included in the package.
- [ ] The app and package both reflect both attempted subsystems.

## Final reminder
Before final submission, stop adding features, verify both output files, validate the ZIP archive, test the hosted app in a clean browser profile, and ensure the package is ready for official evaluation.
