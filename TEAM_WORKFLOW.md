# Team Workflow

## Project scope
Focus only on bogie-temperature anomaly detection for the minimum viable product. Do not broaden the scope into unrelated maintenance or signalling tasks during the hackathon.

## Suggested ownership
- Data lead: data inspection, cleaning and documentation
- ML lead: feature engineering, anomaly detection and prediction output
- Dashboard lead: interface, charts and interactions
- Integration/pitch lead: JSON integration, README, write-up, video and submission checks

Do not assign individual names to the roles yet. The team should agree on ownership together before the work becomes parallelised.

## Working rules
- One main owner for each file or area.
- Pull before starting work.
- Commit small working changes.
- Do not silently change JSON field names.
- Inform both ML and dashboard owners about schema changes.
- Keep the latest working version available.
- Report blockers early.
- Hold a five-minute check-in approximately every two hours.
- Stop adding new features before final submission checking.

## Check-in questions
- What did you finish?
- What are you doing next?
- Are you blocked?
- Do you need anything from another teammate?
- Does the end-to-end pipeline still work?

## Milestones
- Friday night: basic real-data pipeline and placeholder dashboard connected
- Saturday morning: model and features frozen
- Saturday late morning: final dashboard integration
- Saturday 2:15 pm: stop development and perform submission audit
- Submission deadline: 19 September, 4:00 pm
- Submission counter opens at 2:30 pm
- At least one member must physically sign in at the EA Atrium outside LT7A

## Communication expectations
Keep the handoff between data, ML, dashboard and submission work explicit. If a schema or output format changes, note it in writing and tell the relevant owners before continuing. This prevents dashboard or prediction outputs drifting away from the model-generated data.
