# Team Workflow

## Project scope
The team selected Rail Corrugation and Door as its two official PS3 subsystems after the official specifications were released.

## Suggested ownership

| Responsibility | Suggested ownership |
| --- | --- |
| Door data, segmentation and model | Door subsystem owner |
| Rail data, features and model | Rail subsystem owner |
| Shared Streamlit app and visualisation | App owner |
| Integration, validation, write-up, video and submission | Integration / team lead |

Keep actual names as placeholders until the team confirms assignments.

The two subsystem owners are responsible for both their data understanding and initial model. The app owner must not duplicate model logic. The integration lead ensures both pipelines use consistent interfaces.

## Working rules
- One main owner for each subsystem or shared component.
- Keep the Door and Rail pipelines separate at the module level.
- Share only the final validated interfaces with the app owner.
- Pull before starting work.
- Commit small working changes.
- Do not silently change CSV schema names or timestamp conventions.
- Inform the relevant subsystem owner and integration lead before schema changes.
- Keep the latest working version available.
- Report blockers early.
- Hold a five-minute check-in approximately every two hours.
- Stop adding new features before final submission checking.

## Check-in questions
- What did you finish?
- What are you doing next?
- Are you blocked?
- Do you need anything from another teammate?
- Does the end-to-end pipeline still work for both subsystems?

## Milestones
- Friday night: Door and Rail scaffolds are live and validated for safe failure states
- Saturday morning: Info Kit checks complete and model interfaces confirmed
- Saturday late morning: Door segmentation and Rail feature pipeline aligned with the app
- Saturday afternoon: final validation and packaging for `predictions.zip`
- Submission deadline: as advised by the organiser
- At least one member must physically sign in at the event venue for submission if required

## Communication expectations
Keep the handoff between Door data, Rail data, app development and submission work explicit. If a schema or output format changes, record it in writing and inform the relevant owners before continuing. This prevents the shared app or prediction exports drifting away from the subsystem pipelines.
