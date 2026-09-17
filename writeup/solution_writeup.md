# NEBULA X — Train Condition Monitoring

## 1. Shared problem and user

The project addresses a shared requirement for rail condition-monitoring engineers: to process subsystem sensor data quickly, identify recordings that require attention, understand the evidence behind the decision, and produce consistent outputs for further investigation. The primary user is the rail condition-monitoring or reliability engineer.

## 2. Door subsystem

### Data
The Door task uses a single continuous test stream named `Test.csv`, and the pipeline must segment and classify operating cycles over time. The exact schema and timestamp rule remain pending confirmation from the Door Info Kit.

### Segmentation
The Door pipeline is designed to segment a continuous stream into predicted open/close cycles before classifying each segment. The interface is in place, but the real cycle detection method must be confirmed once the Info Kit is read.

### Features
Potential Door features include cycle duration, current statistics, voltage summaries, back-EMF summaries, position progression and movement-phase timing. These are candidate features only until the Info Kit confirms the signal set and cycle rules.

### Model
The Door model is a binary classifier for `Normal` vs `Abnormal resistance`. The final evaluation must use IoU-weighted F1 across temporal segments, so segment timing and class label both matter.

### IoU-weighted F1
This is the official scoring rule for the Door task. A door submission is only complete when both the predicted boundaries and predicted label match the task specification and validation rules.

### Results
No final Door results are claimed here. The project intentionally leaves this section pending until the Info Kit and trained pipeline are available.

## 3. Rail Corrugation subsystem

### Data
The Rail Corrugation task uses file-level multi-channel axle-box vibration and shock data. The final file schema, labels and feature set remain pending confirmation from the Rail Corrugation Info Kit.

### Features
Rail feature engineering will be built around the validated vibration and shock inputs for each file. The exact feature set remains to be confirmed after the data and Info Kit are reviewed.

### Model
The Rail model is a file-level multi-class classifier for `Normal`, `Side I` and `Side II`. It uses a separate pipeline from the Door subsystem and is evaluated on the official Macro F1 metric.

### Macro F1
This is the official Rail Corrugation scoring rule. It measures performance across the multi-class labels rather than a binary decision threshold.

### Results
No final Rail Corrugation results are claimed here. The project intentionally leaves this section pending until the relevant Info Kit and trained model are available.

## 4. Shared app

The app is a single Streamlit workflow that provides a subsystem selector. The user can switch between Door and Rail Corrugation, upload the relevant data, inspect the output and download the correct prediction CSV file. The app is designed to keep the pipelines separate while sharing one interface.

## 5. Explainability

Each subsystem is designed to present evidence supporting the prediction, such as signal plots, segment timings, relevant channels and file-level metadata. The app does not claim a precise mechanical root cause from the sensor data alone.

## 6. Submission outputs

The expected package contains both attempted subsystem outputs in a single ZIP archive at the root:

```text
predictions.zip
├── door_predictions.csv
└── rail_predictions.csv
```

The archives contain no raw data and no extra folders. Each CSV must follow the official subsystem schema exactly.

## 7. Limitations

The project intentionally avoids unsupported claims. Door segmentation and classification depend on the Info Kit and trained pipeline; Rail Corrugation also depends on its own confirmed schema and model; and any final submission must be validated against the official competition rules rather than assumptions.

## 8. Team contributions

The Door subsystem owner is responsible for the Door data understanding and initial model, the Rail owner is responsible for the Rail data understanding and initial model, the app owner manages the shared visual interface, and the integration lead ensures both pipelines use consistent interfaces and validation.
