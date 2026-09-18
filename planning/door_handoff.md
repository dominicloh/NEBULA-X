# Door Handoff Guide

Written for a teammate who is new to both machine learning and this
repository. Read this whole document before writing any code. Nothing here
is invented — every "confirmed" fact below was checked directly against the
official files in `organiser-materials/`, not assumed.

## 1. The Door task, in plain English

Metro train doors sometimes have "abnormal resistance" while opening or
closing — something is physically catching (a jammed rail, a deformed door
leaf, etc.) instead of the door moving smoothly. We're given sensor readings
from the door controller (motor current, voltage, back-EMF, door position,
and several on/off switches) recorded continuously while the train runs.

**You are given one long, continuous recording — not one file per door
cycle.** Picture a security camera that never stops recording, rather than a
folder of pre-cut video clips. Somewhere in that long recording, a door
opens and closes over and over, with quiet gaps in between (someone parked,
no one used the door, etc.). Your job has two parts:

1. **Find each door-open/close cycle** in the continuous recording — where
   does it start, where does it end? (This is called "segmentation".)
2. **Classify each cycle you found** as `Normal` or `Abnormal resistance`.

You get graded on *both* parts together — finding the right time window
**and** giving it the right label. Getting the label right on a badly-timed
guess still scores badly. See Section 5.

## 2. Confirmed dataset facts

Everything below was checked against the actual files in
`organiser-materials/PS3/02_Datasets/Door/` and
`organiser-materials/PS3/03_References/Door/`.

### Files

| File | What it is | Confirmed shape |
|---|---|---|
| `Train.csv` | One continuous labelled recording | **18,036 rows x 17 columns** |
| `Train_Segments_Answer.csv` | Ground truth for `Train.csv`: exactly where each real cycle starts/ends and its label | **110 rows x 6 columns** |
| `Test.csv` | Another continuous recording, same format, **unlabelled** — this is what you must process end-to-end | **6,253 rows x 17 columns** |

`Train.csv` and `Test.csv` have **identical columns and dtypes** (checked directly).

### Train.csv / Test.csv columns (confirmed from the actual header row)

```
Datetime, Motor current(mA), Motor Voltage(10mV), Motor electrodynamic force,
Door opening time(.1s), Door closing time(.1s), Close command, Open command,
DCSR, DCSL, DLSR, DLSL, Door Opened, Door Locked, Door is opening,
Door is closing, Door leaf position
```

- `Datetime` is a string (e.g. `2023-7-5-0-0-3-760`); every other column is an integer.
- **No missing values, no duplicate rows, no duplicate `Datetime` values** in either file. `Datetime` is strictly increasing in both files (verified, not assumed).
- `DCSR`/`DCSL`/`DLSR`/`DLSL` are door switch signals (Close/Locked, Right/Left) — see `Door Data Headers.md` for what each one physically means; they toggle between "released" and "actuated" during a cycle.
- **`Door Locked` is constant at `0` in BOTH files** (checked: exactly 1 unique value in each). It currently carries no information — don't be surprised if it doesn't help your model.
- Full plain-English description of every column is in `organiser-materials/PS3/03_References/Door/Door Data Headers.md`.

### Datetime format (confirmed)

`Year-Month-Date-Hour-Minute-Second-Millisecond`, hyphen-separated, **not
zero-padded** — e.g. `2023-7-5-0-0-3-760` is July 5th, 2023, 00:00:03.760.
`pandas.to_datetime` will NOT parse this correctly on its own (it doesn't
recognise the field order or the unpadded numbers) — use
`src/door/preprocess.py::parse_door_datetime` (already implemented for you,
see Section 7) rather than writing your own parser.

### Sampling interval — important nuance (observed, NOT officially documented)

Row-to-row `Datetime` gaps are almost always exactly **20 milliseconds**
(so roughly 50 Hz) — but not everywhere. We found:

- `Train.csv` has **109 gaps larger than 100ms** between consecutive rows.
- `Train_Segments_Answer.csv` has **110 labelled segments**.
- 109 = 110 - 1. This is not a coincidence: the large gaps in `Train.csv`
  line up almost exactly with the boundaries **between** consecutive
  labelled cycles.
- `Test.csv` shows the same pattern: **37 gaps larger than 100ms**, implying
  roughly 38 candidate cycles are sitting in that file (110 : 18,036 rows is
  roughly the same ratio as 38 : 6,253 rows).

**In plain terms: the file doesn't record data at a steady 20ms rate for the
whole 70-minute period. It appears to record densely (~20ms steps) only
during/near each door cycle, then jump forward in time to the next one.**
This means a big jump in `Datetime` between two consecutive rows is a very
strong, almost free, hint that a cycle boundary is nearby — likely a much
better starting signal than trying to read the "opening"/"closing"
flag columns directly (which is exactly what the Info Kit hints at, without
telling you the answer — see Section 4).

**This is a real pattern we measured in the files, not a guarantee.** Don't
build a pipeline that would silently break if a future data file *did*
sample at a constant rate throughout — check the gap pattern yourself with
`inspect_data.py` (Section 9) before relying on it, and treat "big gap = new
cycle nearby" as a strong *hint* to refine, not the final segmentation rule.

### Train_Segments_Answer.csv (confirmed)

Columns: `segment_id, start_time, end_time, operation, status, n_rows`

| Fact | Value |
|---|---|
| Total labelled segments | 110 |
| `status` = `Normal` | 80 (72.7%) |
| `status` = `Abnormal resistance` | 30 (27.3%) |
| `operation` = `Open` / `Close` | 55 / 55 (exactly balanced) -- **informational only, you do not predict this** |
| Segment duration (end - start) | min 2.72s, mean 3.26s, max 3.78s |
| `n_rows` per segment | min 137, mean 164, max 190 |
| Overlapping consecutive segments | **0** (checked) |
| Any segment with start >= end | **0** (checked) |
| Duplicate `segment_id` values | **0** (checked) |
| All segment boundaries inside `Train.csv`'s own time range | **Yes** (checked) |
| Gap between consecutive segments | min ~10.2s, mean ~35.4s, max ~58.8s ("irregular gaps", as the Info Kit says) |

This is a **moderately imbalanced 2-class problem** (about 73%/27%), much
less extreme than the Rail Corrugation subsystem's 86%/8.8%/5.1% split, but
still imbalanced enough that you should not rely on plain accuracy (see
Section 6).

## 3. Exact input and output schemas

### Input (what you're given)

Same 17 columns for both `Train.csv` and `Test.csv` — see Section 2.

### Official output: `door_predictions.csv`

Confirmed from the Info Kit and `organiser-materials/PS3/04_Example_Submission/door_predictions.csv`:

| Column | Value |
|---|---|
| `start_time` | Your predicted segment's start timestamp. Either the native `Year-Month-Date-Hour-Minute-Second-Millisecond` format or a standard ISO timestamp are both accepted. |
| `end_time` | Same format rules as `start_time`. |
| `prediction` | Exactly `Normal` or `Abnormal resistance` (this exact spelling/capitalisation). |

- **One row per predicted segment** — not one row per file (there's only one file: `Test.csv`).
- **No `file_id` column.** No required `confidence` column (an extra one is fine, just unused for scoring).
- No index column when you save the CSV (`to_csv(..., index=False)`).

## 4. The two stages, explained

### Stage 1 — Segmentation: find where each cycle starts and ends

Input: the whole continuous stream (`Train.csv` or `Test.csv`).
Output: a table of candidate cycles, each with a start time and an end time.

The Info Kit deliberately does not tell you which column to use:

> "don't assume any particular column (e.g. the opening/closing flags) is
> necessarily the easiest or most robust signal for detecting where one
> cycle ends and the next begins — think about what actually changes at a
> cycle boundary versus within a cycle."

See Section 2's "Sampling interval" note above for one strong, *measured*
hint (large `Datetime` gaps). You will likely still need to refine exact
start/end times within a candidate window using the signal columns
themselves (e.g. `Door is opening`/`Door is closing`, or `Door leaf
position` settling to a stable value) to get tight, accurate boundaries —
finding the right "island" of activity is not the same as finding the exact
right edge of it.

**Never use `Test.csv` to tune this.** Only `Train.csv` +
`Train_Segments_Answer.csv` tell you the true boundaries. Hold out part of
`Train.csv` yourself to check your segmentation before ever touching `Test.csv`.

### Stage 2 — Classification: is this cycle Normal or Abnormal resistance?

Input: one candidate cycle's rows (once you know its start/end).
Output: one label, `Normal` or `Abnormal resistance`.

This part is much more standard: turn each cycle's rows into a fixed-length
feature vector (duration, mean/std/peak of the motor signals, etc. — see
`src/door/features.py`), then train a classifier on the 110 labelled
examples from `Train_Segments_Answer.csv`.

**You can start on Stage 2 before Stage 1 is finished** — `Train_Segments_Answer.csv`
already gives you real segment boundaries to extract features and train a
classifier from. `src/door/train.py` does exactly this: it trains and
evaluates a classifier using the *official* ground-truth segments, without
needing your own segmentation code to work yet. This is a genuinely useful
first step, not a shortcut around the real problem — the real problem
(processing `Test.csv` end-to-end) still needs Stage 1 finished too.

## 5. Why segmentation mistakes affect the final score

Door's official metric is **IoU-weighted F1** (see the Info Kit, Section 4,
for the exact formula), not the classification accuracy you might expect.
It rewards getting *both* the timing and the label right:

- A predicted segment can only match a true segment with the **same
  label**. Perfect timing with the wrong label scores exactly like a total
  miss — it contributes nothing.
- A match's credit is its **IoU value itself** (0 to 1), not a flat point —
  sloppy boundaries on an otherwise-correct segment still cost you.
- Missing a true cycle entirely lowers your score (a "miss"). Predicting an
  extra cycle that isn't real also lowers your score (a "false positive") —
  over-segmenting the stream is not free.

**In short: a great classifier cannot rescue bad segmentation, and vice
versa.** Both stages matter, and they matter multiplicatively, not
independently.

## 6. Safe first baseline approach

1. Get `src/door/inspect_data.py` running (Section 9) and look at its output
   — confirm the gap pattern and class balance for yourself.
2. Implement a simple segmentation rule in `src/door/segment.py::detect_cycles`
   using the large-`Datetime`-gap hint from Section 2/4 to find candidate
   cycle windows, then check each candidate against `Train_Segments_Answer.csv`
   (how close are your boundaries to the true ones? did you find all 110?
   did you invent any extra ones?).
3. Once segmentation looks reasonable on `Train.csv`, use
   `src/door/train.py` (which already works against the *official* segments)
   to compare baseline classifiers (`DummyClassifier`, class-weighted
   Logistic Regression, class-weighted Random Forest) using **macro F1** as
   a classification-only proxy metric — this is NOT the official IoU-weighted
   F1 (that needs your segmentation plugged in too), but it's a fast, honest
   way to check "if segmentation were perfect, how good is my classifier?"

   **Confirmed result (2026-09-19 run of `python src/door/train.py`):
   Classification-only cross-validation using official ground-truth cycle
   boundaries achieved 1.000 macro F1. This is a proxy result and not the
   official end-to-end IoU-weighted F1. At the time of that run, automatic
   cycle segmentation was unfinished.** Don't read this as "the Door model works" — it only means
   the two classes are easy to tell apart once you're handed perfect
   boundaries, which your own `detect_cycles()` won't produce on day one.
4. Only after both stages work reasonably on `Train.csv` should you run the
   full pipeline (`detect_cycles` -> `extract_cycle_features` -> trained
   model) on `Test.csv` to produce `door_predictions.csv`.
5. Validate your output with `src/door/validate_predictions.py` before
   trusting it.

**Do not use `Test.csv` to choose features, thresholds, segmentation rules,
or model hyperparameters at any point.** Everything you tune must be
justified using `Train.csv` + `Train_Segments_Answer.csv` only, the same
rule that applies to Rail Corrugation (see the PS3 spec, Section 2.3: "design
and justify your own train/validation split").

## 7. Exact files to edit

| File | What it's for | Status |
|---|---|---|
| `src/door/config.py` | Confirmed paths/columns/labels only | Done — read it, don't need to edit unless something genuinely changes |
| `src/door/preprocess.py` | Loading + validating raw CSVs, Datetime parsing | Done (`load_door_csv`, `parse_door_datetime`) |
| `src/door/inspect_data.py` | Read-only data inspection script | Done and runnable now |
| `src/door/segment.py` | `detect_cycles()` | Implemented and validated on Train; see Stage 2 result below |
| `src/door/features.py` | `extract_cycle_features()` -- turns segments into a feature table | Done (works with any valid segments, including the official ones) |
| `src/door/model.py` | Candidate baseline models | Done (Dummy / Logistic Regression / Random Forest) |
| `src/door/train.py` | Training/evaluation workflow | Done for classification-only evaluation using official segments; extend once `detect_cycles` works |
| `src/door/predict.py` | `predict_door_file()` -- full inference pipeline | Scaffolded; will work once `detect_cycles` + a trained model both exist |
| `src/door/validate_predictions.py` | Checks your `door_predictions.csv` | Done and runnable now |
| `app/door_view.py` | Streamlit page | Done as a safe base -- extend once predictions are real |

## 8. Exact order to work in

1. Read this document fully.
2. Run `python src/door/inspect_data.py` and read its output.
3. Open `Train.csv` and `Train_Segments_Answer.csv` yourself (e.g. in a
   notebook) and plot a couple of known cycles to build intuition.
4. Implement `detect_cycles()` in `segment.py` against `Train.csv` only.
5. Run `python src/door/train.py` to see classifier performance using the
   official segments (this already works without your segmentation code).
6. Once `detect_cycles()` is solid, wire it into `predict.py` and run it on
   `Test.csv`.
7. Validate with `python src/door/validate_predictions.py`.
8. Only then consider Door "ready" for the shared app / submission.

## 9. Commands to run each stage

```powershell
# Stage 0: read-only data inspection (works right now)
python src/door/inspect_data.py

# Stage 2 preview: classifier comparison using OFFICIAL segments
# (works right now, does not need your segmentation code)
python src/door/train.py

# Save a trained classifier artifact (only once you're happy with it)
python src/door/train.py --finalize

# Full pipeline on the real Test.csv (Stage 3: only after classifier integration is validated)
python src/door/predict.py "<path to organiser Test.csv>"

# Validate your official output before trusting it
python src/door/validate_predictions.py --predictions predictions/door_predictions.csv
```

## 10. Definition of done

- [x] `detect_cycles()` matches all 110 official Train cycles with mean IoU 1.000 and zero boundary error; see Stage 2 result below.
- [x] `python src/door/train.py` reports macro F1 and per-class precision/recall/F1 for at least 2 real candidate models, compared against a `DummyClassifier` floor. Classification-only cross-validation using official ground-truth cycle boundaries achieved 1.000 macro F1. This is a proxy result and not the official end-to-end IoU-weighted F1; Stage 2 segmentation results are documented below.
- [ ] `python src/door/predict.py <Test.csv>` runs end-to-end and produces `predictions/door_predictions.csv`
- [ ] `python src/door/validate_predictions.py` passes with no errors
- [ ] The Streamlit Door page shows real validation/status, never an invented prediction
- [ ] No code path in `src/door/` or `app/door_view.py` ever reads `Test.csv` to make a segmentation, feature, or model decision

## 11. Common mistakes to avoid

- **Don't tune anything against `Test.csv`.** You don't have its true labels; if you did use it, your validation numbers would look great and mean nothing.
- **Don't assume a fixed 20ms sampling rate everywhere.** It only holds within the dense blocks (Section 2) -- always compute gaps from the actual `Datetime` values, don't hard-code a row-index-to-time formula.
- **Don't let `pandas.to_datetime` guess the format.** Use `parse_door_datetime` -- the unpadded, hyphen-separated format will silently misparse otherwise.
- **Don't treat "Open"/"Close" (`operation`) as something to predict.** It's informational only; you only predict `status` (`Normal`/`Abnormal resistance`).
- **Don't over-segment.** Every extra predicted segment that doesn't match a real cycle actively hurts your score (Section 5) -- it is not a "free" safety margin.
- **Don't silently drop or fill missing/bad rows.** There currently are none in the official files, so if your code encounters them, something upstream is probably wrong -- surface it, don't paper over it.
- **Don't fabricate a `door_predictions.csv`** just to have "something" for the app or the ZIP -- an incomplete/missing Door submission is honestly reported as missing (see `scripts/package_predictions.py`), not faked.

## 12. Open questions / unresolved assumptions

- **No official sampling frequency in Hz is stated for Door** anywhere in the Info Kit (unlike Rail Corrugation's confirmed 10,000 Hz). The ~20ms/~50Hz figure in Section 2 is *measured from the files*, not organiser-documented -- re-verify it yourself and don't hard-code it as gospel.
- **No official units are given for `Door leaf position`** (the Data Headers doc marks it "self-explanatory" with no unit). Treat its raw values as relative/arbitrary until told otherwise.
- **No official `predict.py` `--input`/`--output` CLI contract was found.** Like the Rail Corrugation Info Kit, the Door Info Kit says this is defined in "the top-level README's Deliverables section," but that section (checked in full) only describes the required **app** (upload -> prediction -> download), not a specific CLI. This is the same open question already flagged in `planning/rail_corrugation_data_audit.md` -- it appears to be a repo-wide documentation inconsistency, not something specific to Door.
- **No official guidance on exactly which signal(s) define a cycle boundary** -- this is a deliberate open design decision for you to make and justify (see Section 4).

## Stage 2 segmentation result (2026-09-19)

`src/door/segment.py::detect_cycles()` now splits the ordered sensor stream
where the difference between consecutive parsed timestamps is **strictly
greater than 100 ms**. Each cycle begins at the first reading after a gap and
ends at the last reading before the next gap. It returns those readings'
original `Datetime` strings as `start_time` and `end_time`; it does not
classify, sort, resample, interpolate, or consult the answer file.

The 100 ms threshold is frozen in `src/door/config.py`. It was selected from
**Train only**: the median and largest ordinary row interval are both 20 ms,
while the smallest of the 109 large gaps is 10,215 ms. Thus 100 ms is 5 times
the dense interval and far below the first observed inter-block gap. This
separation is an observation in this dataset, not an organiser-specified rule.

`src/door/evaluate_segmentation.py` matches detected Train intervals against
`Train_Segments_Answer.csv` one-to-one by maximum total temporal IoU. The
answer file is used only here for evaluation, never by `detect_cycles()` or
Test prediction. On the 18,036-row Train stream:

| Measure | Observed result |
|---|---:|
| Official / detected / matched cycles | 110 / 110 / 110 |
| Missing official / extra detected | 0 / 0 |
| Mean / median / minimum temporal IoU | 1.000 / 1.000 / 1.000 |
| IoU at least 0.50 / 0.75 / 0.90 | 110 / 110 / 110 |
| Mean absolute start / end boundary error | 0 ms / 0 ms |
| Chronological pairwise overlap | All 110 pairs |

Only after that Train result, the unchanged frozen method was applied to the
6,253-row **unlabelled Test** stream. It detected 38 ordered, non-overlapping
cycles, all inside the Test time range. No Test labels were available or used.
These are observed counts, not forced targets or organiser-stated facts.

**Limitations:** The rule depends on a large gap between densely recorded
cycles. A future continuous recording without such gaps, or with a dropped
packet inside a real cycle exceeding 100 ms, needs a new Train-validated
segmentation rule. The perfect Train timing result does not establish Test
classification quality or official end-to-end IoU-weighted F1. The Door app
must remain **NOT READY** until the classification and prediction pipeline is
verified end-to-end.

**Exact next step (Stage 3):** integrate the frozen detected cycles with
`extract_cycle_features()` and a validated classifier, evaluate end-to-end on
a held-out portion of Train, then generate and validate the official
`start_time,end_time,prediction` output from Test without answer-file access.
- **No official train/validation split is prescribed** -- per the PS3 spec, you must design and justify your own (e.g. holding out some of `Train.csv`'s labelled segments).
