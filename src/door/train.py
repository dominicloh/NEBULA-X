#!/usr/bin/env python3
"""Door Stage 2 (classification only): train and compare baseline classifiers
using the OFFICIAL ground-truth segments from Train_Segments_Answer.csv.

WHY this works before segmentation (detect_cycles) is implemented
--------------------------------------------------------------------
This script never calls `segment.py::detect_cycles`. It extracts features
directly from the 110 officially labelled segments, so you can compare
classifiers for real right now. This validates ONLY the classification
stage -- it says nothing about how well your own segmentation will find
cycles in Test.csv. See planning/door_handoff.md Section 4 for why both
stages matter for the real competition score.

WHY macro F1 here (and why it is NOT the official Door metric)
------------------------------------------------------------------
The official Door metric is IoU-weighted F1 over predicted TEMPORAL
SEGMENTS (see the Door Info Kit, Section 4) -- it scores segmentation and
classification together. This script only has real segments to work with
(the official ones), so there is no segmentation quality to measure yet.
Macro F1 over the two classes (Normal / Abnormal resistance) is reported
instead, as an honest classification-only PROXY metric, clearly labelled as
such everywhere it's printed or saved.

WHY stratified cross-validation
--------------------------------
Door's training segments are imbalanced (80 Normal / 30 Abnormal
resistance, ~73%/27%) -- less extreme than Rail Corrugation's 86%/8.8%/5.1%,
but still imbalanced enough that a random split could easily under- or
over-represent the minority class in a fold. Stratified K-Fold keeps each
fold's class balance close to the full dataset's.

Usage:
    python src/door/train.py [dataset_dir] [--output-dir DIR]
        [--n-splits 5] [--n-repeats 5] [--finalize] [--model-path PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_validate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.door import config  # noqa: E402
from src.door.features import extract_cycle_features  # noqa: E402
from src.door.model import build_candidate_models, save_door_pipeline  # noqa: E402
from src.door.preprocess import load_door_csv  # noqa: E402
from src.door.segment import validate_segment_table  # noqa: E402

CLASS_LABELS = list(config.VALID_DOOR_LABELS)


def load_training_data(dataset_dir: Path, verbose: bool = True):
    """Load Train.csv + Train_Segments_Answer.csv and extract one feature
    row per OFFICIAL labelled segment. Test.csv is never read here.
    """
    train_path = dataset_dir / config.TRAIN_FILENAME
    answer_path = dataset_dir / config.TRAIN_SEGMENTS_ANSWER_FILENAME
    if not answer_path.is_file():
        raise FileNotFoundError(f"Train_Segments_Answer.csv not found at: {answer_path}")

    if verbose:
        print(f"Loading Train.csv from: {train_path}")
    train_stream = load_door_csv(train_path)

    answers = pd.read_csv(answer_path)
    missing_columns = set(config.SEGMENT_ANSWER_COLUMNS) - set(answers.columns)
    if missing_columns:
        raise ValueError(f"Train_Segments_Answer.csv is missing expected column(s): {sorted(missing_columns)}")

    problems = validate_segment_table(answers, train_stream)
    if problems:
        raise ValueError("Train_Segments_Answer.csv failed validation:\n  " + "\n  ".join(problems))

    if verbose:
        print(f"Extracting features for {len(answers)} official labelled segments...")
    features = extract_cycle_features(train_stream, answers)
    y = answers.set_index("segment_id")["status"].loc[features.index].to_numpy()

    return features, y


def evaluate_candidate_models(X: pd.DataFrame, y: np.ndarray, n_splits: int, n_repeats: int) -> dict:
    """Same two-CV-scheme pattern as src/rail_corrugation/train.py: a
    repeated stratified K-Fold for a mean/std estimate, and one fixed
    stratified K-Fold (via cross_val_predict) for a single reproducible
    confusion matrix and per-class report.
    """
    repeated_cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=config.RANDOM_STATE)
    single_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)

    results = {}
    for model_name, estimator in build_candidate_models().items():
        print(f"\nEvaluating {model_name} ...")
        start = time.time()

        repeated_scores = cross_validate(
            estimator,
            X,
            y,
            cv=repeated_cv,
            scoring={"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy"},
            n_jobs=-1,
        )
        y_pred_oof = cross_val_predict(estimator, X, y, cv=single_cv, n_jobs=-1)
        report_dict = classification_report(y, y_pred_oof, labels=CLASS_LABELS, output_dict=True, zero_division=0)
        confusion = confusion_matrix(y, y_pred_oof, labels=CLASS_LABELS)

        elapsed = time.time() - start
        print(
            f"  repeated CV macro F1 (proxy metric, NOT official IoU-weighted F1): "
            f"{repeated_scores['test_macro_f1'].mean():.3f} (+/- {repeated_scores['test_macro_f1'].std():.3f})  "
            f"[{elapsed:.1f}s for {n_splits * n_repeats} fold fits]"
        )

        results[model_name] = {
            "mean_macro_f1": float(repeated_scores["test_macro_f1"].mean()),
            "std_macro_f1": float(repeated_scores["test_macro_f1"].std()),
            "mean_balanced_accuracy": float(repeated_scores["test_balanced_accuracy"].mean()),
            "mean_fit_time_seconds": float(repeated_scores["fit_time"].mean()),
            "oof_macro_f1": float(f1_score(y, y_pred_oof, average="macro", labels=CLASS_LABELS)),
            "oof_balanced_accuracy": float(balanced_accuracy_score(y, y_pred_oof)),
            "report_dict": report_dict,
            "confusion_matrix": confusion.tolist(),
            "y_pred_oof": y_pred_oof,
        }
    return results


def build_comparison_table(results: dict) -> pd.DataFrame:
    rows = []
    for model_name, r in results.items():
        row = {
            "model": model_name,
            "mean_macro_f1": r["mean_macro_f1"],
            "std_macro_f1": r["std_macro_f1"],
            "mean_balanced_accuracy": r["mean_balanced_accuracy"],
            "oof_macro_f1": r["oof_macro_f1"],
            "oof_balanced_accuracy": r["oof_balanced_accuracy"],
            "mean_fit_time_seconds": r["mean_fit_time_seconds"],
        }
        for label in CLASS_LABELS:
            key = label.lower().replace(" ", "_")
            class_report = r["report_dict"][label]
            row[f"precision_{key}"] = class_report["precision"]
            row[f"recall_{key}"] = class_report["recall"]
            row[f"f1_{key}"] = class_report["f1-score"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_macro_f1", ascending=False).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Door Stage 2: classifier comparison using OFFICIAL ground-truth segments "
        "(does not require detect_cycles() to be implemented)."
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default=str(config.DEFAULT_DATASET_DIR),
        help=f"Door dataset directory. Default: {config.DEFAULT_DATASET_DIR}",
    )
    parser.add_argument("--output-dir", default=str(config.MODEL_OUTPUT_DIR), help="Directory to save evaluation outputs.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of stratified CV folds (default: 5).")
    parser.add_argument("--n-repeats", type=int, default=5, help="Number of CV repeats for the mean/std estimate (default: 5).")
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="After comparing candidates, also fit the strongest classifier on ALL labelled segments "
        "and save it to --model-path. This saves a CLASSIFIER ONLY -- it still cannot produce "
        "door_predictions.csv until detect_cycles() is implemented (see predict.py).",
    )
    parser.add_argument("--model-path", default=str(config.MODEL_ARTIFACT_PATH), help="Where to save the classifier when --finalize is set.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("DOOR -- CLASSIFIER COMPARISON (official segments only; Test.csv never read)")
    print("=" * 78)
    print(f"Dataset dir: {dataset_dir}")

    X, y = load_training_data(dataset_dir)
    n_segments, n_features = X.shape
    print(f"\nFeature matrix: {n_segments} segments x {n_features} features")
    class_counts = pd.Series(y).value_counts()
    for label in CLASS_LABELS:
        print(f"  {label:22s}: {int(class_counts.get(label, 0))}")

    results = evaluate_candidate_models(X, y, n_splits=args.n_splits, n_repeats=args.n_repeats)
    comparison_table = build_comparison_table(results)
    best_model_name = comparison_table.iloc[0]["model"]
    best_oof_macro_f1 = comparison_table.iloc[0]["oof_macro_f1"]

    # Canonical wording for this finding -- used everywhere it's reported
    # (console, classification_report.txt, summary.json) so the framing
    # never drifts between places. Always describes the ACTUAL computed
    # number for this run, not a hard-coded "1.000".
    proxy_result_sentence = (
        f"Classification-only cross-validation using official ground-truth cycle boundaries "
        f"achieved {best_oof_macro_f1:.3f} macro F1. This is a proxy result and not the official "
        f"end-to-end IoU-weighted F1. Automatic cycle segmentation remains unfinished."
    )

    print("\n" + "=" * 78)
    print("MODEL COMPARISON (sorted by mean macro F1 -- a classification-only PROXY metric,")
    print("NOT the official IoU-weighted F1, which also depends on segmentation)")
    print("=" * 78)
    print(comparison_table[["model", "mean_macro_f1", "std_macro_f1", "oof_macro_f1", "mean_balanced_accuracy"]].to_string(index=False))
    print(f"\nStrongest classifier (proxy metric): {best_model_name}")
    print(proxy_result_sentence)

    comparison_table.to_csv(output_dir / "model_comparison.csv", index=False)
    best_result = results[best_model_name]
    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write("Door classifier comparison -- classification-only proxy metric\n")
        f.write(f"Best model: {best_model_name}\n")
        f.write(proxy_result_sentence + "\n\n")
        f.write(classification_report(y, best_result["y_pred_oof"], labels=CLASS_LABELS, zero_division=0))

    summary = {
        "n_segments": n_segments,
        "n_features": n_features,
        "class_distribution": {label: int(class_counts.get(label, 0)) for label in CLASS_LABELS},
        "primary_metric": "macro_f1 (classification-only proxy; official metric is IoU-weighted F1)",
        "proxy_result_statement": proxy_result_sentence,
        "best_model": best_model_name,
        "test_files_used": False,
        "final_door_predictions_generated": False,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved outputs under: {output_dir}")

    if args.finalize:
        model_path = Path(args.model_path).expanduser().resolve()
        print(f"\n--finalize set: fitting {best_model_name} on all {n_segments} labelled segments...")
        pipeline = clone(build_candidate_models()[best_model_name])
        pipeline.fit(X, y)
        artifact = {
            "pipeline": pipeline,
            "feature_names": list(X.columns),
            "class_labels": CLASS_LABELS,
            "model_name": best_model_name,
            "n_training_segments": int(n_segments),
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "validation_summary": {
                "metric": "macro_f1 (classification-only proxy)",
                "mean_macro_f1": comparison_table.iloc[0]["mean_macro_f1"],
                "note": proxy_result_sentence,
            },
        }
        save_door_pipeline(model_path, artifact)
        print(f"Saved classifier artifact to: {model_path}")
        print("NOTE: this is a CLASSIFIER checkpoint only. predict.py still needs detect_cycles()")
        print("implemented before it can produce real door_predictions.csv from Test.csv.")

    print("\nTest.csv was never read. No door_predictions.csv was generated by this script.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
