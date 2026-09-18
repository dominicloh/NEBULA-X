#!/usr/bin/env python3
"""Train-only Door classification using automatically detected cycle boundaries."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score, make_scorer, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_validate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.door import config  # noqa: E402
from src.door.features import FEATURE_NAMES, extract_cycle_features  # noqa: E402
from src.door.model import build_candidate_models, save_door_pipeline  # noqa: E402
from src.door.preprocess import load_door_csv, parse_door_datetime  # noqa: E402
from src.door.segment import detect_cycles, validate_segment_table  # noqa: E402

CLASS_LABELS = list(config.VALID_DOOR_LABELS)


def load_training_data(dataset_dir: Path, verbose: bool = True) -> tuple[pd.DataFrame, np.ndarray]:
    """Attach one answer label to each detected cycle after exact one-to-one matching."""
    stream = load_door_csv(dataset_dir / config.TRAIN_FILENAME)
    detected = detect_cycles(stream)
    answers = pd.read_csv(dataset_dir / config.TRAIN_SEGMENTS_ANSWER_FILENAME, dtype=str)
    missing = set(config.SEGMENT_ANSWER_COLUMNS) - set(answers.columns)
    if missing:
        raise ValueError(f"Door answers are missing columns: {sorted(missing)}")
    problems = validate_segment_table(answers, stream)
    if problems:
        raise ValueError(f"Door answers failed validation: {problems}")
    if len(detected) != len(answers):
        raise ValueError(f"Detected/official Door cycle counts differ: {len(detected)} / {len(answers)}")
    boundary_cols = ["start_time", "end_time"]
    if detected.duplicated(boundary_cols).any() or answers.duplicated(boundary_cols).any():
        raise ValueError("Door cycle boundaries must be unique for one-to-one labelling.")
    labelled = detected.merge(answers[boundary_cols + ["status"]], on=boundary_cols, how="left", validate="one_to_one", indicator=True)
    if (labelled["_merge"] != "both").any() or len(labelled) != len(answers):
        raise ValueError("Every detected Door cycle must match exactly one official cycle with IoU 1.000.")
    labels = labelled["status"].to_numpy()
    if set(labels) != set(CLASS_LABELS):
        raise ValueError(f"Door answers contain unexpected labels: {sorted(set(labels) - set(CLASS_LABELS))}")
    X = extract_cycle_features(stream, detected)
    if list(X.columns) != list(FEATURE_NAMES) or len(X) != len(labels):
        raise ValueError("Door feature rows/order do not match detected cycles.")
    if verbose:
        print(f"Train: {len(stream)} readings, {len(detected)} exact detected/official cycle matches, {X.shape[1]} features")
    return X, labels


def evaluate_candidate_models(X: pd.DataFrame, y: np.ndarray, n_splits: int = 5, n_repeats: int = 10) -> dict:
    """Repeated stratified CV and one fixed out-of-fold prediction per cycle."""
    if list(X.columns) != list(FEATURE_NAMES) or not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Door feature matrix has invalid order or nonfinite values.")
    if len(X) != len(y) or len(set(y) - set(CLASS_LABELS)):
        raise ValueError("Door feature/label matrix is invalid.")
    repeated_cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=config.RANDOM_STATE)
    single_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    results = {}
    for name, estimator in build_candidate_models().items():
        scores = cross_validate(estimator, X, y, cv=repeated_cv,
                                scoring={"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy",
                                         "abnormal_recall": make_scorer(recall_score, pos_label="Abnormal resistance")}, n_jobs=1)
        predicted = cross_val_predict(estimator, X, y, cv=single_cv, n_jobs=1)
        report = classification_report(y, predicted, labels=CLASS_LABELS, output_dict=True, zero_division=0)
        results[name] = {
            "mean_macro_f1": float(scores["test_macro_f1"].mean()),
            "std_macro_f1": float(scores["test_macro_f1"].std()),
            "min_macro_f1": float(scores["test_macro_f1"].min()),
            "max_macro_f1": float(scores["test_macro_f1"].max()),
            "mean_balanced_accuracy": float(scores["test_balanced_accuracy"].mean()),
            "mean_abnormal_recall": float(scores["test_abnormal_recall"].mean()),
            "oof_macro_f1": float(f1_score(y, predicted, labels=CLASS_LABELS, average="macro")),
            "oof_balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "report_dict": report,
            "confusion_matrix": confusion_matrix(y, predicted, labels=CLASS_LABELS).tolist(),
            "y_pred_oof": predicted,
        }
    return results


def build_comparison_table(results: dict) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        row = {"model": name}
        for key in ("mean_macro_f1", "std_macro_f1", "min_macro_f1", "max_macro_f1",
                    "mean_balanced_accuracy", "mean_abnormal_recall", "oof_macro_f1", "oof_balanced_accuracy"):
            row[key] = result[key]
        row["abnormal_recall"] = result["report_dict"]["Abnormal resistance"]["recall"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["mean_macro_f1", "abnormal_recall", "std_macro_f1", "model"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def chronological_holdout(X: pd.DataFrame, y: np.ndarray, model_name: str) -> dict:
    """Train on first 80% of cycles and test on the last 20%, preserving time order."""
    cut = int(len(y) * 0.8)
    fitted = clone(build_candidate_models()[model_name]).fit(X.iloc[:cut], y[:cut])
    predicted = fitted.predict(X.iloc[cut:])
    return {"train_cycles": cut, "holdout_cycles": len(y) - cut,
            "train_class_counts": pd.Series(y[:cut]).value_counts().to_dict(),
            "holdout_class_counts": pd.Series(y[cut:]).value_counts().to_dict(),
            "macro_f1": float(f1_score(y[cut:], predicted, labels=CLASS_LABELS, average="macro", zero_division=0)),
            "abnormal_recall": float(classification_report(y[cut:], predicted, labels=CLASS_LABELS,
                                                             output_dict=True, zero_division=0)["Abnormal resistance"]["recall"])}


def permuted_label_check(X: pd.DataFrame, y: np.ndarray, model_name: str) -> float:
    """A fixed five-fold shuffled-label check for suspiciously perfect results."""
    shuffled = np.random.default_rng(config.RANDOM_STATE).permutation(y)
    predicted = cross_val_predict(build_candidate_models()[model_name], X, shuffled,
                                  cv=StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE), n_jobs=1)
    return float(f1_score(shuffled, predicted, labels=CLASS_LABELS, average="macro"))


def official_iou_weighted_f1(predicted: pd.DataFrame, official: pd.DataFrame) -> float:
    """Info Kit Section 4: greedy same-label IoU matching and soft F1."""
    if predicted.empty or official.empty:
        return 0.0
    candidates = []
    for i, found in predicted.reset_index(drop=True).iterrows():
        pred_start, pred_end = (parse_door_datetime(str(found[column])) for column in ("start_time", "end_time"))
        for j, truth in official.reset_index(drop=True).iterrows():
            if found["prediction"] != truth["status"]:
                continue
            true_start, true_end = (parse_door_datetime(str(truth[column])) for column in ("start_time", "end_time"))
            intersection = max(0.0, (min(pred_end, true_end) - max(pred_start, true_start)).total_seconds())
            union = (pred_end - pred_start).total_seconds() + (true_end - true_start).total_seconds() - intersection
            if intersection > 0 and union > 0:
                candidates.append((intersection / union, i, j))
    used_pred, used_true, credit = set(), set(), 0.0
    for iou, i, j in sorted(candidates, key=lambda item: (-item[0], item[1], item[2])):
        if i not in used_pred and j not in used_true:
            used_pred.add(i)
            used_true.add(j)
            credit += iou
    soft_recall = credit / len(official)
    soft_precision = credit / len(predicted)
    return 2 * soft_recall * soft_precision / (soft_recall + soft_precision) if credit else 0.0


def _jsonable_result(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "y_pred_oof"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", nargs="?", default=str(config.DEFAULT_DATASET_DIR))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "door" / "classification")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--model-path", type=Path, default=config.MODEL_ARTIFACT_PATH)
    args = parser.parse_args()
    X, y = load_training_data(Path(args.dataset_dir))
    results = evaluate_candidate_models(X, y, args.n_splits, args.n_repeats)
    comparison = build_comparison_table(results)
    choice = str(comparison.iloc[0]["model"])
    best = results[choice]
    holdout = chronological_holdout(X, y, choice)
    permutation = permuted_label_check(X, y, choice) if best["oof_macro_f1"] >= 0.99 else None
    train_segments = detect_cycles(load_door_csv(Path(args.dataset_dir) / config.TRAIN_FILENAME))
    oof_segments = train_segments[["start_time", "end_time"]].copy()
    oof_segments["prediction"] = best["y_pred_oof"]
    official_segments = pd.read_csv(Path(args.dataset_dir) / config.TRAIN_SEGMENTS_ANSWER_FILENAME, dtype=str)
    train_oof_official_score = official_iou_weighted_f1(oof_segments, official_segments)
    summary = {
        "train_cycles": len(y), "class_counts": pd.Series(y).value_counts().to_dict(),
        "feature_count": len(FEATURE_NAMES), "feature_names": list(FEATURE_NAMES),
        "cv": {"method": "RepeatedStratifiedKFold", "folds": args.n_splits, "repeats": args.n_repeats,
               "random_state": config.RANDOM_STATE, "oof_method": "fixed shuffled StratifiedKFold"},
        "models": {name: _jsonable_result(result) for name, result in results.items()},
        "chosen_model": choice, "chronological_holdout": holdout,
        "permuted_label_oof_macro_f1": permutation,
        "train_oof_iou_weighted_f1": train_oof_official_score,
        "metric_note": "Train classification on automatically detected, exact-match cycles; hidden Test accuracy is unknown.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_dir / "model_comparison.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "feature_names.txt").write_text("\n".join(FEATURE_NAMES) + "\n", encoding="utf-8")
    (args.output_dir / "classification_report.txt").write_text(
        classification_report(y, best["y_pred_oof"], labels=CLASS_LABELS, zero_division=0), encoding="utf-8")
    print(comparison.to_string(index=False))
    print(f"Selected: {choice}; blocked holdout macro F1={holdout['macro_f1']:.3f}; permuted-label macro F1={permutation}")
    if args.finalize:
        pipeline = clone(build_candidate_models()[choice]).fit(X, y)
        artifact = {"pipeline": pipeline, "feature_names": list(FEATURE_NAMES), "class_labels": CLASS_LABELS,
                    "model_name": choice, "n_training_segments": len(y),
                    "segmentation_threshold_ms": config.SEGMENTATION_CONFIG["gap_threshold_ms"],
                    "sklearn_version": sklearn.__version__, "trained_at_utc": datetime.now(timezone.utc).isoformat(),
                    "validation_summary": {"mean_macro_f1": best["mean_macro_f1"],
                                           "oof_macro_f1": best["oof_macro_f1"]}}
        save_door_pipeline(args.model_path, artifact)
        print(f"Saved final Door model: {args.model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
