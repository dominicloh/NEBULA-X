#!/usr/bin/env python3
"""Rail Corrugation Stage 2: reproducible, leakage-safe baseline comparison.

WHY Macro F1
------------
The organiser's official metric is macro F1 (see the Info Kit, Section 4),
because the classes are heavily imbalanced (234 Normal / 24 Side II / 14 Side
I). Macro F1 scores each class independently and then averages the three
scores unweighted, so a model that always predicts "Normal" -- which would
score ~85-90% plain accuracy -- scores only ~0.33 macro F1 (0 recall on both
fault classes). Every model selection decision in this script is therefore
made on macro F1, never on accuracy.

WHY stratified cross-validation
--------------------------------
With only 14 Side I examples in the entire training set, an ordinary random
split could easily put zero Side I files in a validation fold, making macro
F1 undefined/misleading for that fold. Stratified K-Fold guarantees every
fold keeps (approximately) the same class ratio as the full dataset. Because
14 examples is still a small number to estimate variability from a single
split, this script also runs a *repeated* stratified CV (multiple different
fold splits) so we can report how much macro F1 swings from split to split,
not just a single point estimate.

WHAT this script does NOT do (by design, per the Stage 2 brief)
------------------------------------------------------------------
- It never reads the organiser's Test/ files -- only Train/ + Train_Labels.csv.
  Test files must stay untouched until a trustworthy baseline is established.
- It never fits a scaler/selector on data outside the current training fold
  (see model.py) -- every learned transformation lives inside the
  cross-validated estimator.
- It never resamples (e.g. SMOTE) before cross-validation; class imbalance is
  handled with class_weight="balanced" only (see model.py docstring).
- It does not produce rail_predictions.csv. This stage only establishes and
  reports a validation baseline.

Usage:
    python src/rail_corrugation/train.py [dataset_dir] [--output-dir DIR]
        [--n-splits 5] [--n-repeats 10]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_validate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.validation import VALID_RAIL_LABELS  # noqa: E402
from src.rail_corrugation.features import SAMPLING_FREQUENCY_HZ, extract_features_for_files  # noqa: E402
from src.rail_corrugation.inspect_data import DEFAULT_DATASET_DIR, find_dataset_layout, list_csv_files  # noqa: E402
from src.rail_corrugation.model import RANDOM_STATE, build_candidate_models  # noqa: E402

# Fixed class order used everywhere (confusion matrix axes, per-class report
# columns, macro F1 label set) so results are directly comparable run to run.
CLASS_LABELS = list(VALID_RAIL_LABELS)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "rail_corrugation" / "baseline"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "rail_corrugation_model.joblib"


def load_training_data(dataset_root: Path, verbose: bool = True):
    """Load Train/ features and labels only. Test/ is located (to confirm the
    dataset layout is intact) but its files are never opened here.
    """
    train_dir, _test_dir, labels_path = find_dataset_layout(dataset_root)
    train_files = list_csv_files(train_dir)

    labels = pd.read_csv(labels_path)
    if "filename" not in labels.columns or "label" not in labels.columns:
        raise ValueError(
            f"Train_Labels.csv must have 'filename' and 'label' columns; found: {list(labels.columns)}"
        )
    labels_by_filename = labels.set_index("filename")["label"]

    if verbose:
        print(f"Extracting features for {len(train_files)} Train file(s) (Test/ is not read in this stage)...")
    features = extract_features_for_files(train_files, SAMPLING_FREQUENCY_HZ, verbose=verbose)

    # Align labels to the exact row order features came out in. This also
    # doubles as a leakage guard: if a filename is missing from the labels
    # file, .loc raises immediately rather than silently dropping a row.
    missing = [name for name in features.index if name not in labels_by_filename.index]
    if missing:
        raise ValueError(f"{len(missing)} Train file(s) have no label row: {missing}")
    y = labels_by_filename.loc[features.index].to_numpy()

    return features, y


def evaluate_candidate_models(X: pd.DataFrame, y: np.ndarray, n_splits: int, n_repeats: int) -> dict:
    """Evaluate every candidate model with two complementary CV schemes:

    1. A REPEATED stratified K-Fold (n_splits x n_repeats different splits)
       to estimate macro F1's mean and spread -- important because 14 Side I
       examples make any single split's score noisy.
    2. A single, fixed stratified K-Fold used with cross_val_predict to get
       one reproducible set of out-of-fold predictions per model, which is
       what per-class precision/recall/F1, balanced accuracy, and the
       confusion matrix are computed from (these need one prediction per
       file, which a repeated CV -- with multiple predictions per file --
       does not directly give).

    Returns a dict keyed by model name with all metrics needed for the
    model_comparison table and for picking the strongest baseline.
    """
    repeated_cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    single_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

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
        report_dict = classification_report(
            y, y_pred_oof, labels=CLASS_LABELS, output_dict=True, zero_division=0
        )
        confusion = confusion_matrix(y, y_pred_oof, labels=CLASS_LABELS)

        recall_side_i = report_dict["Side I"]["recall"]
        recall_side_ii = report_dict["Side II"]["recall"]

        elapsed = time.time() - start
        print(
            f"  repeated CV macro F1: {repeated_scores['test_macro_f1'].mean():.3f} "
            f"(+/- {repeated_scores['test_macro_f1'].std():.3f})  "
            f"[{elapsed:.1f}s for {n_splits * n_repeats} fold fits]"
        )

        results[model_name] = {
            "mean_macro_f1": float(repeated_scores["test_macro_f1"].mean()),
            "std_macro_f1": float(repeated_scores["test_macro_f1"].std()),
            "mean_balanced_accuracy": float(repeated_scores["test_balanced_accuracy"].mean()),
            "std_balanced_accuracy": float(repeated_scores["test_balanced_accuracy"].std()),
            "mean_fit_time_seconds": float(repeated_scores["fit_time"].mean()),
            "oof_macro_f1": float(f1_score(y, y_pred_oof, average="macro", labels=CLASS_LABELS)),
            "oof_balanced_accuracy": float(balanced_accuracy_score(y, y_pred_oof)),
            "report_dict": report_dict,
            "confusion_matrix": confusion,
            "y_pred_oof": y_pred_oof,
            "detects_both_minority_classes": bool(recall_side_i > 0 and recall_side_ii > 0),
        }
    return results


def build_comparison_table(results: dict) -> pd.DataFrame:
    """Flatten the per-model results dict into the model_comparison.csv table."""
    rows = []
    for model_name, r in results.items():
        row = {
            "model": model_name,
            "mean_macro_f1": r["mean_macro_f1"],
            "std_macro_f1": r["std_macro_f1"],
            "mean_balanced_accuracy": r["mean_balanced_accuracy"],
            "std_balanced_accuracy": r["std_balanced_accuracy"],
            "oof_macro_f1": r["oof_macro_f1"],
            "oof_balanced_accuracy": r["oof_balanced_accuracy"],
            "mean_fit_time_seconds": r["mean_fit_time_seconds"],
            "detects_both_minority_classes": r["detects_both_minority_classes"],
        }
        for label in CLASS_LABELS:
            key = label.lower().replace(" ", "_")
            class_report = r["report_dict"][label]
            row[f"precision_{key}"] = class_report["precision"]
            row[f"recall_{key}"] = class_report["recall"]
            row[f"f1_{key}"] = class_report["f1-score"]
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("mean_macro_f1", ascending=False).reset_index(drop=True)
    return table


def select_best_model(comparison_table: pd.DataFrame) -> tuple:
    """Pick the strongest baseline by macro F1, but only among models that
    detect BOTH minority classes at least some of the time (task requirement:
    never pick a model just because its overall/macro number looks good while
    it silently ignores a fault class). Falls back to the highest macro F1
    model with a loud warning if no candidate clears that bar.
    """
    detects_both = comparison_table[comparison_table["detects_both_minority_classes"]]
    if not detects_both.empty:
        best_row = detects_both.iloc[0]
        warning = None
    else:
        best_row = comparison_table.iloc[0]
        warning = (
            f"WARNING: no candidate model detects both Side I and Side II in out-of-fold "
            f"predictions. Falling back to the highest macro F1 model ({best_row['model']}), "
            "but this baseline is NOT yet trustworthy for the minority classes."
        )
    return best_row["model"], warning


def train_final_model(
    X: pd.DataFrame,
    y: np.ndarray,
    model_name: str,
    validation_summary: dict,
    output_path: Path,
) -> dict:
    """Fit the selected model's exact pipeline ONCE on all 272 labelled Train
    files (no CV split here -- this is the real, deployable artifact) and
    persist everything needed to predict later without ever refitting.

    Stage 2's cross-validation intentionally fit the scaler/model fresh inside
    every fold so the validation score never saw held-out data. Stage 3 is
    different: there is no more held-out validation happening once we commit
    to a final model, so fitting the exact same pipeline architecture on ALL
    labelled training rows (not a subset) is the correct thing to do -- more
    data in the final fit, still zero test-file leakage since Test/ is never
    read here.

    `clone()` gives a fresh, UNFITTED copy of the exact pipeline architecture
    and hyperparameters that were actually evaluated by cross-validation, so
    the final model can't silently drift from what was validated.
    """
    candidate_models = build_candidate_models()
    if model_name not in candidate_models:
        raise ValueError(
            f"Unknown model name for final training: {model_name!r}. "
            f"Available: {sorted(candidate_models)}"
        )
    pipeline = clone(candidate_models[model_name])
    pipeline.fit(X, y)

    artifact = {
        "pipeline": pipeline,
        "feature_names": list(X.columns),
        "class_labels": CLASS_LABELS,
        "model_name": model_name,
        "sampling_frequency_hz": SAMPLING_FREQUENCY_HZ,
        "random_state": RANDOM_STATE,
        "n_training_files": int(len(X)),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        # Cross-validation ESTIMATES from Stage 2/3 evaluation on the 272
        # labelled Train files only. These describe expected performance on
        # unseen data of the same kind -- they are NOT a measurement of
        # accuracy on the actual hidden test set, whose true labels the team
        # does not have access to.
        "validation_summary": validation_summary,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    return artifact


def save_confusion_matrix_plot(confusion: np.ndarray, model_name: str, output_path: Path) -> None:
    """Save a confusion matrix heatmap for the strongest baseline.

    Uses a single-hue sequential colormap ("Blues") since cell counts are a
    magnitude, not a category -- and annotates every cell with its count
    directly rather than relying on color alone to convey the value.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
        cbar_kws={"label": "Number of files"},
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{model_name} -- out-of-fold confusion matrix\n(5-fold stratified CV, random_state={RANDOM_STATE})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rail Corrugation Stage 2: leakage-safe baseline model comparison (Train/ only, no test predictions)."
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default=str(DEFAULT_DATASET_DIR),
        help=f"Rail_Corrugation dataset directory. Default: {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to save baseline evaluation outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Number of stratified CV folds (default: 5).")
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=10,
        help="Number of repeats of the n-splits CV, to measure macro F1 variability (default: 10).",
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help=(
            "After comparing candidates, also fit the strongest baseline's exact pipeline on ALL "
            "272 labelled Train files (Test/ still never read) and save the fitted artifact to "
            "--model-path. Off by default so a plain comparison run never silently overwrites the "
            "deployed model."
        ),
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help=f"Where to save the final model artifact when --finalize is set. Default: {DEFAULT_MODEL_PATH}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_start = time.time()

    print("=" * 78)
    print("RAIL CORRUGATION -- STAGE 2 BASELINE (Train/ only, no test predictions)")
    print("=" * 78)
    print(f"Dataset root: {dataset_root}")
    print(f"Output dir:   {output_dir}")

    X, y = load_training_data(dataset_root)
    n_files, n_features = X.shape
    print(f"\nFeature matrix: {n_files} files x {n_features} features")
    class_counts = pd.Series(y).value_counts()
    print("Class distribution used for training:")
    for label in CLASS_LABELS:
        print(f"  {label:10s}: {int(class_counts.get(label, 0))}")

    results = evaluate_candidate_models(X, y, n_splits=args.n_splits, n_repeats=args.n_repeats)
    comparison_table = build_comparison_table(results)
    best_model_name, warning = select_best_model(comparison_table)
    best_result = results[best_model_name]

    print("\n" + "=" * 78)
    print("MODEL COMPARISON (sorted by mean macro F1, repeated stratified CV)")
    print("=" * 78)
    print(
        comparison_table[
            ["model", "mean_macro_f1", "std_macro_f1", "oof_macro_f1", "mean_balanced_accuracy", "detects_both_minority_classes"]
        ].to_string(index=False)
    )
    print(f"\nStrongest baseline: {best_model_name}")
    if warning:
        print(warning)

    # --- Save outputs -------------------------------------------------
    comparison_table.to_csv(output_dir / "model_comparison.csv", index=False)

    feature_names_path = output_dir / "feature_names.txt"
    with open(feature_names_path, "w", encoding="utf-8") as f:
        f.write(f"{n_features} Rail Corrugation baseline features (deterministic; no filename/order-derived features)\n")
        for name in X.columns:
            f.write(f"{name}\n")

    classification_report_text = classification_report(
        y, best_result["y_pred_oof"], labels=CLASS_LABELS, zero_division=0
    )
    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Rail Corrugation Stage 2 baseline -- classification report\n")
        f.write(f"Best model: {best_model_name}\n")
        f.write(
            f"Predictions: out-of-fold, {args.n_splits}-fold stratified CV, random_state={RANDOM_STATE}\n"
        )
        f.write(f"Macro F1 (out-of-fold): {best_result['oof_macro_f1']:.4f}\n")
        f.write(f"Balanced accuracy (out-of-fold): {best_result['oof_balanced_accuracy']:.4f}\n\n")
        f.write(classification_report_text)
        if warning:
            f.write(f"\n\n{warning}\n")

    save_confusion_matrix_plot(
        best_result["confusion_matrix"], best_model_name, output_dir / "confusion_matrix.png"
    )

    total_runtime_seconds = time.time() - overall_start
    summary = {
        "dataset_root": str(dataset_root),
        "n_train_files": n_files,
        "n_features": n_features,
        "class_distribution": {label: int(class_counts.get(label, 0)) for label in CLASS_LABELS},
        "cv_scheme": {
            "repeated_cv": f"RepeatedStratifiedKFold(n_splits={args.n_splits}, n_repeats={args.n_repeats})",
            "single_fixed_cv_for_oof_predictions": f"StratifiedKFold(n_splits={args.n_splits}, shuffle=True)",
            "random_state": RANDOM_STATE,
        },
        "primary_metric": "macro_f1",
        "best_model": best_model_name,
        "best_model_warning": warning,
        "models": {
            name: {
                "mean_macro_f1": r["mean_macro_f1"],
                "std_macro_f1": r["std_macro_f1"],
                "oof_macro_f1": r["oof_macro_f1"],
                "mean_balanced_accuracy": r["mean_balanced_accuracy"],
                "mean_fit_time_seconds": r["mean_fit_time_seconds"],
                "detects_both_minority_classes": r["detects_both_minority_classes"],
            }
            for name, r in results.items()
        },
        "total_runtime_seconds": round(total_runtime_seconds, 1),
        "test_files_used": False,
        "final_test_predictions_generated": False,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved outputs under: {output_dir}")

    if args.finalize:
        model_path = Path(args.model_path).expanduser().resolve()
        print(f"\n--finalize set: fitting {best_model_name} on all {n_files} labelled Train files...")
        best_comparison_row = comparison_table[comparison_table["model"] == best_model_name].iloc[0]
        validation_summary = {
            "selected_model": best_model_name,
            "selection_rule": "highest mean macro F1 among models that detect both Side I and Side II in out-of-fold predictions",
            "cv_scheme": summary["cv_scheme"],
            "mean_macro_f1_repeated_cv": best_comparison_row["mean_macro_f1"],
            "std_macro_f1_repeated_cv": best_comparison_row["std_macro_f1"],
            "oof_macro_f1_fixed_5fold": best_comparison_row["oof_macro_f1"],
            "oof_balanced_accuracy_fixed_5fold": best_comparison_row["oof_balanced_accuracy"],
            "note": (
                "These are cross-validation ESTIMATES computed on the 272 labelled Train files only. "
                "The hidden Test file labels are not available to the team, so this is not a measurement "
                "of accuracy on the actual test set -- only a validation estimate of expected performance."
            ),
        }
        train_final_model(X, y, best_model_name, validation_summary, model_path)
        artifact_size_kb = model_path.stat().st_size / 1024
        print(f"Saved final model artifact to: {model_path} ({artifact_size_kb:.1f} KB)")

    print(f"Total runtime: {total_runtime_seconds:.1f}s")
    print("\nNo test predictions were generated. Test/ files were never read.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
