"""Additive calibration analysis for the finalized ten-seed FGDS study.

The analysis reads preserved prediction probabilities. It does not train or
modify a model. Metric definitions are explicit so the manuscript can report
proper scoring rules and binning sensitivity without relying on library
defaults that may change across versions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.wsnds.evidence_completion.analyze_fgds_group_balanced_routes import (
    exact_paired_wilcoxon,
    holm_adjust,
)


PROBABILITY_SUM_ATOL = 2e-6
NLL_FLOOR = 1e-15
EXPECTED_SEEDS = [42, 123, 456, 789, 1001, 2024, 3141, 5678, 8192, 9999]
EXPECTED_TEST_ROWS = 56_301
CLASS_NAMES = ("Blackhole", "Flooding", "Grayhole", "Normal", "TDMA")
PROBABILITY_COLUMNS = tuple(
    f"probability_{index}_{name}" for index, name in enumerate(CLASS_NAMES)
)
ECE_BIN_COUNTS = (10, 15, 20, 30)
ADAPTIVE_ECE_BIN_COUNT = 15
PROTOCOL_ID = "wsnds_fgds_calibration_10seed_v2"
SOURCE_PROTOCOL_ID = "wsnds_feature_group_split_train_only_scaler_10seed_v2"
DEFAULT_RUN_ROOT = (
    REPO_ROOT
    / "results/wsnds/confirmation_runs_v2/local_feature_group_10seed_20260811"
    / "feature_group_10seed"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results/wsnds/evidence_completion_20260816"
    / "fgds_calibration_10seed_v2"
)
STUDENTS = {
    "student_A": {
        "scratch": "student_A_Small_MLP_scratch_test_predictions.csv",
        "rf_kd": "student_A_KD_from_RF_test_predictions.csv",
    },
    "student_B": {
        "scratch": "student_B_Small_MLP_scratch_test_predictions.csv",
        "rf_kd": "student_B_KD_from_RF_test_predictions.csv",
    },
}
PROPER_SCORE_METRICS = ("brier_multiclass", "negative_log_likelihood")


def _validated_inputs(
    probabilities: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional array")
    if labels.ndim != 1 or len(labels) != len(probabilities):
        raise ValueError("labels must contain one value per probability row")
    if probabilities.shape[1] < 2:
        raise ValueError("probabilities must contain at least two classes")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain non-finite values")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("probabilities must lie in [0, 1]")
    if not np.allclose(
        probabilities.sum(axis=1), 1.0, rtol=0.0, atol=PROBABILITY_SUM_ATOL
    ):
        raise ValueError("probability rows must sum to one")
    if np.any(labels < 0) or np.any(labels >= probabilities.shape[1]):
        raise ValueError("labels contain an invalid class index")
    return probabilities, labels


def multiclass_brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Return mean sum-of-squares multiclass Brier score in the range [0, 2]."""

    probabilities, labels = _validated_inputs(probabilities, labels)
    targets = np.zeros_like(probabilities)
    targets[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probabilities - targets) ** 2, axis=1)))


def negative_log_likelihood(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Return mean natural-log loss with an explicit numerical floor."""

    return float(negative_log_likelihood_details(probabilities, labels)["floored_nll"])


def negative_log_likelihood_details(
    probabilities: np.ndarray, labels: np.ndarray
) -> dict[str, Any]:
    """Return auditable raw and numerically floored natural-log loss details."""

    probabilities, labels = _validated_inputs(probabilities, labels)
    true_class_probability = probabilities[np.arange(len(labels)), labels]
    zero_count = int(np.count_nonzero(true_class_probability == 0.0))
    raw_nll = (
        None
        if zero_count
        else float(-np.mean(np.log(true_class_probability)))
    )
    floored_nll = float(
        -np.mean(np.log(np.maximum(true_class_probability, NLL_FLOOR)))
    )
    return {
        "raw_nll": raw_nll,
        "floored_nll": floored_nll,
        "true_probability_zero_count": zero_count,
        "probability_floor": NLL_FLOOR,
    }


def _equal_width_binary_calibration(
    confidence: np.ndarray, outcomes: np.ndarray, n_bins: int
) -> dict[str, Any]:
    if n_bins < 2:
        raise ValueError("n_bins must be at least two")
    confidence = np.asarray(confidence, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    if confidence.ndim != 1 or outcomes.shape != confidence.shape:
        raise ValueError("confidence and outcomes must be matching vectors")
    if len(confidence) == 0:
        raise ValueError("calibration vectors must not be empty")
    if not np.isfinite(confidence).all() or not np.isfinite(outcomes).all():
        raise ValueError("calibration vectors contain non-finite values")
    if np.any(confidence < 0.0) or np.any(confidence > 1.0):
        raise ValueError("confidence values must lie in [0, 1]")
    if np.any((outcomes != 0.0) & (outcomes != 1.0)):
        raise ValueError("calibration outcomes must be binary")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    assignments = np.searchsorted(edges, confidence, side="left") - 1
    assignments = np.clip(assignments, 0, n_bins - 1)
    bins: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(n_bins):
        selected = assignments == index
        count = int(selected.sum())
        mean_confidence = float(confidence[selected].mean()) if count else None
        accuracy = float(outcomes[selected].mean()) if count else None
        absolute_gap = (
            abs(float(accuracy) - float(mean_confidence)) if count else None
        )
        if count:
            ece += (count / len(confidence)) * float(absolute_gap)
        bins.append(
            {
                "bin_index": index,
                "lower_bound_exclusive": float(edges[index]),
                "upper_bound_inclusive": float(edges[index + 1]),
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "absolute_gap": absolute_gap,
            }
        )
    return {
        "ece": float(ece),
        "n_bins": int(n_bins),
        "nonempty_bin_count": sum(item["count"] > 0 for item in bins),
        "bins": bins,
    }


def top_label_ece(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> dict[str, Any]:
    """Return equal-width ECE based on top-label confidence and correctness."""

    probabilities, labels = _validated_inputs(probabilities, labels)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities[np.arange(len(probabilities)), predictions]
    outcomes = (predictions == labels).astype(np.float64)
    return _equal_width_binary_calibration(confidence, outcomes, n_bins)


def adaptive_top_label_ece(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    source_row_index: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return equal-count top-label ECE with a declared deterministic tie-break."""

    probabilities, labels = _validated_inputs(probabilities, labels)
    if n_bins < 2 or n_bins > len(labels):
        raise ValueError("n_bins must be between two and the number of rows")
    if source_row_index is None:
        tie_break_values = np.arange(len(labels), dtype=np.int64)
        tie_break = "ascending input row position"
    else:
        tie_break_values = np.asarray(source_row_index, dtype=np.int64)
        if tie_break_values.shape != labels.shape:
            raise ValueError("source_row_index must contain one value per probability row")
        if len(np.unique(tie_break_values)) != len(tie_break_values):
            raise ValueError("source_row_index values must be unique")
        tie_break = "ascending source_row_index"
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities[np.arange(len(probabilities)), predictions]
    outcomes = (predictions == labels).astype(np.float64)
    order = np.lexsort((tie_break_values, confidence))
    bins: list[dict[str, Any]] = []
    ece = 0.0
    for index, selected in enumerate(np.array_split(order, n_bins)):
        count = int(len(selected))
        mean_confidence = float(confidence[selected].mean())
        accuracy = float(outcomes[selected].mean())
        absolute_gap = abs(accuracy - mean_confidence)
        ece += (count / len(confidence)) * absolute_gap
        bins.append(
            {
                "bin_index": index,
                "count": count,
                "minimum_confidence": float(confidence[selected].min()),
                "maximum_confidence": float(confidence[selected].max()),
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "absolute_gap": absolute_gap,
            }
        )
    return {
        "ece": float(ece),
        "n_bins": int(n_bins),
        "tie_break": tie_break,
        "bins": bins,
    }


def classwise_ece(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> list[dict[str, Any]]:
    """Return one-vs-rest equal-width ECE for every class probability."""

    probabilities, labels = _validated_inputs(probabilities, labels)
    result = []
    for class_index in range(probabilities.shape[1]):
        calibration = _equal_width_binary_calibration(
            probabilities[:, class_index],
            (labels == class_index).astype(np.float64),
            n_bins,
        )
        result.append(
            {
                "class_index": class_index,
                "ece": calibration["ece"],
                "n_bins": calibration["n_bins"],
                "nonempty_bin_count": calibration["nonempty_bin_count"],
                "bins": calibration["bins"],
            }
        )
    return result


def load_prediction_file(path: Path, expected_rows: int = EXPECTED_TEST_ROWS) -> dict[str, Any]:
    """Load and strictly validate one finalized prediction CSV."""

    frame = pd.read_csv(path)
    required = ["source_row_index", "true_label", "predicted_label", *PROBABILITY_COLUMNS]
    if frame.columns.tolist() != required:
        raise ValueError(f"prediction schema or class order differs: {path}")
    if len(frame) != expected_rows:
        raise ValueError(
            f"prediction row count differs: expected {expected_rows}, observed {len(frame)}"
        )
    probabilities, labels = _validated_inputs(
        frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64),
        frame["true_label"].to_numpy(dtype=np.int64),
    )
    predicted = frame["predicted_label"].to_numpy(dtype=np.int64)
    if not np.array_equal(predicted, probabilities.argmax(axis=1)):
        raise ValueError(f"predicted labels do not match probability argmax: {path}")
    source_row_index = frame["source_row_index"].to_numpy(dtype=np.int64)
    if len(np.unique(source_row_index)) != len(source_row_index):
        raise ValueError(f"source row indices are not unique: {path}")
    return {
        "source_row_index": source_row_index,
        "true_label": labels,
        "predicted_label": predicted,
        "probabilities": probabilities,
        "class_names": list(CLASS_NAMES),
    }


def _scalar_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(EXPECTED_SEEDS),) or not np.isfinite(values).all():
        raise ValueError("summary requires ten finite seed-level values")
    return {
        "values": values.tolist(),
        "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)),
        "median": float(np.median(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def paired_metric_summary(
    records: pd.DataFrame, student: str, metric: str
) -> dict[str, Any]:
    """Compare lower-is-better calibration metrics by paired training seed."""

    subset = records.loc[records["student"] == student, ["seed", "route", metric]].copy()
    if subset.duplicated(["seed", "route"]).any():
        raise ValueError("paired metric records contain duplicate seed-route rows")
    pivot = subset.pivot(index="seed", columns="route", values=metric)
    missing_routes = {"scratch", "rf_kd"} - set(pivot.columns)
    if missing_routes:
        raise ValueError(f"paired metric records are missing routes: {sorted(missing_routes)}")
    pivot = pivot.reindex(EXPECTED_SEEDS)
    if pivot.isna().any().any():
        raise ValueError("paired metric records do not contain all ten paired seeds")
    scratch = pivot["scratch"].to_numpy(dtype=np.float64)
    rf_kd = pivot["rf_kd"].to_numpy(dtype=np.float64)
    improvement = scratch - rf_kd
    return {
        "student": student,
        "metric": metric,
        "direction": "positive improvement means RF-KD is lower than scratch",
        "statistical_unit": "paired model-training seed",
        "paired_seed_count": len(EXPECTED_SEEDS),
        "seeds": list(EXPECTED_SEEDS),
        "scratch": _scalar_summary(scratch),
        "rf_kd": _scalar_summary(rf_kd),
        "improvement_values": improvement.tolist(),
        "improvement_mean": float(improvement.mean()),
        "improvement_sample_std": float(improvement.std(ddof=1)),
        "improvement_median": float(np.median(improvement)),
        "improved_seed_count": int(np.count_nonzero(improvement > 0.0)),
        "tied_seed_count": int(np.count_nonzero(improvement == 0.0)),
        "worsened_seed_count": int(np.count_nonzero(improvement < 0.0)),
        "exact_paired_wilcoxon": exact_paired_wilcoxon(improvement),
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required JSON is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def _source_manifest_index(run_root: Path) -> dict[str, dict[str, Any]]:
    manifest = _read_json(run_root / "artifact_manifest.json")
    _require(manifest.get("protocol_id") == SOURCE_PROTOCOL_ID, "source protocol differs")
    _require(manifest.get("status") == "complete", "source manifest is not complete")
    files = manifest.get("files")
    _require(isinstance(files, list) and files, "source manifest file list is empty")
    _require(
        manifest.get("file_count_excluding_manifest") == len(files),
        "source manifest file count differs",
    )
    index: dict[str, dict[str, Any]] = {}
    for item in files:
        relative = item.get("path")
        _require(isinstance(relative, str) and relative, "source manifest path is invalid")
        _require(relative not in index, f"duplicate source manifest path: {relative}")
        index[relative] = item
    return index


def _verified_source_file(
    run_root: Path, manifest_index: dict[str, dict[str, Any]], relative: str
) -> Path:
    _require(relative in manifest_index, f"source artifact is not manifested: {relative}")
    item = manifest_index[relative]
    path = run_root / relative
    _require(path.is_file(), f"source artifact is missing: {path}")
    _require(path.stat().st_size == item.get("size_bytes"), f"source size differs: {path}")
    _require(_sha256_file(path) == item.get("sha256"), f"source hash differs: {path}")
    return path


def _metric_record(
    seed: int,
    student: str,
    route: str,
    source_path: Path,
    source_sha256: str,
    loaded: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    probabilities = loaded["probabilities"]
    labels = loaded["true_label"]
    ece_results = {
        bins: top_label_ece(probabilities, labels, bins) for bins in ECE_BIN_COUNTS
    }
    adaptive = adaptive_top_label_ece(
        probabilities,
        labels,
        ADAPTIVE_ECE_BIN_COUNT,
        source_row_index=loaded["source_row_index"],
    )
    classwise = classwise_ece(probabilities, labels, 15)
    nll = negative_log_likelihood_details(probabilities, labels)
    metric = {
        "seed": seed,
        "student": student,
        "route": route,
        "prediction_source": source_path.relative_to(REPO_ROOT).as_posix(),
        "prediction_source_sha256": source_sha256,
        "test_rows": len(labels),
        "brier_multiclass": multiclass_brier_score(probabilities, labels),
        "negative_log_likelihood": nll["floored_nll"],
        "negative_log_likelihood_raw": nll["raw_nll"],
        "negative_log_likelihood_true_probability_zero_count": nll[
            "true_probability_zero_count"
        ],
        "negative_log_likelihood_probability_floor": nll["probability_floor"],
        **{f"ece_equal_width_{bins}": ece_results[bins]["ece"] for bins in ECE_BIN_COUNTS},
        "ece_adaptive_15": adaptive["ece"],
        "classwise_ece_15_macro": float(np.mean([item["ece"] for item in classwise])),
    }
    class_rows = [
        {
            "seed": seed,
            "student": student,
            "route": route,
            "class_index": item["class_index"],
            "class_name": CLASS_NAMES[item["class_index"]],
            "ece_equal_width_15": item["ece"],
            "nonempty_bin_count": item["nonempty_bin_count"],
        }
        for item in classwise
    ]
    reliability_rows = [
        {
            "seed": seed,
            "student": student,
            "route": route,
            "bin_index": item["bin_index"],
            "lower_bound_exclusive": item["lower_bound_exclusive"],
            "upper_bound_inclusive": item["upper_bound_inclusive"],
            "count": item["count"],
            "mean_confidence": item["mean_confidence"],
            "accuracy": item["accuracy"],
            "absolute_gap": item["absolute_gap"],
        }
        for item in ece_results[15]["bins"]
    ]
    return metric, class_rows, reliability_rows


def _route_aggregates(metrics: pd.DataFrame) -> dict[str, Any]:
    metric_names = [
        *PROPER_SCORE_METRICS,
        *(f"ece_equal_width_{bins}" for bins in ECE_BIN_COUNTS),
        "ece_adaptive_15",
        "classwise_ece_15_macro",
    ]
    output: dict[str, Any] = {}
    for (student, route), frame in metrics.groupby(["student", "route"], sort=True):
        frame = frame.set_index("seed").reindex(EXPECTED_SEEDS)
        _require(not frame[metric_names].isna().any().any(), "aggregate seed coverage differs")
        output.setdefault(student, {})[route] = {
            metric: _scalar_summary(frame[metric].to_numpy(dtype=np.float64))
            for metric in metric_names
        }
    return output


def _paired_tests(metrics: pd.DataFrame) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    raw_p: dict[str, float] = {}
    for student in STUDENTS:
        for metric in PROPER_SCORE_METRICS:
            name = f"{student}:{metric}:rf_kd_lower_than_scratch"
            comparison = paired_metric_summary(metrics, student, metric)
            comparisons[name] = comparison
            raw_p[name] = comparison["exact_paired_wilcoxon"]["p_value_two_sided"]
    adjusted = holm_adjust(raw_p)
    for name, comparison in comparisons.items():
        comparison["holm_family"] = "two students by two proper scoring rules"
        comparison["holm_family_size"] = len(comparisons)
        comparison["holm_adjusted_p"] = adjusted[name]
        comparison["reject_at_0_05_after_holm"] = adjusted[name] < 0.05
    ece_sensitivity: dict[str, Any] = {}
    for student in STUDENTS:
        for metric in [
            *(f"ece_equal_width_{bins}" for bins in ECE_BIN_COUNTS),
            "ece_adaptive_15",
            "classwise_ece_15_macro",
        ]:
            ece_sensitivity[f"{student}:{metric}"] = paired_metric_summary(
                metrics, student, metric
            )
    return {
        "proper_scoring_rule_family": comparisons,
        "ece_binning_sensitivity_descriptive": ece_sensitivity,
        "interpretation": (
            "The paired model-training seed is the inferential unit. The four proper-score "
            "comparisons form one Holm family. ECE variants are descriptive binning "
            "sensitivity analyses and are not additional confirmatory tests."
        ),
    }


def _reliability_summary(reliability: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, frame in reliability.groupby(["student", "route", "bin_index"], sort=True):
        student, route, bin_index = keys
        frame = frame.set_index("seed").reindex(EXPECTED_SEEDS)
        _require(not frame["count"].isna().any(), "reliability seed coverage differs")
        nonempty = frame.loc[frame["count"] > 0]
        row = {
            "student": student,
            "route": route,
            "bin_index": int(bin_index),
            "lower_bound_exclusive": float(frame["lower_bound_exclusive"].iloc[0]),
            "upper_bound_inclusive": float(frame["upper_bound_inclusive"].iloc[0]),
            "seed_count": len(frame),
            "nonempty_seed_count": len(nonempty),
            "count_mean": float(frame["count"].mean()),
            "count_sample_sd": float(frame["count"].std(ddof=1)),
            "mean_confidence_seed_mean": None,
            "mean_confidence_seed_sample_sd": None,
            "accuracy_seed_mean": None,
            "accuracy_seed_sample_sd": None,
        }
        if len(nonempty):
            row["mean_confidence_seed_mean"] = float(nonempty["mean_confidence"].mean())
            row["mean_confidence_seed_sample_sd"] = (
                float(nonempty["mean_confidence"].std(ddof=1)) if len(nonempty) > 1 else 0.0
            )
            row["accuracy_seed_mean"] = float(nonempty["accuracy"].mean())
            row["accuracy_seed_sample_sd"] = (
                float(nonempty["accuracy"].std(ddof=1)) if len(nonempty) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _artifact_manifest(output: Path) -> dict[str, Any]:
    manifest_path = output / "artifact_manifest.json"
    files = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path == manifest_path or path.name.endswith(".tmp"):
            continue
        files.append(
            {
                "path": path.relative_to(output).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "file_count_excluding_manifest": len(files),
        "files": files,
    }


def analyze_run(run_root: Path, output: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    output = output.resolve()
    _require(not output.exists(), f"refusing to overwrite existing output: {output}")
    source_index = _source_manifest_index(run_root)
    execution_path = _verified_source_file(run_root, source_index, "execution_contract.json")
    execution = _read_json(execution_path)
    _require(execution.get("protocol_id") == SOURCE_PROTOCOL_ID, "execution protocol differs")
    _require(execution.get("seeds") == EXPECTED_SEEDS, "execution seed contract differs")
    _require(execution.get("routes") == ["scratch", "rf_kd"], "execution routes differ")
    _require(
        execution.get("kd_hyperparameters") == {"T": 4.0, "alpha": 0.7},
        "recorded KD hyperparameters differ",
    )

    metric_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    reference_indices: np.ndarray | None = None
    reference_labels: np.ndarray | None = None
    used_sources: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        for student, routes in STUDENTS.items():
            for route, filename in routes.items():
                relative = f"seed_{seed}/{filename}"
                path = _verified_source_file(run_root, source_index, relative)
                source_sha256 = source_index[relative]["sha256"]
                loaded = load_prediction_file(path)
                if reference_indices is None:
                    reference_indices = loaded["source_row_index"]
                    reference_labels = loaded["true_label"]
                else:
                    _require(
                        np.array_equal(reference_indices, loaded["source_row_index"]),
                        f"source row identity differs: {path}",
                    )
                    _require(
                        np.array_equal(reference_labels, loaded["true_label"]),
                        f"true-label identity differs: {path}",
                    )
                metric, per_class, reliability = _metric_record(
                    seed, student, route, path, source_sha256, loaded
                )
                metric_rows.append(metric)
                class_rows.extend(per_class)
                reliability_rows.extend(reliability)
                used_sources.append(
                    {
                        "seed": seed,
                        "student": student,
                        "route": route,
                        "path": path.relative_to(REPO_ROOT).as_posix(),
                        "sha256": source_sha256,
                    }
                )

    metrics = pd.DataFrame(metric_rows).sort_values(["student", "route", "seed"])
    classes = pd.DataFrame(class_rows).sort_values(
        ["student", "route", "class_index", "seed"]
    )
    reliability = pd.DataFrame(reliability_rows).sort_values(
        ["student", "route", "bin_index", "seed"]
    )
    reliability_summary = _reliability_summary(reliability)
    aggregate = {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "test_rows_per_model_seed": EXPECTED_TEST_ROWS,
        "paired_model_training_seeds": EXPECTED_SEEDS,
        "class_order": list(CLASS_NAMES),
        "metric_definitions": {
            "brier_multiclass": (
                "mean over test records of the sum across classes of "
                "(predicted probability minus one-hot target)^2; range [0,2]; lower is better"
            ),
            "negative_log_likelihood": (
                "mean negative natural logarithm of the true-class probability with an "
                "explicit 1e-15 numerical floor; the ledger also records the raw value "
                "when finite and the count of zero true-class probabilities; lower is better"
            ),
            "ece_equal_width": (
                "top-label expected calibration error using equal-width confidence bins with "
                "intervals (lower, upper], except that exact zero is retained in the first bin"
            ),
            "ece_adaptive_15": (
                "top-label expected calibration error after sorting by confidence and then "
                "ascending source_row_index to break ties, followed by partitioning into "
                "15 near-equal-count bins"
            ),
            "classwise_ece_15": (
                "one-vs-rest equal-width ECE for each class probability using 15 bins"
            ),
        },
        "route_aggregates": _route_aggregates(metrics),
        "paired_comparisons": _paired_tests(metrics),
    }
    contract = {
        "protocol_id": PROTOCOL_ID,
        "analysis_type": "additive post-hoc calibration audit without model fitting",
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "source_run_root": run_root.relative_to(REPO_ROOT).as_posix(),
        "source_execution_contract_sha256": _sha256_file(execution_path),
        "analysis_script": SCRIPT_PATH.relative_to(REPO_ROOT).as_posix(),
        "analysis_script_sha256": _sha256_file(SCRIPT_PATH),
        "seeds": EXPECTED_SEEDS,
        "students": list(STUDENTS),
        "routes": ["scratch", "rf_kd"],
        "test_rows_per_model_seed": EXPECTED_TEST_ROWS,
        "ece_equal_width_bin_counts": list(ECE_BIN_COUNTS),
        "adaptive_ece_bin_count": ADAPTIVE_ECE_BIN_COUNT,
        "adaptive_ece_tie_break": "ascending source_row_index",
        "negative_log_likelihood_probability_floor": NLL_FLOOR,
        "proper_score_holm_family": [
            f"{student}:{metric}:rf_kd_lower_than_scratch"
            for student in STUDENTS
            for metric in PROPER_SCORE_METRICS
        ],
        "used_prediction_sources": used_sources,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{output.name}.", dir=output.parent))
    try:
        _atomic_write_json(temporary / "analysis_contract.json", contract)
        _atomic_write_csv(temporary / "per_seed_metrics.csv", metrics)
        _atomic_write_csv(temporary / "classwise_ece_15.csv", classes)
        _atomic_write_csv(temporary / "reliability_bins_15.csv", reliability)
        _atomic_write_csv(temporary / "reliability_summary_15.csv", reliability_summary)
        _atomic_write_json(temporary / "aggregate_results.json", aggregate)
        _atomic_write_json(temporary / "paired_tests.json", aggregate["paired_comparisons"])
        _atomic_write_json(temporary / "artifact_manifest.json", _artifact_manifest(temporary))
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return aggregate


def verify_existing(output: Path) -> dict[str, Any]:
    output = output.resolve()
    manifest = _read_json(output / "artifact_manifest.json")
    _require(manifest.get("protocol_id") == PROTOCOL_ID, "output protocol differs")
    _require(manifest.get("status") == "complete", "output status is not complete")
    files = manifest.get("files")
    _require(isinstance(files, list) and files, "output manifest file list is empty")
    _require(
        manifest.get("file_count_excluding_manifest") == len(files),
        "output manifest count differs",
    )
    for item in files:
        relative = item.get("path")
        _require(isinstance(relative, str) and relative, "output manifest path is invalid")
        path = output / relative
        _require(path.is_file(), f"output artifact is missing: {path}")
        _require(path.stat().st_size == item.get("size_bytes"), f"output size differs: {path}")
        _require(_sha256_file(path) == item.get("sha256"), f"output hash differs: {path}")
    aggregate = _read_json(output / "aggregate_results.json")
    _require(aggregate.get("protocol_id") == PROTOCOL_ID, "aggregate protocol differs")
    return {
        "status": "verified",
        "protocol_id": PROTOCOL_ID,
        "verified_artifacts": len(files),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verify_existing:
        result = verify_existing(args.output_dir)
    else:
        result = analyze_run(args.run_root, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
