import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiment_protocol import PROTOCOL
from bootstrap import bootstrap_indices, bootstrap_unit, dataset_id_from_rows
from figure_plotting import save_report_figure


MODELS = ("base_sam", "med_sa_adapter")
CLICK_COUNTS = PROTOCOL.click_counts
CASE_FIELDS = (
    "case_id",
    "dataset_id",
    "source_pool",
    "analysis_role",
    "adapter_exposure_status",
    "model",
    "click_count",
    "clicks_xy",
    "native_score_raw",
    "native_score",
    "observed_iou",
)
METRIC_FIELDS = (
    "model",
    "click_count",
    "evaluation_role",
    "evaluation_cases",
    "mean_native_score",
    "mean_observed_iou",
    "mae",
    "mae_ci_low",
    "mae_ci_high",
    "rmse",
    "bias",
    "bias_ci_low",
    "bias_ci_high",
    "coarse_ece",
)


def binary_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target masks must have the same shape.")
    union = np.logical_or(prediction, target).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(prediction, target).sum() / union)


def coarse_ece(scores: np.ndarray, targets: np.ndarray, bins: int = 10) -> float:
    scores = np.asarray(scores, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if scores.shape != targets.shape or scores.ndim != 1 or scores.size == 0:
        raise ValueError(
            "Scores and targets must be non-empty one-dimensional arrays "
            "of equal size."
        )
    if bins < 1:
        raise ValueError("bins must be at least 1.")
    if not np.isfinite(scores).all() or not np.isfinite(targets).all():
        raise ValueError("Scores and targets must be finite.")

    scores = np.clip(scores, 0.0, 1.0)
    bin_ids = np.minimum((scores * bins).astype(int), bins - 1)
    return float(
        sum(
            np.mean(bin_ids == index)
            * abs(np.mean(scores[bin_ids == index]) - np.mean(targets[bin_ids == index]))
            for index in range(bins)
            if np.any(bin_ids == index)
        )
    )


def regression_metrics(scores: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    scores = np.asarray(scores, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if scores.shape != targets.shape or scores.ndim != 1 or scores.size == 0:
        raise ValueError(
            "Scores and targets must be non-empty one-dimensional arrays "
            "of equal size."
        )
    if not np.isfinite(scores).all() or not np.isfinite(targets).all():
        raise ValueError("Scores and targets must be finite.")
    scores = np.clip(scores, 0.0, 1.0)
    errors = scores - targets
    return {
        "mean_native_score": float(scores.mean()),
        "mean_observed_iou": float(targets.mean()),
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.square(errors).mean())),
        "bias": float(errors.mean()),
        "coarse_ece": coarse_ece(scores, targets),
    }


def bootstrap_interval(
    scores: np.ndarray,
    targets: np.ndarray,
    statistic: str,
    replicates: int = 2000,
    seed: int = 1,
    case_ids: list[str] | None = None,
    dataset_id: str | None = None,
) -> tuple[float, float]:
    scores = np.asarray(scores, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if scores.shape != targets.shape or scores.ndim != 1 or scores.size == 0:
        raise ValueError(
            "Scores and targets must be non-empty one-dimensional arrays "
            "of equal size."
        )
    if statistic not in {"mae", "bias"}:
        raise ValueError("statistic must be 'mae' or 'bias'.")
    if replicates < 1:
        raise ValueError("replicates must be at least 1.")

    scores = np.clip(scores, 0.0, 1.0)
    if case_ids is None:
        case_ids = [str(index) for index in range(scores.size)]
    if len(case_ids) != scores.size:
        raise ValueError("Case IDs must match the score and target arrays.")
    errors = scores - targets
    values = np.abs(errors) if statistic == "mae" else errors
    estimates = [
        values[indices].mean()
        for indices in bootstrap_indices(case_ids, dataset_id, replicates, seed)
    ]
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def collect_case_results(
    inference_dir: str | Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    inference_dir = Path(inference_dir)
    with (inference_dir / "summary.json").open() as file:
        run_summary = json.load(file)
    if "case_roles" in run_summary:
        case_roles = run_summary["case_roles"]
        if run_summary.get("schema_version", 1) >= 2:
            declared_case_ids = [
                case_id
                for case_ids in run_summary.get("case_role_ids", {}).values()
                for case_id in case_ids
            ]
            declared_roles = {
                case_id: role
                for role, case_ids in run_summary.get("case_role_ids", {}).items()
                for case_id in case_ids
            }
            if (
                len(declared_case_ids) != len(set(declared_case_ids))
                or set(declared_roles) != set(case_roles)
                or any(
                    case_roles[case_id].get("analysis_role") != role
                    for case_id, role in declared_roles.items()
                )
            ):
                raise ValueError("case_roles and case_role_ids must agree.")
    else:
        calibration_ids = run_summary.get("calibration_case_ids", [])
        evaluation_ids = run_summary.get("evaluation_case_ids", [])
        if set(calibration_ids).intersection(evaluation_ids):
            raise ValueError("Legacy calibration and evaluation case IDs must be disjoint.")
        case_roles = {
            case_id: {
                "source_pool": run_summary.get("dataset_split", "legacy_unknown"),
                "analysis_role": "legacy_calibration",
            }
            for case_id in calibration_ids
        }
        case_roles.update(
            {
                case_id: {
                    "source_pool": run_summary.get("dataset_split", "legacy_unknown"),
                    "analysis_role": "internal_evaluation_overlap_risk",
                }
                for case_id in evaluation_ids
            }
        )
    if not case_roles or any(
        not case_id
        or not isinstance(provenance, dict)
        or not provenance.get("source_pool")
        or not provenance.get("analysis_role")
        for case_id, provenance in case_roles.items()
    ):
        raise ValueError("summary.json must provide valid case-role provenance.")
    case_ids = list(case_roles)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Case IDs must have exactly one analysis role.")

    with (inference_dir / "metadata.json").open() as file:
        metadata = json.load(file)
    dataset_id = run_summary.get(
        "dataset_id", metadata.get("protocol", {}).get("dataset_id", "legacy_unknown")
    )
    exposure_status = metadata.get("adapter_exposure_status", "unknown")

    rows = []
    predictions_dir = inference_dir / "predictions"
    for case_id in case_ids:
        prediction_path = predictions_dir / f"{case_id}.npz"
        if not prediction_path.is_file():
            raise FileNotFoundError(f"Missing prediction artifact: {prediction_path}")
        with np.load(prediction_path, allow_pickle=False) as case:
            target = case["ground_truth_disc"]
            clicks = case["clicks_xy"].tolist()
            case_provenance = case_roles[case_id]
            row_provenance = {
                "dataset_id": dataset_id,
                "source_pool": case_provenance["source_pool"],
                "analysis_role": case_provenance["analysis_role"],
                "adapter_exposure_status": exposure_status,
            }
            if run_summary.get("schema_version", 1) >= 2:
                for field, expected in row_provenance.items():
                    if field not in case.files:
                        raise ValueError(
                            f"Prediction artifact {prediction_path} is missing {field}."
                        )
                    recorded = str(case[field].item())
                    if recorded != expected:
                        raise ValueError(
                            f"Prediction artifact {prediction_path} has mismatched {field}."
                        )
            for model in MODELS:
                for click_count in CLICK_COUNTS:
                    mask = case[f"{model}_{click_count}_click_mask"]
                    raw_score = float(case[f"{model}_{click_count}_native_score_raw"])
                    rows.append(
                        {
                            "case_id": str(case_id),
                            **row_provenance,
                            "model": model,
                            "click_count": click_count,
                            "clicks_xy": json.dumps(clicks[:click_count]),
                            "native_score_raw": raw_score,
                            "native_score": float(np.clip(raw_score, 0.0, 1.0)),
                            "observed_iou": binary_iou(mask, target),
                        }
                    )
    return rows, run_summary


def summarize_conditions(
    case_rows: list[dict[str, object]],
    replicates: int = 2000,
    seed: int = 1,
    evaluation_role: str = "internal_evaluation_overlap_risk",
) -> list[dict[str, object]]:
    metrics_rows = []
    for model, click_count in (
        (model, clicks) for model in MODELS for clicks in CLICK_COUNTS
    ):
        selected = [
            row
            for row in case_rows
            if row["model"] == model and row["click_count"] == click_count
        ]
        scores = np.array([row["native_score"] for row in selected], dtype=float)
        targets = np.array([row["observed_iou"] for row in selected], dtype=float)
        metrics = regression_metrics(scores, targets)
        case_ids = [row["case_id"] for row in selected]
        dataset_id = dataset_id_from_rows(selected)
        mae_low, mae_high = bootstrap_interval(
            scores, targets, "mae", replicates, seed, case_ids, dataset_id
        )
        bias_low, bias_high = bootstrap_interval(
            scores, targets, "bias", replicates, seed, case_ids, dataset_id
        )
        metrics_rows.append(
            {
                "model": model,
                "click_count": click_count,
                "evaluation_role": evaluation_role,
                "evaluation_cases": len(selected),
                **metrics,
                "mae_ci_low": mae_low,
                "mae_ci_high": mae_high,
                "bias_ci_low": bias_low,
                "bias_ci_high": bias_high,
            }
        )
    return metrics_rows


def plot_reliability(
    case_rows: list[dict[str, object]],
    metrics_rows: list[dict[str, object]],
    path: str | Path,
    bins: int = 10,
) -> None:
    figure, axes = plt.subplots(
        len(MODELS),
        len(CLICK_COUNTS),
        figsize=(4.5 * len(CLICK_COUNTS), 8),
        sharex=True,
        sharey=True,
    )
    row_by_condition = {
        (row["model"], row["click_count"]): row for row in metrics_rows
    }
    for axis, (model, click_count) in zip(
        axes.flat, ((model, clicks) for model in MODELS for clicks in CLICK_COUNTS)
    ):
        selected = [
            row
            for row in case_rows
            if row["model"] == model and row["click_count"] == click_count
        ]
        scores = np.array([row["native_score"] for row in selected], dtype=float)
        targets = np.array([row["observed_iou"] for row in selected], dtype=float)
        bin_ids = np.minimum((scores * bins).astype(int), bins - 1)
        means = [
            (scores[bin_ids == index].mean(), targets[bin_ids == index].mean())
            for index in range(bins)
            if np.any(bin_ids == index)
        ]
        row = row_by_condition[(model, click_count)]
        axis.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
        if means:
            mean_scores, mean_targets = np.asarray(means).T
            axis.plot(mean_scores, mean_targets, "o-", color="tab:blue")
        axis.set_title(f"{model.replace('_', ' ')}\n{click_count} click(s)", pad=58)
        axis.text(
            0.5,
            1.03,
            f"MAE {row['mae']:.3f}\nbias {row['bias']:+.3f}",
            transform=axis.transAxes,
            va="bottom",
            ha="center",
        )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)
    for axis in axes[-1, :]:
        axis.set_xlabel("Mean predicted IoU")
    for axis in axes[:, 0]:
        axis.set_ylabel("Mean observed IoU")
    save_report_figure(figure, [Path(path)], dpi=180, scale=1.8)


def analyze_predictions(
    inference_dir: str | Path,
    output_dir: str | Path,
    replicates: int = 2000,
    seed: int = 1,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_case_rows, run_summary = collect_case_results(inference_dir)
    with (Path(inference_dir) / "metadata.json").open() as file:
        metadata = json.load(file)
    evaluation_role = run_summary.get(
        "evaluation_role",
        "final_test"
        if any(row["analysis_role"] == "final_test" for row in all_case_rows)
        else "internal_evaluation_overlap_risk",
    )
    case_rows = [
        row for row in all_case_rows if row["analysis_role"] == evaluation_role
    ]
    if not case_rows:
        raise ValueError(f"No prediction cases are assigned to {evaluation_role}.")
    evaluation_case_ids = list(dict.fromkeys(row["case_id"] for row in case_rows))
    metrics_rows = summarize_conditions(
        case_rows, replicates, seed, evaluation_role=evaluation_role
    )

    with (output_dir / "shared_case_results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CASE_FIELDS)
        writer.writeheader()
        writer.writerows(all_case_rows)
    with (output_dir / "native_case_results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CASE_FIELDS)
        writer.writeheader()
        writer.writerows(case_rows)
    with (output_dir / "native_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(metrics_rows)
    plot_reliability(case_rows, metrics_rows, output_dir / "native_reliability.png")

    result = {
        "dataset_id": run_summary.get("dataset_id", "legacy_unknown"),
        "dataset_split": run_summary.get("dataset_split"),
        "sample_count": run_summary.get("sample_count"),
        "calibration_count": run_summary.get("calibration_count"),
        "evaluation_role": evaluation_role,
        "evaluation_cases": len(evaluation_case_ids),
        "evaluation_case_ids": evaluation_case_ids,
        "bootstrap_unit": bootstrap_unit(run_summary.get("dataset_id")),
        "sampling_seed": run_summary.get("sampling_seed"),
        "partition_seed": run_summary.get("partition_seed"),
        "ground_truth": run_summary.get("ground_truth"),
        "gpu": run_summary.get("gpu"),
        "inference_measurements": run_summary.get("measurements"),
        "adapter_exposure_status": metadata.get("adapter_exposure_status", "unknown"),
        "adapter_exposure_evidence": metadata.get("dataset_protocol", {}).get(
            "adapter_exposure_evidence"
        ),
        "adapter_training_overlap": metadata.get("adapter_training_overlap"),
        "metrics": metrics_rows,
        "definitions": {
            "observed_iou": (
                "IoU between the saved binary mask and the majority-vote "
                "optic-disc mask."
            ),
            "bias": (
                "Mean of the clipped native score minus observed IoU. A positive "
                "value means the native score is too high on average."
            ),
            "coarse_ece": (
                "The metric uses 10 equal-width bins over [0, 1]. It reports the "
                "weighted absolute gap between mean score and mean observed IoU."
            ),
            "confidence_intervals": (
                f"We calculate 95% percentile intervals for MAE and signed bias "
                f"with {replicates} {bootstrap_unit(run_summary.get('dataset_id'))}"
                f"-level bootstrap samples and seed {seed}."
            ),
            "roles": (
                "We calculate metrics on final_test cases when available. "
                "Otherwise, we use internal_evaluation_overlap_risk cases and "
                "exclude calibration-fit cases."
            ),
        },
    }
    with (output_dir / "native_summary.json").open("w") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare native SAM and Med-SA predicted-IoU scores with observed "
            "IoU for each case role."
        )
    )
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    result = analyze_predictions(
        args.inference_dir,
        args.output_dir,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    for row in result["metrics"]:
        print(
            f"{row['model']} {row['click_count']} click(s): "
            f"MAE={row['mae']:.3f}, bias={row['bias']:+.3f}, "
            f"ECE={row['coarse_ece']:.3f}, n={row['evaluation_cases']}"
        )


if __name__ == "__main__":
    main()
