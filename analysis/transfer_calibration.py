"""Member 2: does a score correction fitted at one click count still work at other click counts?

Protocol (decided before looking at any non-fit-click result)
--------------------------------------------------------------
Three simple, score-only corrections, all fitted on the `calibration_fit` cases:
  constant   y_hat = mean IoU of the calibration cases     shifts the average level only
  linear     y_hat = clip(a q + b, 0, 1)                   shift + slope
  isotonic   monotone non-decreasing interpolated function flexible, monotone
plus the native score q as the uncorrected reference.

Steps
  1. FIT   every correction on `calibration_fit` cases at --fit-click only. Nothing is tuned.
  2. FREEZE the fitted corrections. They are never refitted after this point.
  3. TEST   each frozen correction, unchanged, on the dataset's evaluation cases at
           every click count in --clicks.
  4. REFERENCE ARM (hypothesis): for each target click count k != fit click, the same method is
           also fitted on calibration cases AT click k ("own-click fit"). The transfer gap is
           MAE(fit-click correction) - MAE(own-click correction) at click k.
  5. GATE  a correction "works at the fit click" only if, on the evaluation cases at the fit click,
           the 95% paired-bootstrap CI of MAE(corrected) - MAE(native) is entirely below 0 and, for
           linear and isotonic, the CI of MAE(corrected) - MAE(constant) is also entirely below 0
           (i.e. using the score adds something beyond the average level).
           If no method passes, the honest result is negative.

Uncertainty: paired bootstrap over patients for PAPILA and cases otherwise. The same draws
are used for every method at a given model and target click count.

REFUGE evaluation may overlap checkpoint development. Other datasets have unknown exposure.
PAPILA confidence intervals resample patients and include both eyes.

Run from the repository root:
    python -m analysis.transfer_calibration
    python -m analysis.transfer_calibration --fit-click 3 --clicks 1 2 3
Outputs go to runs/<dataset-id>/transfer_calibration/fit<k>/ so runs do not overwrite each other.
Scikit-learn is required for isotonic regression and is installed from requirements.txt.
"""

import argparse
import csv
import json
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sklearn.isotonic import IsotonicRegression

from analysis.bootstrap import bootstrap_indices, bootstrap_unit, dataset_id_from_rows
from experiment_protocol import DATASET_PROTOCOLS, PROTOCOL
from analysis.member4_analysis import evaluation_role, fit_line, read_rows, select
from analysis.native_reliability import regression_metrics
from analysis.figure_plotting import save_report_figure

FIT_ROLE = "calibration_fit"
CORRECTIONS = ("constant", "linear", "isotonic")
MODEL_LABELS = {"base_sam": "Original SAM", "med_sa_adapter": "Med-SA adapter"}


def fit_corrections(q, y):
    """Fit the constant, linear and isotonic corrections on (q, y). Returns {name: callable}, params."""
    funcs, params = {}, {}

    const = float(y.mean())
    funcs["constant"] = lambda x, c=const: np.full(np.asarray(x).shape, c)
    params["constant"] = {"mean_iou": const}

    a, b = fit_line(q, y)
    funcs["linear"] = lambda x, a=a, b=b: np.clip(a * np.asarray(x) + b, 0.0, 1.0)
    params["linear"] = {"slope_a": a, "intercept_b": b}

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(q, y)
    funcs["isotonic"] = lambda x, m=iso: np.clip(m.predict(np.asarray(x)), 0.0, 1.0)
    params["isotonic"] = {
        "n_steps": int(len(iso.X_thresholds_)),
        "score_thresholds": iso.X_thresholds_.tolist(),
        "iou_thresholds": iso.y_thresholds_.tolist(),
    }

    params["fit_corr_score_iou"] = (
        float(np.corrcoef(q, y)[0, 1]) if np.ptp(q) > 0 and np.ptp(y) > 0 else None
    )
    params["fit_cases"] = int(q.size)
    return funcs, params


def bootstrap_errors(prediction, observed, samples):
    errors = prediction - observed
    mae = np.array([np.abs(errors[indices]).mean() for indices in samples])
    bias = np.array([errors[indices].mean() for indices in samples])
    return mae, bias


def ci(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def write_csv(path, data):
    if not data:
        raise ValueError(f"No results to write: {path}")
    fields = list(dict.fromkeys(k for row in data for k in row))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(data)


def binned(q, y, bins, min_count):
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(q, edges[1:-1]), 0, bins - 1)
    pts = []
    for b in range(bins):
        m = which == b
        if m.sum() >= min_count:
            pts.append((q[m].mean(), y[m].mean()))
    return pts


def fit_models(rows, fit_click, targets):
    frozen, own_click, fits = {}, {}, {}
    for model in PROTOCOL.models:
        _, q, y = select(rows, model, fit_click, FIT_ROLE)
        frozen[model], fits[model] = fit_corrections(q, y)
        own_click[model] = {fit_click: frozen[model]}
        for k in targets:
            if k != fit_click:
                _, qk, yk = select(rows, model, k, FIT_ROLE)
                own_click[model][k], fits[model][f"own_click_{k}"] = fit_corrections(qk, yk)
    return frozen, own_click, fits


def compare_corrections(preds, own_preds, y, samples, model, target, fit_click):
    rows = []
    estimates = {name: bootstrap_errors(p, y, samples)[0] for name, p in preds.items()}

    def compare(label, method, first, second, first_estimates, second_estimates):
        lo, hi = ci(first_estimates - second_estimates)
        rows.append({
            "model": model, "fit_click": fit_click, "target_click": target, "method": method,
            "comparison": label,
            "difference": float(np.abs(first - y).mean() - np.abs(second - y).mean()),
            "ci_low": lo, "ci_high": hi, "a_better": bool(hi < 0), "a_worse": bool(lo > 0),
        })
        return hi < 0

    for name in CORRECTIONS:
        beats_native = compare("MAE corrected - native", name, preds[name], preds["native"],
                               estimates[name], estimates["native"])
        beats_constant = True
        if name != "constant":
            beats_constant = compare(
                "MAE corrected - constant", name, preds[name], preds["constant"],
                estimates[name], estimates["constant"],
            )
        if target == fit_click:
            rows.append({
                "model": model, "fit_click": fit_click, "target_click": target, "method": name,
                "comparison": "GATE (beats native, and constant for linear/isotonic, at fit click)",
                "a_better": bool(beats_native and beats_constant),
            })
        else:
            compare(
                "transfer gap: fit-click correction - own-click correction", name,
                preds[name], own_preds[name], estimates[name],
                bootstrap_errors(own_preds[name], y, samples)[0],
            )
    return rows


def evaluate_condition(rows, model, target, args, frozen, own_click):
    case_ids, q, y = select(rows, model, target, evaluation_role(rows))
    samples = list(bootstrap_indices(
        case_ids, args.dataset_id, args.bootstrap_replicates, PROTOCOL.bootstrap_seed
    ))
    preds = {"native": q, **{name: frozen[name](q) for name in CORRECTIONS}}
    own_preds = {name: own_click[name](q) for name in CORRECTIONS}

    def metric_row(method, fitted_at, prediction):
        metrics = regression_metrics(prediction, y)
        mae, bias = bootstrap_errors(prediction, y, samples)
        lo_m, hi_m = ci(mae)
        lo_b, hi_b = ci(bias)
        return {
            "model": model, "fit_click": args.fit_click, "target_click": target, "method": method,
            "fitted_at_click": fitted_at, "n_cases": int(y.size),
            "mae": metrics["mae"], "mae_ci_low": lo_m, "mae_ci_high": hi_m,
            "rmse": metrics["rmse"], "bias": metrics["bias"],
            "bias_ci_low": lo_b, "bias_ci_high": hi_b,
            "coarse_ece": metrics["coarse_ece"], "mean_score": metrics["mean_native_score"],
            "mean_observed_iou": metrics["mean_observed_iou"],
        }

    results = [metric_row("native", args.fit_click, q)]
    for name in CORRECTIONS:
        results.append(metric_row(name, args.fit_click, preds[name]))
    if target != args.fit_click:
        for name in CORRECTIONS:
            results.append(metric_row(name, target, own_preds[name]))
    comparisons = compare_corrections(
        preds, own_preds, y, samples, model, target, args.fit_click
    )
    return results, comparisons, (
        {name: preds[name] for name in ("native", "linear", "isotonic")}, y
    )


def plot_reliability(plot_data, targets, args, out_dir):
    fig, axes = plt.subplots(
        len(PROTOCOL.models), len(targets),
        figsize=(3.0 * len(targets), 2.9 * len(PROTOCOL.models)),
        squeeze=False, sharex=True, sharey=True,
    )
    styles = {"native": "o-", "linear": "s-", "isotonic": "^-"}
    for i, model in enumerate(PROTOCOL.models):
        for j, k in enumerate(targets):
            ax = axes[i][j]
            scores, y = plot_data[(model, k)]
            ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--")
            for label, score in scores.items():
                pts = binned(score, y, 10, args.min_bin_count)
                if pts:
                    ax.plot(*zip(*pts), styles[label], ms=3.5, lw=1, label=label)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"{MODEL_LABELS[model]}\n{k} click(s)", fontsize=8)
            if i == len(PROTOCOL.models) - 1:
                ax.set_xlabel("Mean predicted\nscore (bin)", fontsize=8)
            if j == 0:
                ax.set_ylabel("Mean observed\nIoU (bin)", fontsize=8)
            ax.tick_params(labelsize=7)
    axes[0][0].legend(fontsize=6, loc="upper left")
    fig.suptitle(
        f"Corrections fitted at {args.fit_click} click(s), frozen\n"
        f"bins with >= {args.min_bin_count} cases",
        fontsize=8,
    )
    save_report_figure(fig, [out_dir / "transfer_reliability.png"], dpi=200, scale=1.8)


def print_summary(out_dir, result_rows, comparison_rows):
    print(f"Wrote results to {out_dir}\n")
    print(f"{'model':16s} {'target':>6s} {'method':10s} {'fit@':>4s} {'MAE':>7s} {'MAE 95% CI':>16s} {'bias':>8s}")
    for row in result_rows:
        print(
            f"{row['model']:16s} {row['target_click']:6d} {row['method']:10s} {row['fitted_at_click']:4d} "
            f"{row['mae']:7.3f} [{row['mae_ci_low']:.3f},{row['mae_ci_high']:.3f}] {row['bias']:+8.3f}"
        )
    print("\nGate at fit click, paired CI must beat native and constant:")
    for row in comparison_rows:
        if row["comparison"].startswith("GATE"):
            print(f"  {row['model']:16s} {row['method']:10s}: {'PASS' if row['a_better'] else 'FAIL'}")
    print("\nTransfer gap, MAE of transferred correction minus own-click correction:")
    for row in comparison_rows:
        if row["comparison"].startswith("transfer gap"):
            print(
                f"  {row['model']:16s} {row['method']:10s} target {row['target_click']}: "
                f"{row['difference']:+.3f} [{row['ci_low']:+.3f},{row['ci_high']:+.3f}]"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", choices=DATASET_PROTOCOLS, default=PROTOCOL.dataset_id)
    parser.add_argument("--fit-click", type=int, choices=PROTOCOL.click_counts, default=1,
                        help="click count the correction is fitted at")
    parser.add_argument("--clicks", type=int, nargs="+", choices=PROTOCOL.click_counts,
                        default=list(PROTOCOL.click_counts), help="target click counts")
    parser.add_argument("--min-bin-count", type=int, default=5, help="smallest bin drawn in the figure")
    parser.add_argument("--results-csv", default=None, help="default: runs/<dataset-id>/shared_case_results.csv")
    parser.add_argument("--out-dir", default=None, help="default: runs/<dataset-id>/transfer_calibration/fit<k>")
    parser.add_argument("--bootstrap-replicates", type=int, default=PROTOCOL.bootstrap_replicates)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1 or args.min_bin_count < 1:
        parser.error("bootstrap-replicates and min-bin-count must be positive")

    run_dir = Path(__file__).resolve().parents[1] / "runs" / args.dataset_id
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "transfer_calibration" / f"fit{args.fit_click}"
    B, SEED = args.bootstrap_replicates, PROTOCOL.bootstrap_seed
    rows = read_rows(args.results_csv or run_dir / "shared_case_results.csv")
    dataset_id = dataset_id_from_rows(rows) or args.dataset_id
    if dataset_id != args.dataset_id:
        raise ValueError(f"Input dataset {dataset_id} does not match --dataset-id {args.dataset_id}")
    eval_role = evaluation_role(rows)
    targets = sorted(set(args.clicks) | {args.fit_click})

    # Fit every arm on calibration cases before evaluating any correction.
    frozen, own_click, fits = fit_models(rows, args.fit_click, targets)
    result_rows, comparison_rows, plot_data = [], [], {}
    for model in PROTOCOL.models:
        for k in targets:
            results, comparisons, plot_data[(model, k)] = evaluate_condition(
                rows, model, k, args, frozen[model], own_click[model][k]
            )
            result_rows.extend(results)
            comparison_rows.extend(comparisons)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "transfer_results.csv", result_rows)
    write_csv(out_dir / "transfer_comparisons.csv", comparison_rows)
    with (out_dir / "transfer_fit.json").open("w") as file:
        json.dump(
            {
                "protocol": "fit at fit_click on calibration_fit -> freeze -> test on evaluation cases",
                "dataset_id": dataset_id,
                "fit_role": FIT_ROLE, "evaluation_role": eval_role, "fit_click": args.fit_click,
                "target_clicks": targets, "bootstrap_replicates": B, "seed": SEED,
                "bootstrap_unit": bootstrap_unit(dataset_id),
                "evaluation_case_ids": select(rows, PROTOCOL.models[0], args.fit_click, eval_role)[0],
                "uncertainty": "Bootstrap holds fitted corrections fixed.",
                "scikit_learn_version": version("scikit-learn"),
                "isotonic_interpolation": "Linear between thresholds; clip at endpoints.",
                "fits": fits,
                "note": DATASET_PROTOCOLS[dataset_id].adapter_exposure_evidence,
            },
            file, indent=2, allow_nan=False,
        )
    plot_reliability(plot_data, targets, args, out_dir)
    print_summary(out_dir, result_rows, comparison_rows)


if __name__ == "__main__":
    main()