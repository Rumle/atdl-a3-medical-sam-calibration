"""Member 2: does a score correction fitted at one click count still work at other click counts?

Protocol (decided before looking at any non-fit-click result)
--------------------------------------------------------------
Three simple, score-only corrections, all fitted on the `calibration_fit` cases:
  constant   y_hat = mean IoU of the calibration cases     shifts the average level only
  linear     y_hat = clip(a q + b, 0, 1)                   shift + slope
  isotonic   monotone non-decreasing step function         flexible, monotone
plus the native score q as the uncorrected reference.
(Boosted trees were part of an earlier version; they gave the same conclusions and were dropped
 for space.)

Steps
  1. FIT   every correction on `calibration_fit` cases at --fit-click only. Nothing is tuned.
           Validation cases are not used to select anything (all methods are reported); they are
           only reported as a sanity column.
  2. FREEZE the fitted corrections. They are never refitted after this point.
  3. TEST   each frozen correction, unchanged, on `internal_evaluation_overlap_risk` cases at
           every click count in --clicks.
  4. REFERENCE ARM (hypothesis): for each target click count k != fit click, the same method is
           also fitted on calibration cases AT click k ("own-click fit"). The transfer gap is
           MAE(fit-click correction) - MAE(own-click correction) at click k.
  5. GATE  a correction "works at the fit click" only if, on the evaluation cases at the fit click,
           the 95% paired-bootstrap CI of MAE(corrected) - MAE(native) is entirely below 0 and, for
           linear and isotonic, the CI of MAE(corrected) - MAE(constant) is also entirely below 0
           (i.e. using the score adds something beyond the average level).
           If no method passes, the honest result is negative.

Uncertainty: case-level paired bootstrap. Whole cases are resampled, with the same indices for
every method at a given model and target click count.

Evaluation cases carry the `internal_evaluation_overlap_risk` label: results are exploratory.

Run from the repository root:
    python transfer_calibration.py                       # fit at 1 click, test at 1, 2, 3
    python transfer_calibration.py --fit-click 3 --clicks 1 2 3
Outputs go to runs/<dataset-id>/transfer_calibration/fit<k>/ so runs do not overwrite each other.
Needs scikit-learn for isotonic regression (pip install scikit-learn).
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from sklearn.isotonic import IsotonicRegression
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scikit-learn is required: pip install scikit-learn") from exc

from experiment_protocol import PROTOCOL
from native_reliability import regression_metrics

FIT_ROLE = "calibration_fit"
VAL_ROLE = "validation"
EVAL_ROLE = "internal_evaluation_overlap_risk"
CORRECTIONS = ("constant", "linear", "isotonic")
REFERENCES = ("native",)
MODEL_LABELS = {"base_sam": "Original SAM", "med_sa_adapter": "Med-SA adapter"}


# --------------------------------------------------------------------------- data
def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def get(rows, model, role, k):
    sel = sorted(
        (r for r in rows if r["model"] == model and r["analysis_role"] == role and int(r["click_count"]) == k),
        key=lambda r: r["case_id"],
    )
    if not sel:
        raise SystemExit(f"No rows for {model}, {role}, {k} clicks")
    q = np.array([float(r["native_score"]) for r in sel])
    y = np.array([float(r["observed_iou"]) for r in sel])
    return q, y


# ----------------------------------------------------------------- corrections
def fit_corrections(q, y, seed):
    """Fit the constant, linear and isotonic corrections on (q, y). Returns {name: callable}, params."""
    funcs, params = {}, {}

    const = float(y.mean())
    funcs["constant"] = lambda x, c=const: np.full(np.asarray(x).shape, c)
    params["constant"] = {"mean_iou": const}

    a, b = (float(v) for v in np.polyfit(q, y, 1))
    funcs["linear"] = lambda x, a=a, b=b: np.clip(a * np.asarray(x) + b, 0.0, 1.0)
    params["linear"] = {"slope_a": a, "intercept_b": b}

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(q, y)
    funcs["isotonic"] = lambda x, m=iso: np.clip(m.predict(np.asarray(x)), 0.0, 1.0)
    params["isotonic"] = {"n_steps": int(len(iso.X_thresholds_))}

    params["fit_corr_score_iou"] = float(np.corrcoef(q, y)[0, 1])
    params["fit_cases"] = int(q.size)
    return funcs, params


# --------------------------------------------------------------------- stats
def ci(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def write_csv(path, data):
    if not data:
        return
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default=PROTOCOL.dataset_id)
    parser.add_argument("--fit-click", type=int, default=1, help="click count the correction is fitted at")
    parser.add_argument("--clicks", type=int, nargs="+", default=[1, 2, 3], help="target click counts")
    parser.add_argument("--min-bin-count", type=int, default=5, help="smallest bin drawn in the figure")
    parser.add_argument("--results-csv", default=None, help="default: runs/<dataset-id>/shared_case_results.csv")
    parser.add_argument("--out-dir", default=None, help="default: runs/<dataset-id>/transfer_calibration/fit<k>")
    args = parser.parse_args()

    run_dir = Path("runs") / args.dataset_id
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "transfer_calibration" / f"fit{args.fit_click}"
    out_dir.mkdir(parents=True, exist_ok=True)
    B, SEED = PROTOCOL.bootstrap_replicates, PROTOCOL.bootstrap_seed
    rows = load_rows(args.results_csv or run_dir / "shared_case_results.csv")
    targets = sorted(set(args.clicks) | {args.fit_click})

    # ---- steps 1-2: FIT and FREEZE (all fitting happens here, before any evaluation) ----
    frozen, own_click, fits = {}, {}, {}
    for model in PROTOCOL.models:
        q_fit, y_fit = get(rows, model, FIT_ROLE, args.fit_click)
        frozen[model], fits[model] = fit_corrections(q_fit, y_fit, SEED)
        own_click[model] = {}
        for k in targets:
            if k == args.fit_click:
                own_click[model][k] = frozen[model]
            else:
                qk, yk = get(rows, model, FIT_ROLE, k)
                own_click[model][k], fits[model][f"own_click_{k}"] = fit_corrections(qk, yk, SEED)

    # ---- step 3-5: TEST frozen corrections, reference arm, gate ----
    result_rows, comparison_rows, plot_data = [], [], {}
    for model in PROTOCOL.models:
        for k in targets:
            q, y = get(rows, model, EVAL_ROLE, k)
            qv, yv = get(rows, model, VAL_ROLE, k)
            preds = {"native": np.clip(q, 0, 1)}
            for name in CORRECTIONS:
                preds[name] = frozen[model][name](q)
            own_preds = {name: own_click[model][k][name](q) for name in CORRECTIONS}
            idx = np.random.default_rng(SEED).integers(0, y.size, size=(B, y.size))

            def boot_mae(p):
                return np.abs(p[idx] - y[idx]).mean(axis=1)

            def boot_bias(p):
                return (p[idx] - y[idx]).mean(axis=1)

            def row(method, fitted_at, p):
                m = regression_metrics(p, y)
                lo_m, hi_m = ci(boot_mae(p))
                lo_b, hi_b = ci(boot_bias(p))
                val_p = np.clip(q_of(method, fitted_at, qv), 0, 1)
                return {
                    "model": model, "fit_click": args.fit_click, "target_click": k, "method": method,
                    "fitted_at_click": fitted_at, "n_cases": int(y.size),
                    "mae": m["mae"], "mae_ci_low": lo_m, "mae_ci_high": hi_m,
                    "rmse": m["rmse"], "bias": m["bias"], "bias_ci_low": lo_b, "bias_ci_high": hi_b,
                    "coarse_ece": m["coarse_ece"], "mean_score": m["mean_native_score"],
                    "mean_observed_iou": m["mean_observed_iou"],
                    "validation_mae": float(np.abs(val_p - yv).mean()),
                }

            def q_of(method, fitted_at, scores):
                if method == "native":
                    return scores
                funcs = frozen[model] if fitted_at == args.fit_click else own_click[model][fitted_at]
                return funcs[method](scores)

            for name in REFERENCES + CORRECTIONS:
                result_rows.append(row(name, args.fit_click, preds[name]))
            if k != args.fit_click:
                for name in CORRECTIONS:
                    result_rows.append(row(name, k, own_preds[name]))

            def comp(label, method, a, b):
                diff = boot_mae(a) - boot_mae(b)
                lo, hi = ci(diff)
                comparison_rows.append({
                    "model": model, "fit_click": args.fit_click, "target_click": k, "method": method,
                    "comparison": label,
                    "difference": float(np.abs(a - y).mean() - np.abs(b - y).mean()),
                    "ci_low": lo, "ci_high": hi, "a_better": bool(hi < 0), "a_worse": bool(lo > 0),
                })
                return hi < 0

            for name in CORRECTIONS:
                beats_native = comp("MAE corrected - native", name, preds[name], preds["native"])
                # the constant cannot be compared with itself; for it the gate is "beats native" only
                beats_const = True if name == "constant" else comp("MAE corrected - constant", name, preds[name], preds["constant"])
                if k == args.fit_click:
                    comparison_rows.append({
                        "model": model, "fit_click": args.fit_click, "target_click": k, "method": name,
                        "comparison": "GATE (beats native, and constant for linear/isotonic, at fit click)",
                        "a_better": bool(beats_native and beats_const),
                    })
                else:
                    comp("transfer gap: fit-click correction - own-click correction", name, preds[name], own_preds[name])
            plot_data[(model, k)] = ({"native": preds["native"], "linear": preds["linear"], "isotonic": preds["isotonic"]}, y)

    write_csv(out_dir / "transfer_results.csv", result_rows)
    write_csv(out_dir / "transfer_comparisons.csv", comparison_rows)
    json.dump(
        {
            "protocol": "fit at fit_click on calibration_fit -> freeze -> test on evaluation cases",
            "fit_role": FIT_ROLE, "evaluation_role": EVAL_ROLE, "fit_click": args.fit_click,
            "target_clicks": targets, "bootstrap_replicates": B, "seed": SEED,
            "fits": fits,
            "note": "Evaluation cases carry the internal_evaluation_overlap_risk label; results are exploratory.",
        },
        open(out_dir / "transfer_fit.json", "w"), indent=2,
    )

    # ---- figure: rows = models, columns = target click counts ----
    fig, axes = plt.subplots(
        len(PROTOCOL.models), len(targets),
        figsize=(3.0 * len(targets), 2.9 * len(PROTOCOL.models)), squeeze=False, sharex=True, sharey=True,
    )
    styles = {"native": "o-", "linear": "s-", "isotonic": "^-"}  # constant is a single flat prediction, not drawn
    for i, model in enumerate(PROTOCOL.models):
        for j, k in enumerate(targets):
            ax = axes[i][j]
            scores, y = plot_data[(model, k)]
            ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--")
            for label, s in scores.items():
                pts = binned(s, y, 10, args.min_bin_count)
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], styles[label], ms=3.5, lw=1, label=label)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"{MODEL_LABELS[model]}, {k} click{'s' if k > 1 else ''}", fontsize=8)
            if i == len(PROTOCOL.models) - 1:
                ax.set_xlabel("Mean predicted score (bin)", fontsize=8)
            if j == 0:
                ax.set_ylabel("Mean observed IoU (bin)", fontsize=8)
            ax.tick_params(labelsize=7)
            if i == 0 and j == 0:
                ax.legend(fontsize=6, loc="upper left")
    fig.suptitle(
        f"Corrections fitted at {args.fit_click} click(s), frozen; bins with >= {args.min_bin_count} cases", fontsize=8
    )
    fig.tight_layout()
    fig.savefig(out_dir / "transfer_reliability.png", dpi=200)

    # ---- printed summary ----
    print(f"Wrote results to {out_dir}\n")
    print("fit correlation (score vs IoU):", {m: round(fits[m]["fit_corr_score_iou"], 3) for m in PROTOCOL.models})
    print("linear fits:", {m: {k: round(v, 3) for k, v in fits[m]["linear"].items()} for m in PROTOCOL.models})
    print(f"\n{'model':16s} {'target':>6s} {'method':10s} {'fit@':>4s} {'MAE':>7s} {'MAE 95% CI':>16s} {'bias':>8s}")
    for r in result_rows:
        print(
            f"{r['model']:16s} {r['target_click']:6d} {r['method']:10s} {r['fitted_at_click']:4d} "
            f"{r['mae']:7.3f} [{r['mae_ci_low']:.3f},{r['mae_ci_high']:.3f}] {r['bias']:+8.3f}"
        )
    print(f"\nGATE at fit click {args.fit_click} (beats native AND constant, paired CI below 0):")
    for r in comparison_rows:
        if r["comparison"].startswith("GATE"):
            print(f"  {r['model']:16s} {r['method']:10s}: {'PASS' if r['a_better'] else 'FAIL'}")
    print("\nTransfer gap (fit-click correction minus own-click correction, MAE; positive = transfer is worse):")
    for r in comparison_rows:
        if r["comparison"].startswith("transfer gap"):
            print(
                f"  {r['model']:16s} {r['method']:10s} target {r['target_click']}: "
                f"{r['difference']:+.3f} [{r['ci_low']:+.3f},{r['ci_high']:+.3f}]"
            )


if __name__ == "__main__":
    main()