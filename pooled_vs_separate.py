"""Member 3: one pooled score correction vs separate per-click corrections.

Reads the saved shared results (no inference needed) and compares, per model,
  separate: y_hat = clip(a_k q + b_k, 0, 1), fitted on calibration_fit cases at click count k
  pooled:   y_hat = clip(a q + b, 0, 1),     fitted on calibration_fit cases at all click counts
Both use the same calibration cases (same annotation budget). Extra controls:
  pooled_matched: pooled fit on one randomly drawn click count per case (200 pairs, R draws)
  pooled_13:      pooled fit on 1 and 3 clicks only (excludes 2 clicks)
  shared_slope:   y_hat = clip(a q + b_k, 0, 1), one slope for all click counts, one intercept per k
  constant:       calibration-set mean IoU at click count k
Validation cases are not used. Evaluation uses final_test when present, otherwise
internal_evaluation_overlap_risk.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiment_protocol import PROTOCOL
from bootstrap import bootstrap_indices, bootstrap_unit, dataset_id_from_rows
from member4_analysis import evaluation_role
from native_reliability import regression_metrics
from figure_plotting import plot_pooled_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-id", default="refuge2")
parser.add_argument("--run-dir", type=Path)
parser.add_argument("--matched-draws", type=int, default=200)
args = parser.parse_args()

script_dir = Path(__file__).resolve().parent
run_dir = args.run_dir if args.run_dir else script_dir / "runs" / args.dataset_id
out_dir = run_dir / "pooled_vs_separate"
out_dir.mkdir(parents=True, exist_ok=True)
MODELS, CLICKS = PROTOCOL.models, PROTOCOL.click_counts
FIT = "calibration_fit"
B, SEED = PROTOCOL.bootstrap_replicates, PROTOCOL.bootstrap_seed

with (run_dir / "shared_case_results.csv").open(newline="") as f:
    rows = list(csv.DictReader(f))
EVAL = evaluation_role(rows)
dataset_id = dataset_id_from_rows(rows)

# data[model][role][k] = (case_ids, q, y), sorted by case_id so cases align across k
data = {}
for m in MODELS:
    for role in (FIT, EVAL):
        for k in CLICKS:
            sel = sorted(
                (r for r in rows if r["model"] == m and r["analysis_role"] == role and int(r["click_count"]) == k),
                key=lambda r: r["case_id"],
            )
            if not sel:
                if role == FIT:
                    raise ValueError(f"No calibration_fit cases for {m}, click count {k}")
                raise ValueError(f"No evaluation cases for {m}, click count {k}, {role}")
            data.setdefault(m, {}).setdefault(role, {})[k] = (
                [r["case_id"] for r in sel],
                np.array([float(r["native_score"]) for r in sel]),
                np.array([float(r["observed_iou"]) for r in sel]),
            )
    for role in (FIT, EVAL):
        assert all(data[m][role][k][0] == data[m][role][CLICKS[0]][0] for k in CLICKS), "case ids differ across clicks"


def fit_line(q, y):
    a, b = np.polyfit(q, y, 1)  # ordinary least squares, y ~ a q + b
    return float(a), float(b)


rng = np.random.default_rng(SEED)
coef_rows, metric_rows, diff_rows, matched_rows, preds = [], [], [], [], {}
for m in MODELS:
    fit, ev = data[m][FIT], data[m][EVAL]
    n_fit = len(fit[CLICKS[0]][0])
    coefs = {("separate", k): fit_line(fit[k][1], fit[k][2]) for k in CLICKS}
    coefs["pooled"] = fit_line(np.concatenate([fit[k][1] for k in CLICKS]), np.concatenate([fit[k][2] for k in CLICKS]))
    coefs["pooled_13"] = fit_line(np.concatenate([fit[k][1] for k in (1, 3)]), np.concatenate([fit[k][2] for k in (1, 3)]))
    design = np.column_stack([np.concatenate([fit[k][1] for k in CLICKS]),
                              np.kron(np.eye(len(CLICKS)), np.ones((n_fit, 1)))])  # q and one dummy per click count
    theta = np.linalg.lstsq(design, np.concatenate([fit[k][2] for k in CLICKS]), rcond=None)[0]
    coefs.update({("shared_slope", k): (float(theta[0]), float(theta[1 + i])) for i, k in enumerate(CLICKS)})
    for key, (a, b) in coefs.items():
        name, k = key if isinstance(key, tuple) else (key, "all" if key == "pooled" else "1,3")
        coef_rows.append({"model": m, "method": name, "fit_click_count": k, "slope_a": a, "intercept_b": b, "fit_cases": n_fit})

    for k in CLICKS:
        q, y = ev[k][1], ev[k][2]
        line = lambda c: np.clip(c[0] * q + c[1], 0, 1)
        preds[m, k] = {
            "native": q,
            "constant": np.full_like(q, fit[k][2].mean()),
            "separate": line(coefs["separate", k]),
            "pooled": line(coefs["pooled"]),
            "pooled_13": line(coefs["pooled_13"]),
            "shared_slope": line(coefs["shared_slope", k]),
        }

    # Budget-matched pooled fit: each calibration case contributes one random click count.
    stack_q = np.stack([fit[k][1] for k in CLICKS])
    stack_y = np.stack([fit[k][2] for k in CLICKS])
    draws = {k: [] for k in CLICKS}
    for _ in range(args.matched_draws):
        pick = rng.integers(0, len(CLICKS), n_fit)
        a, b = fit_line(stack_q[pick, np.arange(n_fit)], stack_y[pick, np.arange(n_fit)])
        for k in CLICKS:
            draws[k].append(np.abs(np.clip(a * ev[k][1] + b, 0, 1) - ev[k][2]).mean())
    for k in CLICKS:
        d = np.array(draws[k])
        matched_rows.append({"model": m, "click_count": k, "draws": len(d), "mae_mean": d.mean(),
                             "mae_p2.5": np.quantile(d, 0.025), "mae_p97.5": np.quantile(d, 0.975),
                             "mae_full_pooled": np.abs(preds[m, k]["pooled"] - ev[k][2]).mean()})

    # Reuse the same patient/case draws for all paired comparisons.
    n_ev = len(ev[CLICKS[0]][0])
    idx = list(bootstrap_indices(ev[CLICKS[0]][0], dataset_id, B, SEED))
    abs_err = {(k, meth): np.abs(p - ev[k][2]) for k in CLICKS for meth, p in preds[m, k].items()}
    for k in CLICKS:
        for meth, p in preds[m, k].items():
            row = {"model": m, "click_count": k, "method": meth, "evaluation_cases": n_ev, **regression_metrics(p, ev[k][2])}
            boot_mae = [abs_err[k, meth][indices].mean() for indices in idx]
            boot_bias = [(p - ev[k][2])[indices].mean() for indices in idx]
            row["mae_ci_low"], row["mae_ci_high"] = np.quantile(boot_mae, [0.025, 0.975])
            row["bias_ci_low"], row["bias_ci_high"] = np.quantile(boot_bias, [0.025, 0.975])
            row["mean_prediction"] = row.pop("mean_native_score")
            metric_rows.append(row)
    comparisons = [("pooled", "separate"), ("pooled_13", "separate"), ("shared_slope", "separate"), ("separate", "native"), ("pooled", "native")]
    for a_meth, b_meth in comparisons:
        for k in list(CLICKS) + ["mean"]:
            ks = CLICKS if k == "mean" else (k,)
            diff = np.mean([abs_err[j, a_meth] - abs_err[j, b_meth] for j in ks], axis=0)  # per-case MAE difference
            boot = np.array([diff[indices].mean() for indices in idx])
            diff_rows.append({"model": m, "click_count": k, "comparison": f"{a_meth} - {b_meth}",
                              "mae_difference": diff.mean(), "ci_low": np.quantile(boot, 0.025),
                              "ci_high": np.quantile(boot, 0.975), "p_boot_le_0": float((boot <= 0).mean())})


def write(name, table):
    with open(out_dir / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()} for r in table)


write("coefficients.csv", coef_rows)
write("metrics.csv", metric_rows)
write("paired_differences.csv", diff_rows)
write("budget_matched_pooled.csv", matched_rows)

# LaTeX table: MAE (95% CI) and signed bias on evaluation cases.
label = {"base_sam": "SAM", "med_sa_adapter": "Med-SA"}
methods = ["native", "separate", "pooled", "shared_slope"]
get = {(r["model"], r["click_count"], r["method"]): r for r in metric_rows}
lines = [r"\begin{tabular}{llcccc}", r"\toprule",
         r"Model & $k$ & Native & Separate & Pooled & Shared slope \\", r"\midrule"]
for m in MODELS:
    for k in CLICKS:
        cells = [f"{get[m, k, meth]['mae']:.3f} ({get[m, k, meth]['bias']:+.3f})" for meth in methods]
        lines.append(f"{label[m] if k == CLICKS[0] else ''} & {k} & " + " & ".join(cells) + r" \\")
    if m != MODELS[-1]:
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
(out_dir / "table_mae_bias.tex").write_text("\n".join(lines) + "\n")

plot_pooled_metrics(metric_rows, out_dir, scale=1.5)

json.dump({"dataset_id": args.dataset_id, "fit_role": FIT, "evaluation_role": EVAL,
           "bootstrap_unit": bootstrap_unit(dataset_id),
           "bootstrap_replicates": B, "bootstrap_seed": SEED, "matched_draws": args.matched_draws,
           "coefficients": coef_rows, "paired_differences": diff_rows},
          open(out_dir / "summary.json", "w"), indent=2, default=float)
print(f"Wrote results to {out_dir}")
