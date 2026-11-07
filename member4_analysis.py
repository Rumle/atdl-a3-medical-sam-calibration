"""OLS score calibration and paired click-rule comparison; no model inference."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

MODELS = ('base_sam', 'med_sa_adapter')
CLICKS = (1, 2, 3)
EVAL = 'internal_evaluation_overlap_risk'


def read_rows(path):
    with Path(path).open(newline='') as f:
        rows = list(csv.DictReader(f))
    required = {'case_id', 'analysis_role', 'model', 'click_count', 'native_score', 'observed_iou'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f'Missing required columns or empty file: {path}')
    seen, roles = set(), {}
    for r in rows:
        r['click_count'] = int(r['click_count'])
        for key in ('native_score', 'observed_iou'):
            r[key] = float(r[key])
            if not np.isfinite(r[key]) or not 0 <= r[key] <= 1:
                raise ValueError(f'Invalid {key}: {r[key]}')
        key = (r['case_id'], r['model'], r['click_count'])
        if key in seen:
            raise ValueError(f'Duplicate prediction: {key}')
        seen.add(key)
        if roles.setdefault(r['case_id'], r['analysis_role']) != r['analysis_role']:
            raise ValueError('A case appears in multiple roles')
    for case in roles:
        for model in MODELS:
            for clicks in CLICKS:
                if (case, model, clicks) not in seen:
                    raise ValueError(f'Missing condition: {case}, {model}, {clicks}')
    return rows


def select(rows, model, clicks, role):
    selected = sorted((r for r in rows if r['model'] == model and
                       r['click_count'] == clicks and r['analysis_role'] == role),
                      key=lambda r: r['case_id'])
    if not selected:
        raise ValueError(f'No cases for {model}, {clicks}, {role}')
    return ([r['case_id'] for r in selected],
            np.array([r['native_score'] for r in selected]),
            np.array([r['observed_iou'] for r in selected]))


def fit_line(q, y):
    # Fit BEFORE output clipping: explicitly ordinary least squares.
    # If scores have no variation, only a constant is identifiable.
    if np.ptp(q) < 1e-12:
        return 0.0, float(y.mean())
    a, b = np.linalg.lstsq(np.column_stack((q, np.ones(len(q)))), y, rcond=None)[0]
    return float(a), float(b)


def metrics(q, y):
    e = q - y
    bins = np.minimum((q * 10).astype(int), 9)
    ece = sum(np.mean(bins == j) * abs(e[bins == j].mean())
              for j in range(10) if np.any(bins == j))
    return dict(mae=float(abs(e).mean()), rmse=float(np.sqrt((e**2).mean())),
                bias=float(e.mean()), coarse_ece=float(ece))


def interval(values, replicates=2000, seed=1):
    if replicates < 1:
        raise ValueError('replicates must be positive')
    rng = np.random.default_rng(seed)
    estimates = [rng.choice(values, len(values), replace=True).mean()
                 for _ in range(replicates)]
    lo, hi = np.quantile(estimates, [.025, .975])
    return float(lo), float(hi)


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(rows, output, replicates=2000, seed=1):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fits, summaries, predictions, gaps = [], [], [], []
    for model in MODELS:
        functions = {}
        for fit_clicks in CLICKS:
            _, q, y = select(rows, model, fit_clicks, 'calibration_fit')
            a, b = fit_line(q, y)
            functions[fit_clicks] = (a, b, float(y.mean()))
            fits.append(dict(model=model, fit_clicks=fit_clicks, cases=len(q),
                             a=a, b=b, mean_baseline=float(y.mean())))
        for target in CLICKS:
            ids, q, y = select(rows, model, target, EVAL)
            summaries.append(dict(model=model, fit_clicks='', target_clicks=target,
                                  method='native', cases=len(q), **metrics(q, y)))
            for fit_clicks, (a, b, mean) in functions.items():
                for method, corrected in (
                    ('linear', np.clip(a*q+b, 0, 1)),
                    ('mean', np.full_like(q, mean)),
                ):
                    summaries.append(dict(model=model, fit_clicks=fit_clicks,
                                          target_clicks=target, method=method,
                                          cases=len(q), **metrics(corrected, y)))
                    if method == 'linear':
                        for case_id, raw, actual, pred in zip(ids, q, y, corrected):
                            predictions.append(dict(case_id=case_id, model=model,
                                fit_clicks=fit_clicks, target_clicks=target,
                                native_score=float(raw), observed_iou=float(actual),
                                calibrated_score=float(pred),
                                absolute_error=float(abs(pred-actual))))
            if target == 3:
                a1, b1, _ = functions[1]
                a3, b3, _ = functions[3]
                differences = abs(np.clip(a1*q+b1, 0, 1)-y)-abs(np.clip(a3*q+b3, 0, 1)-y)
                for case_id, value in zip(ids, differences):
                    gaps.append(dict(case_id=case_id, model=model, gap=float(value)))
    write_csv(output/'fitted_corrections.csv', fits)
    write_csv(output/'calibration_metrics.csv', summaries)
    write_csv(output/'calibrated_case_results.csv', predictions)
    write_csv(output/'transfer_gap_cases.csv', gaps)
    report = []
    for model in MODELS:
        values = np.array([r['gap'] for r in gaps if r['model'] == model])
        lo, hi = interval(values, replicates, seed)
        report.append(dict(model=model, mean_transfer_gap=float(values.mean()),
                           ci_low=lo, ci_high=hi, cases=len(values)))
    write_csv(output/'transfer_gap_summary.csv', report)
    (output/'analysis_settings.json').write_text(json.dumps({
        'fit_role': 'calibration_fit', 'evaluation_role': EVAL,
        'method': 'OLS on clipped native score, then clip output to [0,1]',
        'validation': 'Not used: method fixed in advance; no model selection performed',
        'mean_baseline': 'Mean actual IoU of the source click-count calibration cases',
        'bootstrap_replicates': replicates, 'bootstrap_seed': seed,
        'uncertainty': 'Paired evaluation-case bootstrap conditional on fixed fitted corrections; excludes fitting uncertainty',
        'limitation': 'Independence from checkpoint development is unverified',
    }, indent=2)+'\n')
    return gaps, predictions


def compare(original, alternative, output, replicates, seed):
    # Strict matching prevents accidental comparisons across different splits.
    def signature(rows):
        return {(r['case_id'], r['analysis_role'], r['model'], r['click_count']) for r in rows}
    if signature(original) != signature(alternative):
        raise ValueError('The two inputs must have identical cases, roles and conditions')
    output = Path(output)
    ga, pa = analyze(original, output/'original', replicates, seed)
    gb, pb = analyze(alternative, output/'alternative', replicates, seed)
    lookup = {(r['case_id'], r['model']): r['gap'] for r in ga}
    reports = []
    for model in MODELS:
        values = np.array([r['gap']-lookup[(r['case_id'], model)] for r in gb if r['model'] == model])
        lo, hi = interval(values, replicates, seed)
        reports.append(dict(model=model, alternative_minus_original_gap=float(values.mean()),
                            ci_low=lo, ci_high=hi, cases=len(values)))
    write_csv(output/'paired_rule_comparison.csv', reports)
    # Predefined descriptive ranking: largest change in transferred-score error.
    lookup = {(r['case_id'], r['model']): r for r in pa
              if r['fit_clicks'] == 1 and r['target_clicks'] == 3}
    candidates = []
    for r in pb:
        if r['fit_clicks'] == 1 and r['target_clicks'] == 3:
            old = lookup[(r['case_id'], r['model'])]
            candidates.append(dict(case_id=r['case_id'], model=r['model'],
                original_error=old['absolute_error'], alternative_error=r['absolute_error'],
                error_change=r['absolute_error']-old['absolute_error']))
    candidates.sort(key=lambda r: abs(r['error_change']), reverse=True)
    write_csv(output/'failure_example_candidates.csv', candidates)
    print(json.dumps(reports, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alternative', type=Path, required=True,
                        help='shared_case_results.csv, including calibration cases')
    parser.add_argument('--original', type=Path, help='Baseline shared_case_results.csv')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--bootstrap-replicates', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    alternate = read_rows(args.alternative)
    if args.original:
        compare(read_rows(args.original), alternate, args.output_dir,
                args.bootstrap_replicates, args.seed)
    else:
        analyze(alternate, args.output_dir, args.bootstrap_replicates, args.seed)


if __name__ == '__main__':
    main()
