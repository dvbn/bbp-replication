"""Build paper display inputs from newly generated solver fit records."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
from numeric_summary import numeric_summary


def inverse_quantile(values, probability):
    """Inverse empirical CDF used by the historical appendix summaries."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def write_csv(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def outcome(row):
    if not row['fit_attempted']:
        return 'structural_ineligible'
    if row['accepted_1e8']:
        return 'accepted_native_converged' if row['native_converged'] else 'accepted_native_not_converged'
    if row['native_status'] == 'time_budget_exhausted':
        return 'time_budget_exhausted'
    if row.get('budget_exhausted'):
        return 'budget_limit'
    return 'native_success_audit_miss' if row['native_converged'] else 'native_failure'


def target_statistics(rows, name, truth):
    """Retain nonfinite/moment-overflow qualifications instead of clipping targets."""
    raw = [r.get(name) for r in rows]
    values = [float(x) for x in raw if x is not None and math.isfinite(x)]
    prefix = 'beta' if name == 'beta_d' else 'theta'
    summary = numeric_summary(values, truth)
    result = {prefix + '_' + key: value for key, value in summary.items()}
    result[prefix + '_arithmetic_overflow_fields'] = json.dumps(summary['arithmetic_overflow_fields'])
    result[prefix + '_missing_nonfinite'] = len(raw) - len(values)
    return result


def summarize(output, config):
    output = Path(output)
    destination = output / 'summary'
    destination.mkdir(exist_ok=True)
    pooled = defaultdict(list)
    by_dataset = defaultdict(dict)
    # Only scalar display fields are retained in memory; full vectors stay in the JSONL.
    keep = ['cell_id', 'dgp', 'group', 'n', 'k', 'draw', 'split', 'estimator', 'solver',
            'fit_attempted', 'screen_status', 'accepted_1e8', 'accepted_1e6', 'cpu_ns', 'wall_ns',
            'prospective_cap', 'common_cap_eligible', 'common_cap_accepted_1e8', 'common_cap_accepted_1e6',
            'native_status', 'native_converged', 'budget_exhausted', 'beta_d', 'theta',
            'beta_truth', 'theta_truth']
    with (output / 'fits.jsonl').open() as stream:
        for line in stream:
            record = json.loads(line)
            row = {k: record.get(k) for k in keep}
            pooled[(row['group'], row['n'], row['k'], row['estimator'], row['solver'])].append(row)
            by_dataset[(row['cell_id'], row['draw'], row['estimator'])][row['solver']] = row
    rows_outcomes, rows_main, rows_accuracy = [], [], []
    categories = ['structural_ineligible', 'accepted_native_converged', 'accepted_native_not_converged',
                  'native_success_audit_miss', 'native_failure', 'budget_limit', 'time_budget_exhausted',
                  'historical_over_cap', 'prospective_completed_over_cap']
    for (group, n, k, estimator, method), rows in sorted(pooled.items()):
        tried = [r for r in rows if r['fit_attempted']]
        accurate = sum(r['accepted_1e8'] for r in tried)
        secondary = sum(r['accepted_1e6'] for r in tried)
        common_accurate = sum(r['common_cap_accepted_1e8'] for r in tried)
        common_secondary = sum(r['common_cap_accepted_1e6'] for r in tried)
        counts = Counter(outcome(r) for r in rows)
        base = dict(variant='full', scope='pooled', group_id=f'{group}_n{n}_k{k}',
            group=group, n=n, k=k, estimator=estimator, solver=method,
            scheduled=len(rows), attempted=len(tried), structural=len(rows)-len(tried),
            accurate=accurate, raw_accurate=accurate, accepted_1e6=secondary,
            expected_schedule_records=len(rows), missing_from_schedule=0,
            failure_fraction_attempted=(1-accurate/len(tried)) if tried else None,
            native_converged=sum(bool(r['native_converged']) for r in tried),
            timeout=counts['time_budget_exhausted'],
            native_iteration_budget_exhausted=counts['budget_limit'],
            raw_outcome_counts=json.dumps(dict(counts), sort_keys=True))
        base.update({'outcome_' + key: counts[key] for key in categories})
        for clock in ['cpu', 'wall']:
            values = [r[clock + '_ns']/1e9 for r in tried]
            values_cap = [r[clock + '_ns']/1e9 for r in tried if r['common_cap_eligible']]
            for label, v in [('raw', values), ('common_cap', values_cap)]:
                prefix = label + '_' + clock + '_'
                base.update({prefix + 'n': len(v), prefix + 'missing': len(tried)-len(v),
                    prefix + 'mean': statistics.fmean(v) if v else None,
                    prefix + 'median': inverse_quantile(v, .5),
                    prefix + 'p95': inverse_quantile(v, .95),
                    prefix + 'maximum': max(v) if v else None,
                    prefix + 'total_seconds': math.fsum(v),
                    prefix + 'total_ns': round(math.fsum(v)*1e9)})
        rows_outcomes.append(dict(base, view='raw'))
        common_counts = counts.copy()
        for r in tried:
            if not r['common_cap_eligible'] and r['native_status'] != 'time_budget_exhausted':
                common_counts[outcome(r)] -= 1
                common_counts['prospective_completed_over_cap' if r['prospective_cap'] else 'historical_over_cap'] += 1
        common = dict(base, view='common_cap', accurate=common_accurate, accepted_1e6=common_secondary,
            failure_fraction_attempted=(1-common_accurate/len(tried)) if tried else None)
        common.update({'outcome_' + key: common_counts[key] for key in categories})
        rows_outcomes.append(common)
        if method in ('iols_cpp', 'statsmodels_newton', 'statsmodels_irls'):
            cpu_ms = [min(r['cpu_ns']/1e9, config['cpu_cap_seconds'])*1000 for r in tried]
            rows_main.append(dict(group=group, n=n, K=k, equation=estimator, method=method,
                scheduled=len(rows), attempted=len(tried), structural_exclusions=len(rows)-len(tried),
                strict_accepted=common_accurate, strict_failures=len(tried)-common_accurate,
                failure_percent=100*((len(tried)-common_accurate)/len(tried)) if tried else None,
                capped_cpu_median_ms=statistics.median(cpu_ms) if cpu_ms else None,
                capped_cpu_mean_ms=statistics.fmean(cpu_ms) if cpu_ms else None,
                cpu_cap_seconds=config['cpu_cap_seconds'], source_variant='full', accuracy_view='common_cap'))
        if group == 'stress' and n == 10000:
            labels = dict(iols_cpp='IOLS', iols_numpy='', statsmodels_newton='Newton',
                statsmodels_irls='IRLS', sklearn_damped_newton='Damped Newton', scipy_lbfgs='L-BFGS-B')
            rows_accuracy.append(dict(equation=estimator, method=method, n=n, K=k,
                attempted=len(tried), strict_accepted=common_accurate, secondary_accepted=common_secondary,
                strict_failures=len(tried)-common_accurate, secondary_failures=len(tried)-common_secondary,
                strict_failure_percent=100*((len(tried)-common_accurate)/len(tried)) if tried else None,
                secondary_failure_percent=100*((len(tried)-common_secondary)/len(tried)) if tried else None,
                displayed_in_current_figure=method!='iols_numpy', current_figure_label=labels[method],
                source_variant='full', accuracy_view='common_cap'))
    fields_outcomes = ['variant', 'view', 'scope', 'group_id', 'group', 'n', 'k', 'estimator', 'solver',
        'scheduled', 'attempted', 'structural', 'accurate', 'raw_accurate', 'accepted_1e6',
        'expected_schedule_records', 'missing_from_schedule', 'failure_fraction_attempted',
        'native_converged', 'timeout', 'native_iteration_budget_exhausted', 'raw_outcome_counts']
    fields_outcomes += ['outcome_'+key for key in categories]
    for label in ['raw', 'common_cap']:
        for clock in ['cpu', 'wall']:
            fields_outcomes += [label+'_'+clock+'_'+x for x in ['n','missing','mean','median','p95','maximum','total_seconds','total_ns']]
    write_csv(destination/'outcomes.csv', rows_outcomes, fields_outcomes)
    write_csv(destination/'paper_main_summary.csv', rows_main,
        ['group','n','K','equation','method','scheduled','attempted','structural_exclusions',
         'strict_accepted','strict_failures','failure_percent','capped_cpu_median_ms','capped_cpu_mean_ms',
         'cpu_cap_seconds','source_variant','accuracy_view'])
    write_csv(destination/'paper_accuracy_summary.csv', rows_accuracy,
        ['equation','method','n','K','attempted','strict_accepted','secondary_accepted','strict_failures',
         'secondary_failures','strict_failure_percent','secondary_failure_percent',
         'displayed_in_current_figure','current_figure_label','source_variant','accuracy_view'])
    # Statistical stress panels condition on each IOLS C++ root, with or without both
    # stock statsmodels methods failing the common strict gate on the same data.
    samples = defaultdict(list)
    for methods in by_dataset.values():
        cpp = methods.get('iols_cpp')
        if cpp is None or cpp['group'] != 'stress':
            continue
        key = (cpp['cell_id'], cpp['dgp'], cpp['n'], cpp['k'], cpp['estimator'])
        samples[(key, 'own_strict')]  # retain an empty group explicitly
        if cpp['estimator'] == 'gpml':
            samples[(key, 'cpp_strict_statsmodels_fail')]
        if cpp['common_cap_accepted_1e8']:
            samples[(key, 'own_strict')].append(cpp)
            sm = [methods.get(m) for m in ['statsmodels_newton','statsmodels_irls']]
            if cpp['estimator'] == 'gpml' and all(r is not None and not r['common_cap_accepted_1e8'] for r in sm):
                samples[(key, 'cpp_strict_statsmodels_fail')].append(cpp)
    rows_statistics = []
    for (key, sample), rows in sorted(samples.items()):
        cell_id, dgp, n, k, estimator = key
        record = dict(variant='full', view='common_cap', scope='cell', group='stress',
            cell_id=cell_id, group_id=cell_id, dgp=dgp, n=n, k=k, estimator=estimator,
            solver='iols_cpp', sample=sample, selected=len(rows), inaccurate_selected=0)
        record.update(target_statistics(rows, 'beta_d', math.log(1.08)))
        record.update(target_statistics(rows, 'theta', math.expm1(math.log(1.08))))
        record['finite_only_qualification'] = any(record[x+'_missing_nonfinite'] for x in ['beta','theta'])
        rows_statistics.append(record)
    fields_statistics = ['variant','view','scope','group','cell_id','group_id','dgp','n','k','estimator',
        'solver','sample','selected','inaccurate_selected','finite_only_qualification']
    for prefix in ['beta', 'theta']:
        fields_statistics += [prefix+'_'+x for x in ['n','truth','mean','bias','sd','rmse','mcse_bias',
            'minimum','median','p95','maximum','arithmetic_overflow_fields','missing_nonfinite']]
    write_csv(destination/'statistics.csv', rows_statistics, fields_statistics)
    return destination


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(Path(__file__).with_name('config.json').read_text())
    print(summarize(args.output, config))
