"""Refit the political-centralization and nighttime-light application."""
from __future__ import annotations

import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import csv
import dataclasses
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import sysconfig
import warnings

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'vendor'))

SLOPE_FIELDS = ['sample_id','fit_id','outcome','omega','tau','shift_c','kind','variable',
    'coefficient_scale','coefficient','standard_error_cr0','ci95_lo','ci95_hi','p_value_normal',
    'significance_stars','gamma','se_gamma_cr0','n_obs','fit_status','inference_status']
COUNTRY_FIELDS = ['sample_id','fit_id','kind','country_code_wbcode','omega','tau','shift_c',
    'alpha_gamma_original_X','alpha_power_original_X','alpha_shifted_log_original_X',
    'log_a_centered_X','n_country_obs','standard_error','standard_error_status']
SAMPLE_FIELDS = ['sample_id','n_obs','n_zero_outcomes','n_countries','n_families',
    'n_intersections','n_requested_deletions','n_additional_singletons','n_removed_total']


def clean(value):
    import numpy as np
    if dataclasses.is_dataclass(value):
        return {field.name: clean(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(clean(value), allow_nan=False, sort_keys=True)+'\n')
    temporary.replace(path)


def write_csv(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(clean(rows))


def p_and_stars(coefficient, standard_error):
    from scipy.stats import norm
    if not (math.isfinite(coefficient) and math.isfinite(standard_error) and standard_error > 0):
        return None, ''
    p = float(2*norm.sf(abs(coefficient/standard_error)))
    return p, '***' if p < .01 else '**' if p < .05 else '*' if p < .1 else ''


def power_id(kind, omega, tau):
    return f'{"joint" if kind == "J" else "cancelled"}_omega{omega:g}_tau{tau:g}'


def fit_sample(plan, config):
    import numpy as np
    from core.data import CONTROLS, cohort_rows, design_rank, within_design
    from core.fits import fit_ols
    from core.inference import Z95, summarize, two_way
    from core.power_mean import fit_omega, paired_difference

    sample, metadata = plan['sample'], plan['metadata']
    sample_id = metadata['sample_id']
    if sample.audit['n_all_zero_country_groups']:
        raise ValueError(sample_id+': an all-zero country has no finite joint log intercept; no country was silently dropped')
    B, scale = within_design(sample)
    rank = design_rank(sample, B)
    if rank['within_design_rank'] != len(CONTROLS) or rank['dummy_design_rank'] != rank['dummy_design_columns']:
        raise ValueError(sample_id+': the specified within-country design is not full rank')
    country_counts = np.bincount(sample.country)
    xbar = np.stack([np.bincount(sample.country, weights=column, minlength=len(country_counts))
                    for column in sample.X.T], axis=1)/country_counts[:,None]
    coefficients, country_intercepts, covariance, paired, cross_covariance = [], [], [], [], []
    diagnostics, setups = {}, {}

    for shift in config['shifted_log_constants']:
        fit = fit_ols(sample, B, shift)
        influence_scaled = fit.hinv_solve(fit.scores.T).T
        tw = two_way(influence_scaled, sample.country, sample.family, sample.inter)
        beta, V, se, lo, hi, status = summarize(fit.theta, tw.cr0, scale, 'cr0')
        fit_id = f'ols_c{shift:g}'
        for i, variable in enumerate(CONTROLS):
            p, stars = p_and_stars(beta[i], se[i])
            coefficients.append(dict(sample_id=sample_id,fit_id=fit_id,outcome='log(Y+c)',
                shift_c=shift,kind='absorbed_OLS',variable=variable,coefficient_scale='shifted_log_OLS',
                coefficient=beta[i],standard_error_cr0=se[i],ci95_lo=lo[i],ci95_hi=hi[i],
                p_value_normal=p,significance_stars=stars,n_obs=sample.n,fit_status='ok',
                inference_status='not_applicable_closed_form' if status[i]=='ok' else status[i]))
            for j, other in enumerate(CONTROLS):
                covariance.append(dict(sample_id=sample_id,fit_id=fit_id,row_variable=variable,
                    column_variable=other,covariance_cr0=V[i,j]))
        ylog = np.log(sample.y+shift)
        intercept = np.bincount(sample.country,weights=ylog)/country_counts-xbar@beta
        for g, label in enumerate(sample.country_labels):
            country_intercepts.append(dict(sample_id=sample_id,fit_id=fit_id,kind='absorbed_OLS',
                country_code_wbcode=label,shift_c=shift,alpha_shifted_log_original_X=intercept[g],
                n_country_obs=country_counts[g],standard_error_status='not_exported'))
        diagnostics[fit_id] = dict(fit_status='ok',inference_status=status,
            covariance_cr0=V,cluster_counts={key:count for key,(_,count) in tw.meats.items()})

    rows, fits = cohort_rows(sample.country), {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', RuntimeWarning)
        for omega in config['outcome_powers']:
            for tau in config['score_indices']:
                fitted, setup = fit_omega(sample,B,scale,rows,config,float(omega),tau=float(tau))
                fits.update({(kind,float(omega),float(tau)):fit for kind,fit in fitted.items()})
                setups[f'omega{omega:g}_tau{tau:g}'] = setup
    for (kind,omega,tau), fit in fits.items():
        fit_id = power_id(kind,omega,tau)
        for i, variable in enumerate(CONTROLS):
            p, stars = p_and_stars(fit.beta[i],fit.se_power[i])
            coefficients.append(dict(sample_id=sample_id,fit_id=fit_id,outcome='Y' if omega==1 else 'sqrt(Y)',
                omega=omega,tau=tau,kind=kind,variable=variable,coefficient_scale='gamma/omega',
                coefficient=fit.beta[i],standard_error_cr0=fit.se_power[i],
                ci95_lo=fit.beta[i]-Z95*fit.se_power[i],ci95_hi=fit.beta[i]+Z95*fit.se_power[i],
                p_value_normal=p,significance_stars=stars,gamma=fit.gamma[i],se_gamma_cr0=fit.se_gamma[i],
                n_obs=sample.n,fit_status=fit.root.status,inference_status=fit.inference_status[i]))
            for j, other in enumerate(CONTROLS):
                covariance.append(dict(sample_id=sample_id,fit_id=fit_id,row_variable=variable,
                    column_variable=other,covariance_cr0=fit.covariance_power[i,j],
                    covariance_gamma_cr0=fit.covariance_gamma[i,j]))
        if kind=='J' and 'country_alpha_gamma_original_X' in fit.audit:
            for g, label in enumerate(sample.country_labels):
                country_intercepts.append(dict(sample_id=sample_id,fit_id=fit_id,kind='joint',
                    country_code_wbcode=label,omega=omega,tau=tau,
                    alpha_gamma_original_X=fit.audit['country_alpha_gamma_original_X'][g],
                    alpha_power_original_X=fit.audit['country_alpha_power_original_X'][g],
                    log_a_centered_X=fit.audit['country_log_a_centered_X'][g],
                    n_country_obs=country_counts[g],standard_error_status='not_exported'))
        diagnostics[fit_id] = fit.audit

    comparisons = [
        ('arithmetic_ppml_minus_gpml',(1.,1.),(1.,2.)),
        ('half_ppml_minus_gpml',(.5,1.),(.5,2.)),
        ('half_minus_arithmetic_ppml',(.5,1.),(1.,1.)),
        ('half_minus_arithmetic_gpml',(.5,2.),(1.,2.)),
    ]
    for kind in ['C','J']:
        for comparison, left, right in comparisons:
            difference = paired_difference(fits[(kind,*left)],fits[(kind,*right)],sample,
                config['tolerances']['inference_rel'])
            left_id, right_id = power_id(kind,*left), power_id(kind,*right)
            for i, variable in enumerate(CONTROLS):
                paired.append(dict(sample_id=sample_id,comparison=comparison,kind=kind,
                    left_fit_id=left_id,right_fit_id=right_id,variable=variable,
                    difference=difference['difference'][i],standard_error_cr0=difference['se'][i],
                    ci95_lo=difference['ci95_lo'][i],ci95_hi=difference['ci95_hi'][i],
                    z=difference['z'][i],p_value=difference['p_value'][i],
                    inference_status=difference['status'][i],variance_cr0=difference['variance'][i],
                    cross_covariance_cr0=difference['cross_covariance'][i,i],
                    step_bound_over_se=difference['step_bound_over_se'][i],n_obs=sample.n))
                for j, other in enumerate(CONTROLS):
                    cross_covariance.append(dict(sample_id=sample_id,comparison=comparison,kind=kind,
                        left_fit_id=left_id,right_fit_id=right_id,row_variable=variable,column_variable=other,
                        covariance_left_right_cr0=difference['cross_covariance'][i,j],
                        covariance_difference_cr0=difference['difference_covariance'][i,j]))
    return dict(sample={key:metadata[key] for key in SAMPLE_FIELDS},slopes=coefficients,
        country_intercepts=country_intercepts,covariance=covariance,paired_tests=paired,
        cross_covariance=cross_covariance,diagnostics=dict(sample=sample.audit,
            column_scales=dict(zip(CONTROLS,scale)),design_rank=rank,setup=setups,fits=diagnostics,
            numerical_warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in caught]))


def scientific_identity(data,config,samples):
    versions = {name:importlib.metadata.version(name) for name in ['numpy','scipy','pandas']}
    metadata = dict(data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
        python=platform.python_version(),python_abi=sysconfig.get_config_var('SOABI'),
        platform=sysconfig.get_platform(),dependency_versions=versions,samples=samples,config=config)
    digest=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode())
    sources=[ROOT/'run.py']+sorted((ROOT/'core').rglob('*.py'))+sorted((ROOT/'vendor').rglob('*.py'))
    for path in sources:
        digest.update(path.relative_to(ROOT).as_posix().encode()+b'\0'+path.read_bytes()+b'\0')
    return digest.hexdigest(),metadata


def export(output,results):
    def collect(name):
        return [row for result in results for row in result[name]]
    write_csv(output/'slopes.csv',collect('slopes'),SLOPE_FIELDS)
    write_csv(output/'country_intercepts.csv',collect('country_intercepts'),COUNTRY_FIELDS)
    write_csv(output/'samples.csv',[result['sample'] for result in results],SAMPLE_FIELDS)
    write_csv(output/'covariance.csv',collect('covariance'),
        ['sample_id','fit_id','row_variable','column_variable','covariance_cr0','covariance_gamma_cr0'])
    write_csv(output/'paired_tests.csv',collect('paired_tests'),
        ['sample_id','comparison','kind','left_fit_id','right_fit_id','variable','difference',
         'standard_error_cr0','ci95_lo','ci95_hi','z','p_value','inference_status','variance_cr0',
         'cross_covariance_cr0','step_bound_over_se','n_obs'])
    write_csv(output/'cross_covariance.csv',collect('cross_covariance'),
        ['sample_id','comparison','kind','left_fit_id','right_fit_id','row_variable','column_variable',
         'covariance_left_right_cr0','covariance_difference_cr0'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['smoke','full'],required=True)
    parser.add_argument('--data',type=Path,default=ROOT/'data/greek_econometrica_data.csv')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    if not args.data.is_file():
        parser.error('The application CSV is missing; see empirical/README.md for the data instructions')
    from core.samples import all_samples
    from threadpoolctl import threadpool_limits
    config=json.loads((ROOT/'config.json').read_text())
    plans=all_samples(args.data)
    if args.mode=='smoke':
        plans=plans[:1]
    names=[plan['metadata']['sample_id'] for plan in plans]
    identity,metadata=scientific_identity(args.data,config,names)
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    run_file=output/'run.json'
    if run_file.exists():
        if not args.resume:
            parser.error('Output already contains a run; use --resume with the same settings or a fresh directory')
        if json.loads(run_file.read_text())['scientific_id']!=identity:
            parser.error('Data, numerical source, dependencies or requested samples changed; choose a fresh directory')
    elif any(output.iterdir()):
        parser.error('Choose an empty output directory')
    write_json(run_file,dict(scientific_id=identity,**metadata))
    cache=output/'sample_results'
    cache.mkdir(exist_ok=True)
    results=[]
    with threadpool_limits(limits=1):
        for plan in plans:
            name=plan['metadata']['sample_id']
            path=cache/(name+'.json')
            if args.resume and path.exists():
                result=json.loads(path.read_text())
                if result['scientific_id']!=identity:
                    raise ValueError('Cached sample has a different scientific identity')
            else:
                result=fit_sample(plan,config)
                result['scientific_id']=identity
                write_json(path,result)
            results.append(result)
            print(f'{name}: n={plan["sample"].n}; results saved',flush=True)
    export(output,results)
    print('Application coefficients, covariances and paired tests: '+str(output),flush=True)


if __name__=='__main__':
    main()
