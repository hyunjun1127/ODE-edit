"""Publication arithmetic/denominator/hash checks, never experiment gating."""
import argparse
import json
import numpy as np
import pandas as pd
from .common import *


def verify(report,output):
    report=Path(report);manifest=read(report/'report-manifest.json')
    for m in manifest['members']:
        p=report/m['path'];assert p.stat().st_size==m['bytes'] and sha256(p)==m['sha256']
    ctx=read(report/'context.json')
    assert ctx['diagnostic_gates_enabled'] is False and ctx['numerical_validation']=='NOT_ESTABLISHED'
    assert ctx['numerical_parity']['historical_C01_status']=='FAILED'
    assert ctx['numerical_parity']['new_C00_C01_execution']=='SKIPPED_USER_DIRECTED'
    cells=pd.read_csv(report/'cell_status.csv');assert len(cells)==31 and cells.cell_id.nunique()==31
    assert cells.loc[cells.cell_id.eq('C01'),'status'].item()=='FAILED'
    old=REPO/'experiment-reports/servers/server2/checkpoint-mechanism-audit-2026-09-20-v1'
    for name in ('functional_summary.csv','paired_transitions.csv','paired_bootstrap.csv','at_write_outcomes.csv','checkpoint_geometry.csv'):
        pd.testing.assert_frame_equal(pd.read_csv(report/name),pd.read_csv(old/name),check_exact=True)
    counts={}
    p=report/'fixed_probe_history.csv'
    if p.exists():
        f=pd.read_csv(p);assert not f.duplicated(['history_batch','case_id']).any()
        for h,g in f.groupby('history_batch'):
            assert len(g)==512
        counts['fixed_probe_rows']=len(f)
    m=pd.read_csv(report/'native_write_modes.csv');r=pd.read_csv(report/'reconstruction.csv')
    if 'mode' in m:
        assert not m.duplicated(['target_batch','mode']).any()
        for batch,g in m.groupby('target_batch'):
            row=r[r.target_batch.eq(batch)].iloc[0];decomp=json.loads(row.decomposition)
            assert len(g)==100
            assert np.isclose(g.write_energy.sum(),decomp['mode_energy_sum'],rtol=2e-15,atol=0)
        counts['native_modes']=len(m)
    c=pd.read_csv(report/'counterfactuals.csv')
    if 'permutation_index' in c:
        assert not c.duplicated(['history_batch','target_batch','permutation_index']).any()
        for _,g in c.groupby(['history_batch','target_batch']):assert set(g.permutation_index)==set(range(-1,20))
        counts['counterfactual_rows']=len(c)
    a=pd.read_csv(report/'activation_margin.csv')
    if 'identity' in a:
        assert not a.duplicated(['interval_start','interval_end','identity']).any()
        assert np.allclose(a.endpoint_margin-a.entry_margin,a.actual_margin_change,rtol=1e-14,atol=1e-14)
        assert np.allclose(a.actual_margin_change-a.predicted_margin_change,a.nonlinear_remainder,rtol=1e-14,atol=1e-14)
        counts['activation_rows']=len(a)
    figures=read(report/'figure-manifest.json')['figures']
    assert len(figures)==4 and all(f['regenerated_twice_byte_exact'] for f in figures)
    banned={'.pt','.pth','.bin','.safetensors','.parquet','.pkl','.log'}
    columns={'prompt','target_new','target_true','token_ids','generation_text','raw_stdout','access_token','password','new_nll','true_nll'}
    for p in report.iterdir():
        assert p.is_file() and not p.is_symlink() and p.suffix not in banned
        if p.suffix=='.csv':assert not columns.intersection(pd.read_csv(p,nrows=0).columns)
    index=pd.read_csv(report/'artifact-index.csv');assert set(index.required_output)==set(CONTRACT['required_outputs'])
    result=dict(status='PUBLICATION_ARITHMETIC_AND_HASH_CHECKED',report_sha256=sha256(report/'report-ko.md'),
        members_rehashed=len(manifest['members']),counts=counts,cells=cells.status.value_counts().to_dict(),
        old_scientific_tables_equal=True,figures_regenerated_byte_exact=True,raw_free=True,
        model_gpu_calls=0,new_numerical_experiment_gate=False,**EXECUTION_POLICY)
    write_json(output,result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();verify(a.report,a.output)
