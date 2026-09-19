"""Independent CPU NLL/sign/transition reducer and publication hash checks."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .common import *


def verify(report,reproduction,output):
    report=Path(report);reproduction=Path(reproduction)
    manifest=read(report/'report-manifest.json')
    for m in manifest['members']:
        p=report/m['path'];assert p.stat().st_size==m['bytes'] and sha256(p)==m['sha256']
        assert sha256(reproduction/m['path'])==m['sha256'],'REPRODUCTION_DRIFT:'+m['path']
    x=pd.read_parquet(ATTEMPT/'results/archival/functional_long.parquet')
    sign=np.where(x.metric_tag.eq('NS'),x.new_nll-x.true_nll,x.true_nll-x.new_nll)
    assert np.isfinite(sign).all() and np.array_equal(sign>0,x.success.to_numpy())
    final=x[x.checkpoint_batch.eq(100)].copy()
    anchor=x[x.checkpoint_batch.eq(x.arrival_batch)].copy()
    keys=['arm','metric_tag','identity']
    assert not anchor.duplicated(keys).any() and not final.duplicated(keys).any()
    j=anchor.merge(final,on=keys,suffixes=('_a','_z'),validate='one_to_one')
    assert len(j)==130000
    published=pd.read_csv(report/'paired_transitions.csv')
    published=published[(published.cohort=='at_write')&(published.view=='original_all_case')&published.end_batch.eq(100)]
    rows=[]
    for tag,g in j.groupby('metric_tag'):
        a=g.success_a.to_numpy(bool);z=g.success_z.to_numpy(bool)
        observed=dict(n11=int((a&z).sum()),n10=int((a&~z).sum()),n01=int((~a&z).sum()),n00=int((~a&~z).sum()))
        p=published[published.metric_tag.eq(tag)].iloc[0]
        assert all(int(p[k])==v for k,v in observed.items())
        assert int(p.start_success)==int(a.sum()) and int(p.end_success)==int(z.sum())
        rows.append(dict(metric=tag,denominator=len(g),at_write_success=int(a.sum()),final_success=int(z.sum()),**observed))
    final_summary=pd.read_csv(report/'functional_summary.csv')
    final_summary=final_summary[final_summary.checkpoint_batch.eq(100)&final_summary.cohort.eq('all_seen')]
    for r in rows:
        s=final_summary[final_summary.metric_tag.eq(r['metric'])].iloc[0]
        assert int(s.numerator)==r['final_success'] and int(s.denominator)==r['denominator']
    cells=pd.read_csv(report/'cell_status.csv')
    assert len(cells)==31 and cells.status.value_counts().to_dict()=={'BLOCKED':25,'PASS':5,'FAILED':1}
    assert cells.loc[cells.cell_id.eq('C01'),'status'].iloc[0]=='FAILED'
    figures=read(report/'figure-manifest.json')['figures']
    assert len(figures)==4 and all(f['regenerated_twice_byte_exact'] for f in figures)
    forbidden_extensions={'.pt','.pth','.bin','.safetensors','.parquet','.pkl','.log'}
    forbidden_columns={'prompt','target_new','target_true','generation_text','token_ids','raw_stdout','password','access_token'}
    for p in report.iterdir():
        assert p.is_file() and not p.is_symlink() and p.suffix not in forbidden_extensions
        if p.suffix=='.csv':assert not(forbidden_columns & set(pd.read_csv(p,nrows=0).columns))
    result=dict(status='PASS_CPU_POSTRUN_NOT_C01',rows_reduced_from_original_nll=len(x),final_paired_rows=len(j),
        transitions=rows,report_sha256=sha256(report/'report-ko.md'),manifest_sha256=sha256(report/'report-manifest.json'),
        full_member_rehash=len(manifest['members']),reproduction_member_match=len(manifest['members']),
        png_pdf_reproduction=True,cell_counts=cells.status.value_counts().to_dict(),raw_free_report_extensions_headers=True,
        numerical_gate_not_waived=True,checkpoint_saved=False,model_calls=0)
    write_json(output,result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--reproduction',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();verify(a.report,a.reproduction,a.output)
