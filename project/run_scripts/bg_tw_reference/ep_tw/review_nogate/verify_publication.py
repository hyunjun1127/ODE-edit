"""Postreview arithmetic, reproduction and raw-free package checks; CPU only."""
import argparse
import ast
import csv
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from review_nogate import ROOT,read,ref,save,sha

def run(worktree,package):
    w,p=Path(worktree),Path(package);local=ROOT/'completed-review-v1'
    def rows(n):return list(csv.DictReader((p/n).open()))
    metrics=list((local/'metrics-v3').glob('*'))
    assert len(metrics)==12
    assert all(sha(f)==sha(local/'metrics-reproduction-v1'/f.name) for f in metrics)
    figs=list((p/'figures').glob('*.png'));assert len(figs)==6
    assert all(sha(f)==sha(local/'figures-reproduction-v1'/f.name) for f in figs)
    expected=[('RS',998,1000),('PS',1942,2000),('NS',8056,10000)]
    actual=[(r['metric'],int(r['numerator']),int(r['denominator'])) for r in rows('first-final-table.csv')]
    assert actual==expected
    for r in rows('paired-transitions.csv')+rows('baseline-paired.csv')+rows('cohort-retention.csv'):
        assert int(r['before_num'])-int(r['lost'])+int(r['gained'])==int(r['after_num'])
        assert sum(int(r[k]) for k in ('retained','lost','gained','both_failed'))==int(r['denominator'])
        assert int(r['conditional_loss_den'])+int(r['conditional_recovery_den'])==int(r['denominator'])
    expected_lengths={'candidate-details.csv':40,'batch-policy.csv':10,'per-batch-mechanism.csv':10,
                      'checkpoint-inventory.csv':10,'state-links.csv':9,'generic-reduction.csv':42,
                      'baseline-first1000.csv':18,'design-conformance.csv':15}
    for filename,count in expected_lengths.items():assert len(rows(filename))==count
    raw=rows('raw-member-inventory.csv')
    assert len(raw)==116 and sum(int(x['bytes']) for x in raw)==13407776774
    # Reuse prior full hashes; current stat consistency is not a new full rehash.
    for x in raw:
        s=Path(x['path']).stat()
        assert (s.st_ino,s.st_size,s.st_mtime_ns)==(int(x['inode']),int(x['bytes']),int(x['mtime_ns']))
    report=(p/'diagnostic-report-ko.md').read_text()
    for i in range(1,10):assert f'### 4.{i} ' in report
    for text in ('NOT_ESTABLISHED','SKIPPED_USER_DIRECTED','SOURCE_CONFIRMED','STORED_EVIDENCE_CONSISTENT','B1 (C1)','B5 (RAW)'):
        assert text in report
    for name in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',report):assert (p/name).is_file()
    code_root=w/'project/run_scripts/bg_tw_reference/ep_tw'
    code=[code_root/'review_nogate.py',code_root/'plot_nogate_review.py',code_root/'test_review_nogate.py',
          *sorted((code_root/'review_nogate').glob('*.py'))]
    for f in code:ast.parse(f.read_text(),filename=str(f))
    result=subprocess.run([sys.executable,'-B','-m','unittest','project.run_scripts.bg_tw_reference.ep_tw.test_review_nogate'],
                          cwd=w,capture_output=True,text=True)
    assert result.returncode==0 and 'Ran 10 tests' in result.stderr
    forbidden={'.pt','.npz','.npy','.bin','.safetensors','.pkl','.gz','.tar','.log'}
    bad_headers={'prompt','target','subject','raw_text','input_ids','new_target','true_target'}
    for f in p.rglob('*'):
        if not f.is_file():continue
        assert f.suffix not in forbidden
        if f.suffix=='.csv':
            header=next(csv.reader(f.open()))
            assert not (set(header)&bad_headers)
        if f.suffix in ('.json','.csv','.md'):
            text=f.read_text()
            assert not re.search(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{30,}',text)
    save(p/'verification-receipt.json',dict(status='CPU_ANALYSIS_PACKAGE_CHECKS_PASS_NOT_MODEL_VALIDATION',
        reducer_fixture_tests=10,syntax_files=len(code),metrics_byte_reproduction_files=12,PNG_byte_reproduction=6,
        plot_visual_inspection=['mechanism-actions.png','candidate-screen.png','baseline-reference.png'],
        table_cardinality=expected_lengths,paired_integer_conservation=True,canonical_final=actual,
        raw_new_fullsha_evidence=ref(local/'state-v1/raw-member-inventory.csv'),current_raw_stat_unchanged=True,
        shared_heavy_rehash=False,scope='OWN_NEW_ANALYSIS_CODE_REPORT_ONLY',raw_payload_Git=0,
        source_code=[dict(ref(f),relative_path=str(f.relative_to(w))) for f in code],
        model_load=0,new_GPU=0,Slurm_mutations=0,FD_revalidation=0,new_evaluations=0,
        skipped_validation='SKIPPED_USER_DIRECTED',numerical_validation='NOT_ESTABLISHED',
        review_limitations=['no independent GPU gradient/parity/replay','no purewriter timing','no baseline C4 evaluation',
                           'single order/seed; no causal interpretation','source/CPU consistency not efficacy validation']))
    print(json.dumps({'verification':'PASS','numerical_validation':'NOT_ESTABLISHED','report_sha256':sha(p/'diagnostic-report-ko.md')}))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--worktree',required=True);a.add_argument('--package',required=True)
    x=a.parse_args();run(x.worktree,x.package)
