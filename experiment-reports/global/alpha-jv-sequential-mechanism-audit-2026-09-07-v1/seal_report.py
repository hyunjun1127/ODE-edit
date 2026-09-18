"""Local report verification/sealing only; no experiment actions or Git writes."""
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

ROOT=Path(__file__).resolve().parent
LOCAL=Path('/mnt/raid5/janghj/ODE-edit/local/state/alpha-jv-gh-mechanism-review-20260907-v2')
SOURCE=Path('/mnt/raid5/janghj/.codex/worktrees/odeedit-gh-alpha-jv-review-20260907')
EXECUTION='77358b1546d1baf83b3e251afcce663b08d7bfd7'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def rows(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def save(name,value):
    with (ROOT/name).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')


reproduced=[]
for name in ['controller_barrier_audit.csv','batch_mechanism_comparison.csv','verification.json','mechanism-evidence.png']:
    assert sha(ROOT/'derived'/name)==sha(LOCAL/'plot-recheck-v1'/name)
    reproduced.append(dict(path='derived/'+name,sha256=sha(ROOT/'derived'/name)))
for name in ['llama-spectrum-all80.csv','llama-lambda-shadow-all1040.csv','llama-sensitivity-summary.json','candidate-sensitivity-summary.csv']:
    assert sha(ROOT/'supporting'/name)==sha(LOCAL/name)
    reproduced.append(dict(path='supporting/'+name,sha256=sha(ROOT/'supporting'/name)))
allocation=SOURCE/'experiment-reports/servers/server2/alpha-native-response-v31-sequential-routing-2026-09-06-v1/main-four-terminal-v1/layer_allocation_nodes.csv'
spectra=rows(ROOT/'supporting/llama-spectrum-all80.csv')
max_eig_rel=0.
for a,s in zip(rows(allocation),spectra,strict=True):
    assert (a['alias'],a['batch'],a['node'])==(s['alias'],s['batch'],s['node'])
    h=np.array(json.loads(a['full_H']));g=np.array(json.loads(a['g']))
    eig=np.linalg.eigvalsh((h+h.T)/2)
    err=float(np.max(np.abs(eig-np.array(json.loads(s['eigenvalues']))))/max(1.,float(np.max(np.abs(eig)))))
    max_eig_rel=max(err,max_eig_rel)
    assert err<1e-12
    fraction=float(g@np.linalg.solve(h,g)/(2*float(a['V_before'])))
    assert abs(fraction-float(s['unconstrained_explainable_fraction']))<1e-10
assert len(spectra)==80
assert len(rows(ROOT/'supporting/llama-lambda-shadow-all1040.csv'))==1040
assert len(rows(ROOT/'supporting/candidate-sensitivity-summary.csv'))==52
configs=rows(ROOT/'sweep-candidates.csv')
assert len(configs)==10 and len({(x['lambda_response'],x['T'],x['N']) for x in configs})==10
for c in configs:
    assert abs(float(c['T'])-int(c['N'])*float(c['h']))<1e-12
    assert int(c['main_JVP_per_B100'])==5*int(c['N'])
    assert c['compute_z_policy']=='PINNED_OFFICIAL_UNCHANGED_ONE_PER_REQUEST_PER_BATCH'
code=[]
for rel in [
    'alpha_native_response_ode_v31_sequential/'+x for x in
    ['runtime.py','trajectory.py','state.py','telemetry.py','contracts.py']
]+['native_response_ode_v31/'+x for x in ['algebra.py','native_binding.py']]+[
    'ordered_response_barrier_ode/'+x for x in ['runtime.py','terminal_jvp.py','fp32_overlay.py','counterfact_locality_evaluator.py']]:
    path='project/run_scripts/'+rel
    original=subprocess.check_output(['git','-C',str(SOURCE),'show',EXECUTION+':'+path])
    current=(SOURCE/path).read_bytes()
    assert original==current,path
    code.append(dict(path=path,execution=EXECUTION,sha256=hashlib.sha256(current).hexdigest()))
report=ROOT/'gh-mechanism-review-ko.md'
for target in re.findall(r'\]\(([^)]+)\)',report.read_text()):
    if not target.startswith(('http:','https:','/','#')):
        assert (ROOT/target).is_file(),target
validation=dict(status='PASS_PUBLICATION_CPU_REPRODUCTION_AND_LOCAL_REPORT_LINKS',
    source_members_checked=82,controller_nodes=80,lambda_shadows=1040,candidate_summary_rows=52,
    candidate_configurations=10,core_execution_source_files_unchanged=code,
    independent_numpy_spectrum_max_relative_error=max_eig_rel,byte_reproduction=reproduced,
    source_raw_rehash='NOT_PERFORMED_SERVER2_UNAVAILABLE',new_GPU_model_Slurm_evaluation=0,
    main_push=0,execution_status='SWEEP_HOLD_CANDIDATES_ONLY')
save('validation.json',validation)
members=[]
for p in sorted(ROOT.rglob('*')):
    if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.pyc':continue
    assert not p.is_symlink()
    assert p.name not in ['analysis-manifest.json','rooted-receipt.json']
    members.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p),
                        rows=len(rows(p)) if p.suffix=='.csv' else None))
manifest=dict(status='GH_ANALYSIS_ONLY',members=members,
    members_root=hashlib.sha256(canonical(members)).hexdigest(),
    scientific_promotion=False,execution_source=EXECUTION)
save('analysis-manifest.json',manifest)
receipt=dict(status='ANALYSIS_COMPLETE_SWEEP_CANDIDATES_ONLY',
    report_sha256=sha(report),manifest_sha256=sha(ROOT/'analysis-manifest.json'),
    members_root=manifest['members_root'],member_count=len(members),
    validation_sha256=sha(ROOT/'validation.json'),scientific_promotion=False)
receipt['identity']=hashlib.sha256(canonical(receipt)).hexdigest()
save('rooted-receipt.json',receipt)
print(json.dumps(receipt,ensure_ascii=False,indent=2))
