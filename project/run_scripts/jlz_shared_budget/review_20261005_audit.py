"""Bounded CPU receipt/source audit for the frozen v12-MEMIT run; no model/scheduler calls."""
import argparse,csv,hashlib,json,unicodedata,subprocess,sys,datetime
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
root=Path.cwd();p=args.out
a=Path('/data/janghj/ODE-edit/local/jlz-v12-shared-budget/20261004-v1/attempt-s3-r1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def entry(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
config=json.loads((a/'config.json').read_text());records=json.loads(Path(config['stream']).read_text())[:2000]
future={};active={}
for r in reversed(records):
 w=r['requested_rewrite'];key=(unicodedata.normalize('NFC',' '.join(w['subject'].split())),w['relation_id']);target=w['target_new'].get('id',w['target_new']['str'])
 active[r['case_id']]=not(future.get(key,set())-{target});future.setdefault(key,set()).add(target)
rows=[r for f in sorted((a/'main-V12_MAIN/batch-20/post').glob('chunk-*.json')) for r in json.loads(f.read_text())['rows']]
assert len(rows)==26000 and all(r['active_at_endpoint']==active[r['case_id']] for r in rows)
assert sum(active.values())==1984
qs=sorted((a/'pilot-V12_MAIN/batch-01/qualification').glob('*.json'))
files=qs+[a/x for x in ['pilot-V12_MAIN/runtime.json','main-V12_MAIN/runtime.json','pilot-V12_MAIN/rollback-probe.json','main-V12_MAIN/initial.json','pilot-V12_MAIN/initial-state.json','main-V12_MAIN/initial-state.json','config.json','execution.lock.json','report/terminal.json']]
source=[a/'source/project/run_scripts/jlz_shared_budget'/x for x in ['optimize.py','optimizer.py','writer.py','run.py','qualification.py','collect.py']]
source += [a/'source/project/run_scripts/jlz_realization/observe.py']
refs=['experiment-reports/servers/server3/memit-history-fixed10k-20260928-v1/completion-review-r1/four-method-comparison-manifest.json','experiment-reports/servers/server4/blue-native-lifelong-comprehensive-review-2026-09-11-v1/source-config-compatibility.csv','experiment-reports/servers/server4/blue-native-lifelong-comprehensive-review-2026-09-11-v1/new-source-config-compatibility.csv','experiment-reports/servers/server4/cake-native-lifelong-b100x100-2026-09-15-v1/completed-review-v2/compatibility.json','experiment-reports/servers/server4/cake-native-lifelong-b100x100-2026-09-15-v1/completed-review-v2/diagnostic-report-ko.md']
test=subprocess.run([sys.executable,'-m','unittest','discover','-s','project/run_scripts/jlz_shared_budget','-p','test_review_20261005.py','-v'],capture_output=True,text=True)
assert test.returncode==0
out=dict(review_date='2026-10-05',analysis_scope='owner CPU/source-backed review; no independent agent',
 active_flags=dict(independent_reverse_future_targets=True,rows_checked=26000,active_occurrences=1984,superseded_occurrences=16,semantics='old occurrence invalidated by any later different target for NFC-normalized subject+relation, matching original observer; no silent latest-target reinterpretation'),
 pilot=dict(status=json.loads((qs[0].parent/'ready.json').read_text()),qualification_files=[entry(f) for f in qs],rollback=json.loads((a/'pilot-V12_MAIN/rollback-probe.json').read_text())),
 source_audit=dict(files=[entry(f) for f in source],scope='request budget/stop, fresh K/h, target tracking no divisor, FP64 ridge, FP32 applied state, postfivewriteHonce, observer/state assertion, reducer completion'),
 input_runtime_receipts=[entry(f) for f in files if f not in qs],baseline_compatibility=[entry(root/f) for f in refs],
 regression=dict(tests=7,exit_code=test.returncode,output=test.stderr),
 no_checkpoint=dict(source_reports_noCP=True,checkpoint_saved_flags_false=True,observed_main_tensor_extension_files=[],reconstructible_state='NOT_AVAILABLE',tensor_reload_validation=False),
 model_calls=0,new_scientific_data=False,Slurm_writes=0,scheduler_observation='one exact accounting snapshot saved before review; no repeated monitoring',
 new_method_or_threshold_changes=False,precision_limit='bounded original model qualification only; source/receipt checks not new tensor proof')
(p/'supplemental-audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
s=json.loads((p/'summary.json').read_text());print('L4share',s['layers']['4']['terminal_plan_share']['mean']);print('files',len(files),'tests',test.returncode)
