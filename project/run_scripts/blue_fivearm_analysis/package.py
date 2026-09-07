"""Focused tests, reproducibility, raw-free scope and final package hashes."""
import argparse
import io
import subprocess
import unittest
from .common import *
from .plots import generate

def validate(repo,reproduce):
 out=repo/OUT_REL
 suite=unittest.defaultTestLoader.loadTestsFromName('project.run_scripts.blue_fivearm_analysis.test_focused')
 stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=1).run(suite)
 print(stream.getvalue());assert result.wasSuccessful()
 for name in ['current_batch','seen_prefix','final_metrics']:
  for r in csvread(out/(name+'.csv')):
   for t,m in [('RS',1),('PS',2),('NS',10)]:
    n,d=int(r[t+'_num']),int(r[t+'_den']);assert 0<=n<=d and d==int(r['request_count'])*m
    assert abs(float(r[t+'_rate'])-n/d)<1e-12
 for r in csvread(out/'endpoint_partitions.csv'):
  assert int(r['at_write_success'])-int(r['at_write_success_to_final_failure'])+int(r['initially_failed_to_final_recovery'])==int(r['final_RS'])
 for name,n in [('final_metrics',6),('current_batch',60),('seen_prefix',18),('allseen_rewrite',60),('retention_cohort',330),('batch-integrity',30),('checkpoint-integrity',9)]:
  assert len(csvread(out/(name+'.csv')))==n,(name,n)
 repro=generate(out,reproduce);original=read(out/'figures/plot-manifest.json')
 assert {x['path']:x['sha256'] for x in repro}=={x['path']:x['sha256'] for x in original}
 forbidden=['.pt','.safetensors','.log','.out','.err','.tar','.npz']
 assert not any(p.suffix in forbidden for p in out.rglob('*') if p.is_file())
 # Source compilation is CPU only; do not create __pycache__ artifacts.
 for p in (repo/'project/run_scripts/blue_fivearm_analysis').glob('*.py'):compile(p.read_text(),str(p),'exec')
 save(out/'focused-tests.json',dict(status='PASS',focused_tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
  actual_table_cardinality_arithmetic='PASS',PNG_byte_reproduction=len(repro),raw_free_extension_scope='PASS',source_compile='PASS',
  experimental_preflight_distinction='Analysis tests only; L4 pre-run skipped remains SKIPPED_USER_DIRECTED',GPU=0,model=0,evaluator=0))

def seal(repo):
 out=repo/OUT_REL
 source=[dict(path=str(p.relative_to(repo)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted((repo/'project/run_scripts/blue_fivearm_analysis').glob('*.py'))]
 members=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p),**({'rows':len(csvread(p))} if p.suffix=='.csv' else {})) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['analysis-manifest.json','rooted-receipt.json']]
 manifest=dict(schema='blue-fivearm-sequential-analysis.v1',instruction_id='ODEEDIT-S06-BLUE-FIVEARM-SEQUENTIAL-DETAILED-REPORT-SH4-V1',
  analysis_source_head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
  analysis_source_tree=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD^{tree}'],text=True).strip(),
  sample_root=SAMPLE_ROOT,arms=ARMS,denominators=dict(final_requests_per_arm=1000,RS=1000,PS=2000,NS=10000,current_rows=60,full_seen_rows=18,retention_cohort_rows=330),
  blue_jobs=[38940,38988,38997],reference_publication='0d0a0131e4a6a2a645dfa6530377d420a084d136',
  reference_report_sha='a7b07d16ea36e0aefe7aa16fb2c259275ee1879d486f084a469f9f94dd414732',
  source_code=source,members=members,member_root=digest(members),scientific_promotion=False,new_model_GPU_evaluator_Slurm_actions=0,
  raw_broadcast='LOCAL_ONLY_NO_LARGE_TRANSFER',local_only_source='L4/L8 runtime hooks remain local; archive paths and SHA in source-config-compatibility.csv')
 save(out/'analysis-manifest.json',manifest)
 # Independent file reads verify every newly generated package member.
 verify_members(out,members,'final_package')
 receipt=dict(status='REVIEW_READY',report_sha256=sha(out/'factual-report-ko.md'),manifest_sha256=sha(out/'analysis-manifest.json'),
  member_root=manifest['member_root'],members=len(members),CSV_count=len(list(out.glob('*.csv'))),PNG_count=len(list((out/'figures').glob('*.png'))),
  tests=read(out/'focused-tests.json'),new_jobs=0,BLUE_task_owned_active_pending=0,terminal_observation='bounded audit all three COMPLETED0; no repeated polling',
  main_integration='USER_AUTHORIZED; actual final HEAD/tree recorded in external closeout receipt',scientific_promotion=False,next='TASK_COMPLETE_STOP_AFTER_MAIN_PUSH_AND_GH_REPORT')
 receipt['root_sha256']=digest(receipt);save(out/'rooted-receipt.json',receipt)
 print(json.dumps(receipt,ensure_ascii=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['validate','seal']);p.add_argument('--repo',type=Path,required=True);p.add_argument('--reproduce',type=Path);a=p.parse_args()
 if a.stage=='validate':validate(a.repo,a.reproduce)
 else:seal(a.repo)
