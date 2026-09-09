"""Essential CPU table/test/plot/raw-free checks and create-once package root."""
import argparse,io,subprocess,unittest
from .common import *
from .plots import generate

def validate(out,repro):
    suite=unittest.defaultTestLoader.loadTestsFromName('project.run_scripts.blue_lifelong_analysis.test_focused')
    log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite);print(log.getvalue());assert result.wasSuccessful()
    expected={'final-summary':6,'cumulative-metrics':216,'current-metrics':1800,'age-strata-metrics':648,'cohort-retention':10008,'layer-updates':800,'chain-integrity':6,'checkpoint-tensors':96,'checkpoint-recovery-metadata':72,'dataset-evaluator-identity':672,'target-tensor-audit':600,'raw-member-inventory':3762}
    for name,n in expected.items():assert len(csvread(out/(name+'.csv')))==n,(name,n)
    for name in ['cumulative-metrics','current-metrics','age-strata-metrics']:
        for r in csvread(out/(name+'.csv')):
            n,d=int(r['numerator']),int(r['denominator']);assert 0<=n<=d and abs(n/d-float(r['rate']))<1e-12
            assert d==int(r['request_denominator'])*MULT[r['metric']]
    for name in ['prompt-transitions','paired-final-transitions','overwrite-strata']:
        for r in csvread(out/(name+'.csv')):
            assert int(r['retained'])+int(r['lost'])+int(r['gained'])+int(r['both_failed'])==int(r['denominator'])
            assert int(r['before_num'])-int(r['lost'])+int(r['gained'])==int(r['after_num'])
    repl=generate(out,repro);original=read(out/'figures/plot-manifest.json')
    assert {r['path']:r['sha256'] for r in repl}=={r['path']:r['sha256'] for r in original}
    assert len(original)==10
    forbidden={'.pt','.bin','.npz','.safetensors','.tar','.log','.out','.err','.pyc'}
    assert not any(p.suffix in forbidden for p in out.rglob('*') if p.is_file())
    for p in Path(__file__).parent.glob('*.py'):compile(p.read_text(),str(p),'exec')
    save(out/'focused-tests.json',dict(status='PASS',test_count=result.testsRun,failures=len(result.failures),errors=len(result.errors),table_counts=expected,
                                      denominator_partition_arithmetic='PASS',PNG_byte_stable_count=len(repl),raw_free_scope='PASS',GPU_tests=0,pre_run_smoke='SKIPPED_USER_DIRECTED remains unchanged'))

def seal(out):
    repo=Path(__file__).resolve().parents[3]
    source=[dict(path=str(p.relative_to(repo)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(Path(__file__).parent.glob('*.py'))]
    members=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p),**({'rows':len(csvread(p))} if p.suffix=='.csv' else {})) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['analysis-manifest.json','rooted-receipt.json']]
    man=dict(instruction_id=INSTRUCTION,analysis_source_head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),analysis_source_tree=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD^{tree}'],text=True).strip(),
             source=source,members=members,member_root=digest(members),sample_root=SAMPLE_ROOT,prefix_root=PREFIX_ROOT,arms=ARMS,jobs=JOBS,
             complete_chains=6,batches=600,checkpoints=72,cumulative_request_states=333600,unique_requests=10000,final_request_observations=60000,
             experimental_source='source-config-compatibility.csv (local archives, not analysis HEAD)',raw_member_root=read(out/'full-audit-receipt.json')['raw_member_root'],scientific_promotion=False)
    save(out/'analysis-manifest.json',man)
    for m in members:
        p=out/m['path'];assert sha(p)==m['sha256'] and p.stat().st_size==m['bytes']
    rec=dict(instruction_id=INSTRUCTION,status='REVIEW_READY',member_root=man['member_root'],report_sha256=sha(out/'factual-report-ko.md'),manifest_sha256=sha(out/'analysis-manifest.json'),
             member_count=len(members),CSV_count=len(list(out.glob('*.csv'))),PNG_count=len(list((out/'figures').glob('*.png'))),tests=read(out/'focused-tests.json'),
             complete_chains=6,selected_task_active_pending=0,scheduler_check='single bounded initial audit; all six COMPLETED0',new_GPU_model_evaluator_Slurm=0,
             other_L567_jobs='NOT_INSPECTED_OR_MUTATED',raw_rehash=read(out/'full-audit-receipt.json'),scientific_promotion=False,next='MAIN_INTEGRATION_GH_REPORT_THEN_STOP')
    rec['root_sha256']=digest(rec);save(out/'rooted-receipt.json',rec);print(json.dumps(rec,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['validate','seal']);p.add_argument('--out',type=Path,required=True);p.add_argument('--repro',type=Path);a=p.parse_args()
    if a.stage=='validate':validate(a.out,a.repro)
    else:seal(a.out)
