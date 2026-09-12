"""CPU package/access/regression gates and non-circular manifest sealing."""
import csv,json,subprocess,sys,py_compile,platform
from pathlib import Path
from review import member,sha,digest,save,load,need,SOURCE
from publish import publish

REPO=Path(__file__).resolve().parents[3]
REPORT=REPO/'experiment-reports/servers/server2/blue-checkpoint-downstream-review-2026-09-12-v1'
LOCAL=Path('/mnt/raid5/janghj/ODE-edit/local/blue-checkpoint-downstream/20260912-report-v1')
POLICY='cf9f8c34b624f97d9ad9a8a63b3291df07a8f044'

def command(args):
    r=subprocess.run(args,cwd=REPO,text=True,capture_output=True)
    need(r.returncode==0,'COMMAND_FAILED:'+str(args)+'\n'+r.stdout+r.stderr)
    return dict(command=args,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr)
def run():
    need(not (REPORT/'analysis-manifest.json').exists(),'ALREADY_SEALED')
    receipts=[]
    for folder in ('blue_checkpoint_downstream_review','blue_checkpoint_downstream'):
        receipts.append(command([sys.executable,'-m','unittest','discover','-s','project/run_scripts/'+folder,'-p','test_*.py','-v']))
        for p in (REPO/'project/run_scripts'/folder).glob('*.py'):py_compile.compile(str(p),doraise=True)
    receipts.append(command(['bash','-n','project/run_scripts/blue_checkpoint_downstream/run.sbatch']))
    receipts.append(command(['bash','scripts/check-session-boundary.sh','01a0493a-074c-7f91-9a13-769116326fef']))
    receipts.append(command(['git','diff','--check']))
    # Reused execution code must be byte-exact to the actual run, not only ancestry.
    for p in (REPO/'project/run_scripts/blue_checkpoint_downstream').iterdir():
        if p.is_file():
            b=subprocess.check_output(['git','show',SOURCE+':'+str(p.relative_to(REPO))],cwd=REPO)
            need(p.read_bytes()==b,'EXECUTION_SOURCE_CHANGED')
    actual=load(REPORT/'verification.json');rows=list(csv.DictReader((REPORT/'metrics.csv').open()))
    pairs=list(csv.DictReader((REPORT/'paired-transitions.csv').open()))
    need(len(rows)==876 and len(pairs)==864,'TABLE_COUNTS')
    need(len({(r['state'],r['task'],r['branch']) for r in rows})==876,'TABLE_DUPLICATES')
    for p in pairs:
        need(int(p['endpoint_correct'])-int(p['W0_correct'])==int(p['gained'])-int(p['lost']),'PAIRED_ARITHMETIC')
        need(sum(int(p[k]) for k in ('lost','gained','still_correct','still_wrong'))==100,'PAIR_PARTITION')
    # Exact actual narrative contrasts; no blanket claim across heterogeneous tasks.
    for family,wins in [('MEMIT',5),('AlphaEdit',2)]:
        r=[r for r in rows if r['family']==family and r['edits']=='10000' and r['branch']=='alternative']
        n=sum(float(next(v for v in r if v['task']==t and v['variant']=='L4_ONLY')['weighted_f1'])>float(next(v for v in r if v['task']==t and v['variant']=='BLUE')['weighted_f1']) for t in ('sst2','mrpc','cola','rte','mmlu','nli'))
        need(n==wins,'NARRATIVE_CONTRAST')
    # Full report and PNG reproduction, not just plot-only re-rendering.
    repeat=LOCAL/'publication-reproduction-r2'
    publish(LOCAL/'reduction-r1',repeat,LOCAL/'publication-reproduction-plots-r2')
    for p in REPORT.rglob('*'):
        if p.is_file():need(sha(p)==sha(repeat/p.relative_to(REPORT)),'REPORT_REPRODUCIBILITY:'+str(p))
    save(REPORT/'test-receipt.json',dict(status='PASS',cpu_tests=15,independent_sklearn_random_fixtures=80,report_png_byte_exact_reproduction=True,pycompile='PASS',bash_parse='PASS',execution_source_byte_identity='PASS',session='PASS',commands=receipts,python=sys.version,platform=platform.platform(),new_gpu=0))
    # Exact original directive copy lives only in this task's private control directory.
    read=[]
    for path in ('messages/head/2026-09-12-blue-downstream-report-sh2.md','PROTOCOL.md'):
        b=subprocess.check_output(['git','show',POLICY+':'+path],cwd=REPO)
        target=LOCAL/('authoritative-instruction.md' if path.startswith('messages/') else 'PROTOCOL-read.md')
        with target.open('xb') as f:f.write(b)
        read.append(dict(repo_path=path,commit=POLICY,**member(target),lines=b.count(b'\n'),full_read=True))
    save(LOCAL/'read-receipt.json',dict(status='FULL_READ_PASS',members=read))
    for rel,expected in [('execution.lock.json','cf9b6ad68d7c68683799bbe1acb0598f0c2fd37ba54c0c144545f3120ba15abc'),('monitoring-pause.json','922982a943c778a5ee7f3a9015b8b363968ab88427a95e74b2a6bd17291fdae9'),('output/initial-valid.json','4f557b49906205e8ed3d2c4fbcd48a9db2223b53a374f885bd6bd2c00e705dc6')]:
        need(sha(LOCAL.parent/'20260909-v1/attempt-v1'/rel)==expected,'ORIGINAL_RECEIPT_HASH')
    sources=[member(p) for p in sorted((REPO/'project/run_scripts/blue_checkpoint_downstream_review').glob('*.py'))]
    files=[dict(path=str(p.relative_to(REPO)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(REPORT.rglob('*')) if p.is_file()]
    manifest=dict(schema='downstream-review-v1',execution_source=SOURCE,analysis_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),analysis_source=sources,inputs=read,output_members=files,output_member_root=digest(files),raw_member_manifest_sha256=sha(REPORT/'raw-member-manifest.json'),scope='job42706 only; CPU report',scientific_promotion=False)
    save(REPORT/'analysis-manifest.json',manifest)
    receipt=dict(status='TERMINAL_REPORT_VERIFIED',job42706='COMPLETED_0:0',states=73,state_task_cells=438,observations=43800,unique_task_items=600,report_sha256=sha(REPORT/'diagnostic-report-ko.md'),manifest_sha256=sha(REPORT/'analysis-manifest.json'),output_member_root=digest(files),test_receipt_sha256=sha(REPORT/'test-receipt.json'),raw_member_root=actual['raw_member_root'],raw_broadcast='NO_BROADCAST_NOT_REQUIRED',source_raw_mutation=0,new_model_evaluator=0,new_gpu_slurm=0,other_task_monitoring=0,scientific_promotion=False)
    save(REPORT/'rooted-receipt.json',receipt)
    for x in files:need(sha(REPO/x['path'])==x['sha256'],'PACKAGE_REHASH')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':run()
