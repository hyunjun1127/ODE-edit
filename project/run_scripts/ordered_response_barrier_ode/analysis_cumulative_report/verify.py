"""Raw-free package arithmetic, input/output seal and focused test receipt."""
import argparse,subprocess,sys
from pathlib import Path
from .common import *
from .report import REPORT

FORBIDDEN={'.pt','.pth','.bin','.safetensors','.log','.out','.err','.jsonl'}
def scan_public(root):
    for p in root.rglob('*'):
        require(not p.is_symlink(),'publication symlink')
        if not p.is_file():continue
        require(p.suffix not in FORBIDDEN,'raw binary/log publication '+str(p))
        if p.suffix=='.json':
            def visit(v):
                if isinstance(v,dict):
                    require(not set(v)&{'prompt','target_token_ids','token_predictions','token_correct','token_logits','password','access_token','api_key'},'raw/credential key')
                    for x in v.values():visit(x)
                elif isinstance(v,list):
                    for x in v:visit(x)
            visit(read(p))
    return True

def verify(package,seal=False):
    p=Path(package);scan_public(p)
    integ=read(p/'tables/integrity/integrity-receipt.json');deep_identity(integ)
    for cell in read(p/'tables/integrity/source-runtime-provenance.json'):
        r=cell['result'];d=r['dtype_receipt'];inv=r['parameter_inventory'];a=r['attention_backend_receipt'];e=r['easyedit']
        require(set(inv['parameter_elements_by_dtype'])==set(inv['parameter_tensors_by_dtype'])=={'torch.float32'},'parameter inventory FP32')
        require(not inv['quantized'] and not d['quantized'] and not d['autocast_enabled'] and not d['tf32_enabled'] and d['bf16_fp16_cast_count']==0,'precision modes')
        require(a['terminal_jvp_evaluator_shared_backend'] and a['per_observation_backend_switch_count']==0,'attention identity')
        require(sha256_file(Path(a['model_class_source_path']))==a['model_class_source_sha256'],'model class source binding')
        require(sha256_file(Path(a['hf_config_path']))==a['hf_config_sha256'],'HF config binding')
        require(e['head']=='14cea8245f06715684592ab55184939b99d70784' and e['tree']=='9c52aadbc0883da422badf0a730fff21aaa3a8a7' and e['tracked_clean'],'pinned Official source')
    c=pd.read_csv(p/'tables/performance/cumulative-core.csv');f=pd.read_csv(p/'tables/performance/final-20-arm.csv')
    require(len(c)==200 and len(f)==20,'full table counts')
    require(not c.duplicated(['cell','arm','batch']).any(),'checkpoint duplicate')
    for (cell,arm),g in c.groupby(['cell','arm']):require(sorted(g.batch)==list(range(1,11)),'all checkpoints')
    for metric,scale in [('RS',100),('PS',200),('PS_strict',100),('NS',1000),('rewrite_acc',100),('rephrase_acc',200)]:
        require((c[metric+'_den']==c.batch*scale).all(),'metric cardinality')
        require(((c[metric+'_num']>=0)&(c[metric+'_num']<=c[metric+'_den'])).all(),'numerator range')
        require(np.allclose(c[metric+'_num']/c[metric+'_den'],c[metric+'_rate'],atol=1e-15,rtol=0),'rate arithmetic')
    pd.testing.assert_frame_equal(f.reset_index(drop=True),c[c.batch==10].reset_index(drop=True))
    pp=pd.read_csv(p/'tables/performance/paired-prompt-transitions.csv')
    require((pp.arm_success==pp.both_success+pp.recovery).all() and (pp.O_success==pp.both_success+pp.loss).all(),'paired transitions arithmetic')
    r=pd.read_csv(p/'tables/performance/atwrite-final-partitions.csv')
    require((r.final_RS==r.at_write_success-r.success_to_loss+r.recovery).all(),'forgetting partitions')
    inv=read(p/'table-inventory.json')
    for m in inv:
        path=p/m['path'];require(sha256_file(path)==m['sha256'] and path.stat().st_size==m['bytes'],'table member seal')
        require(len(pd.read_csv(path))==m['rows'],'table exact rowcount')
    plots=read(p/'figures/plot-manifest.json')
    for m in plots['figures']:
        require(sha256_file(p/'figures'/m['path'])==m['sha256'] and m['byte_stable_rerender'],'plot rehash/reproduction')
    # Same input CSV bytes available within this package, despite local original path.
    for m in plots['inputs']:
        matches=list((p/'tables').rglob(Path(m['path']).name));require(len(matches)==1 and sha256_file(matches[0])==m['sha256'],'plot published input binding')
    report=(p/REPORT).read_text();require('전체 1,000개 재평가' in report and 'Layer-wise Update Magnitude' in report,'report headline/title')
    source=read(p/'analysis-source.json');code=Path(__file__).parent;repo=code.parents[3]
    for m in source['modules']:require(sha256_file(repo/m['path'])==m['sha256'],'analysis code identity')
    if not seal:
        manifest=read(p/'analysis-manifest.json');receipt=read(p/'rooted-analysis-receipt.json');deep_identity(receipt)
        members=manifest['members'];actual=[member(x,relative_to=p) for x in sorted(p.rglob('*')) if x.is_file() and x.name not in ('analysis-manifest.json','rooted-analysis-receipt.json')]
        require(members==actual,'exact package inventory')
        require(canonical_hash(members)==manifest['member_root']==receipt['member_root'],'package root')
        require(sha256_file(p/'analysis-manifest.json')==receipt['manifest_sha256'],'manifest SHA');return receipt
    require(not (p/'analysis-manifest.json').exists(),'create-once seal')
    result=subprocess.run([sys.executable,'-m','unittest','-q','project.run_scripts.ordered_response_barrier_ode.analysis_cumulative_report.test_focused'],cwd=repo,text=True,capture_output=True)
    require(result.returncode==0,'focused tests '+result.stdout+result.stderr)
    audit=dict(status='ESSENTIAL_ANALYSIS_GATES_PASS',focused_command='python -m unittest -q project.run_scripts.ordered_response_barrier_ode.analysis_cumulative_report.test_focused',focused_stdout=result.stdout,focused_stderr=result.stderr,table_count=len(inv),figure_count=len(plots['figures']),checkpoints=200,final_arms=20,cumulative_request_states=110000,model_GPU_evaluator=0,raw_free_scan=True,cardinality_rate_partition_arithmetic=True,plot_bytes_reproduced=True,all_input_raw_member_root=integ['raw_member_root'])
    write_json_once(p/'tests-and-audit.json',audit)
    members=[member(x,relative_to=p) for x in sorted(p.rglob('*')) if x.is_file()]
    manifest=dict(schema='orbode-cumulative-analysis-manifest/v1',instruction_id=INSTRUCTION,members=members,member_root=canonical_hash(members),input_raw_member_root=integ['raw_member_root'],executed_source=SOURCE,analysis_source=source,status='ANALYSIS_COMPLETE')
    write_json_once(p/'analysis-manifest.json',manifest)
    receipt=dict(schema='orbode-cumulative-rooted-receipt/v1',instruction_id=INSTRUCTION,status='FULL_REHASH_ANALYSIS_COMPLETE',manifest_sha256=sha256_file(p/'analysis-manifest.json'),report_path=REPORT,report_sha256=sha256_file(p/REPORT),member_root=manifest['member_root'],members=len(members),checkpoints=200,primary_arms=20,cumulative_request_states=110000,final_endpoint_request_states=20000,final_in_cumulative_once=True,scientific_promotion=False,task_owned_active_pending=0,no_new_evaluator_model_GPU=True)
    receipt['identity_sha256']=canonical_hash(receipt);write_json_once(p/'rooted-analysis-receipt.json',receipt)
    return verify(p,seal=False)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,required=True);ap.add_argument('--seal',action='store_true');a=ap.parse_args();print(json.dumps(verify(a.package,a.seal)))
