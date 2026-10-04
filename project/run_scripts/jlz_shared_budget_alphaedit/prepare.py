"""Fresh asset/authority binding; reuse exact completed CPU packing, never old scientific state."""
import argparse,copy,json,shutil,os,subprocess
from pathlib import Path
import torch
from project.run_scripts.jlz_writer_coupled.prepare import runtime_binding,asset_binding
from .common import *

REFERENCE_CONFIG=Path('/data/janghj/ODE-edit/local/jlz-v12-shared-budget/20261004-v1/attempt-s3-r1/config.json')
REFERENCE_SHA='b09cfae0d6a8a29071818c114017b9f350ec9c58ec3143ac211b42179145458b'

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',default='preparation-r1');args=p.parse_args()
    out=LOCAL/args.name;require(out.parent==LOCAL and not out.exists(),'CREATE_ONCE')
    require(sha(ROOT/ENVELOPE)=='d9d52fec3fcd7155df7bd167018dd5dbbe7fd95f0447897e3a2531700b551da0','ENVELOPE_SHA')
    execution=ROOT/'plans/global/2026-10-04-jlz-v12-alphaedit-writer/execution-command.json'
    require(sha(execution)=='15731b6b2ef1c9c9db2aadda4cfa23384668262ae45cbabb1e5f67d8b2e292ce','USER_EXECUTION_SHA')
    require(sha(REFERENCE_CONFIG)==REFERENCE_SHA,'REFERENCE_CONFIG')
    prior=json.loads(REFERENCE_CONFIG.read_text());c=copy.deepcopy(prior)
    old_read=ROOT/'audits/servers/server3/jlz-v12-shared-budget-bs100x20-20261004-v1/full-read.json'
    members=json.loads(old_read.read_text())['members']
    members=members+[dict(path=str(Path(DESIGN)/'execution-manifest.json'),bytes=(ROOT/DESIGN/'execution-manifest.json').stat().st_size,
        sha256='3028d2796efb646fca5e316ea09678b1d2f539869cdbed30b99f27d05162be45')]
    for r in members:
        path=ROOT/r['path'];require(path.stat().st_size==r['bytes'] and sha(path)==r['sha256'],'PARENT_FULL_READ_REBIND')
    for field in ('observer_identity','native_input_alignment'):verify(c[field])
    runtime=runtime_binding()
    require(runtime['source_root_sha256']==prior['runtime']['source_root_sha256'] and runtime['transformers']=='4.57.1' and runtime['torch']=='2.9.1+cu128','RUNTIME_INPUT_REUSE')
    assets=[]
    for r in prior['assets']:
        x=asset_binding(r['path'],{r['path']:r},member(REFERENCE_CONFIG))
        if 'logical_path' in r:
            require(Path(r['logical_path']).resolve()==Path(x['path']).resolve(),'MODEL_BINDING');x['logical_path']=r['logical_path']
        assets.append(x)
    ready=json.loads((ROOT/'agents/server3/experiment-ready-paths-20260919-v1.json').read_text())['models']['llama3-8b-inst']['assets']
    pi=ready['projector'];row=member(pi['path']);require(row['sha256']==pi['sha256'],'PROJECTOR_SHA')
    torch.set_num_threads(8)
    tensor=torch.load(pi['path'],map_location='cpu',mmap=True,weights_only=True)
    require(list(tensor.shape)==[5,14336,14336] and tensor.dtype==torch.float32,'PROJECTOR_HEADER')
    require(pi['layer_mapping']=={str(l):l-4 for l in range(4,9)},'MAPPING')
    for slab in tensor:
        for block in slab.split(512):require(bool(torch.isfinite(block).all()),'PROJECTOR_FINITE')
    assets.append(row)
    native=Path(c['native_root'])/'AlphaEdit/AlphaEdit_main.py'
    native_hp=Path(c['native_root'])/'hparams/AlphaEdit/Llama3-8B.json'
    hp=json.loads(native_hp.read_text());require(hp['L2']==10 and hp['nullspace_threshold']==.02 and hp['layers']==list(range(4,9)),'NATIVE_ALPHAEDIT_PROFILE')
    # Original BLUE source is read-only; exact Git object comparison, no upstream fetch.
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=c['native_root'],text=True).strip()
    require(revision=='311b076a92e4ed0f14f5c8b4909732da781bc5f7','BLUE_REVISION')
    for path in sorted((Path(c['native_root'])/'AlphaEdit').glob('*.py')):
        rel=path.relative_to(c['native_root']);original=subprocess.check_output(['git','show',revision+':'+str(rel)],cwd=c['native_root'])
        require(path.read_bytes()==original,'NATIVE_SOURCE_DIRTY');c['native_reference'].append(member(path))
    c['native_reference'].append(member(native_hp))
    c.update(instruction_id=NONCE,task_id=TASK,authority=AUTHORITY,runtime=runtime,assets=assets)
    c.pop('migration_instruction',None);c.pop('source_handoff',None)
    c['profile']['lambda_C']=0;c['settings']['chains']=['V12_ALPHAEDIT']
    c['native_writer_sha']=sha(native)
    c['alphaedit']=dict(native_source=member(native),native_hparams=member(native_hp),native_git=revision,
        L2=10,blue=False,dtype='float32',projector=row,layer_mapping=pi['layer_mapping'],threshold=.02,
        projector_provenance=member(ROOT/'agents/server4/alphaedit-runtime-path-seal.json'),
        C0_provenance=[r for r in assets if r['path'].endswith('.npz')],
        covariance_ridge_in_writer=0,parent_FP64_solve_gate=False,shared_budget_projection='float64',
        key_reference='native MEMIT/AlphaEdit compute_ks byte comparison',main_residual='z_virtual_layer-h_current_layer_no_divisor')
    memitks=Path(c['native_root'])/'memit/compute_ks.py';alphaks=Path(c['native_root'])/'AlphaEdit/compute_ks.py'
    # Module hparams import/type names differ; native function body comparison is bound below.
    import ast
    def body(p):return ast.dump(next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='compute_ks'),include_attributes=False)
    ma=ast.parse(memitks.read_text());aa=ast.parse(alphaks.read_text())
    fm=next(n for n in ma.body if isinstance(n,ast.FunctionDef));fa=next(n for n in aa.body if isinstance(n,ast.FunctionDef))
    require([ast.dump(n) for n in fm.body]==[ast.dump(n) for n in fa.body],'NATIVE_KEY_FUNCTION_BODY')
    c['alphaedit']['native_key_body_exact']=True
    c['resources'].update(host_mib=59392,reserve_bytes=24*1024**3,free_bytes=shutil.disk_usage(LOCAL).free)
    require(c['resources']['free_bytes']>=c['resources']['reserve_bytes'],'STORAGE')
    c['resources']['host_plan_GiB']=dict(loader_peak=34,history=3.83,outer_transaction=4.93,entry_cache=3,
        optional_probe_transaction=4.93,one_layer_prior_temporary=3.1,projector_mmap_RSS=3.83,Python_IO_margin=8)
    c['resources']['host_plan_note']='Nonconcurrent loader34GiB; conservative probe/fit peak<33GiB including resident projector mmap and outer+inner RAM transactions; bounded58GiB. ActualRSS not established.'
    c['resources']['GPU_plan_GiB']=dict(model=30,projectors=3.83,weights_entry_write=4,activation_head=20,
        largest_probe_ridge_FP64_solve=10,margin=10)
    c['resources']['ETA_hours']='NOT_MEASURED_ALPHAEDIT; main168h ceiling; no baseline throughput parity claim'
    c['runtime']['geometry_dtype']='float64_shared_projection_only';c['runtime']['writer_dtype']='float32'
    c['runtime_port']['isolated_overlay']=str(OVERLAY)
    c['runtime_port']['actual_S3_model_qualification']='NEW_ALPHAEDIT_NOT_RUN'
    c['W0']=dict(mode='ONE_SHARED_FRESH_FIRST2000',reason='No completed exact-static W0 receipt prebound; live ridge scientific artifacts not inspected',allowed_by_current_envelope=True)
    c['B1_probe']=dict(enabled=True,extra_fit=0,RAM_only=True,exact_restore_required=True,cost_separate=True)
    c['authority_members']=[member(ROOT/r['path']) for r in members]+[member(ROOT/ENVELOPE),member(ROOT/EXCEPTION),member(execution)]
    c['authority_members'] += [member(ROOT/'agents/server3/experiment-ready-paths-20260919-v1.json'),c['alphaedit']['projector_provenance']]
    c['parent_config']=member(REFERENCE_CONFIG)
    write(out/'configuration.json',c)
    write(out/'full-read.json',dict(nonce=NONCE,authority=AUTHORITY,envelope=member(ROOT/ENVELOPE),execution=member(execution),
        parent_members=members,prior_FULL_READ=member(old_read),reuse_mode='EXACT_BYTES_ALL27_REBOUND',
        input_pack_reuse=member(REFERENCE_CONFIG),packs=len(c['packing']),observer_rows=26000,
        projector=dict(**row,shape=list(tensor.shape),dtype=str(tensor.dtype),layer_mapping=pi['layer_mapping'],finite=True),
        CPU_header_only=True,actual_GPU='NOT_RUN',duplicate_local_submission_receipts=list(map(str,LOCAL.glob('attempt-*/submission.json')))))
    print(json.dumps(dict(config=str(out/'configuration.json'),projector_sha=row['sha256'],status='CPU_PREPARED_GPU_NOT_RUN')))

if __name__=='__main__':main()
