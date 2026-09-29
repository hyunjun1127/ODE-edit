"""Bind S2 existing assets to the received frozen BS1 configuration, CPU only."""
import copy
import gc
import shutil
from pathlib import Path
from .server2_entry import bind, ROOT, NONCE, DEPS, NATIVE, MODEL, DATA
bind()
from .common import read,record,save,sha,require,tensor_sha,csv_rows,validate_execution

def prepare(repo):
    import torch
    torch.set_num_threads(8)
    repo=Path(repo).resolve();landing=ROOT/'inputs/server4-handoff'
    cfg=read(landing/'s4-attempt/configuration.json')
    require(sha(landing/'s4-attempt/configuration.json')=='d903f36882086482ddceee410e00eb6f4ec98f935a37292b5ee3592ee115ec47','S4_CONFIG')
    cfg['s4_provenance']={k:record(landing/'s4-attempt'/k) for k in ('configuration.json','execution.lock.json','submission.json')}
    cfg['cancellation']=record(landing/'cancellation-receipt.json')
    cfg['instruction_id']=NONCE
    cfg['envelope']=record(repo/'messages/head/2026-09-29-joint-multilayer-bs1-sh2-migration.md')
    require(cfg['envelope']['sha256']=='2a16ceefdd2b4e46323f83ec7247a9b4ca4da7c1623034065605206380f3ef9a','MIGRATION_AUTHORITY')
    cfg['authority']=[]
    for m in read(repo/'audits/global/2026-09-29-joint-multilayer-bs10-sh4-dispatch/input-manifest.json')['members']:
        r=record(repo/m['path']);require((r['bytes'],r['sha256'])==(m['size'],m['sha256']),'AUTHORITY')
        cfg['authority'].append(r)
    cfg['design']=str(repo/'plans/global/2026-09-29-joint-multilayer-bs10-diagnostic-v1')
    for key,name in (('dataset','counterfact.json'),('sample','source-sample.lock.json')):
        expected=cfg[key]['sha256'];cfg[key]=record(DATA/name);require(cfg[key]['sha256']==expected,'DATA_IDENTITY')
    stream=csv_rows(Path(cfg['design'])/'continuation-ids.csv')
    require(cfg['execution_ids']==[int(r['case_id']) for r in stream[:100]],'BS1_FIRST100_OF500')
    order=read(cfg['sample']['path'])['records'];byord={r['ordinal']:r for r in order}
    for row in stream+csv_rows(Path(cfg['design'])/'panel-ids.csv'):
        old=byord[int(row['source_ordinal'])]
        require(all(str(old[k])==row[k] for k in ('case_id','subject_relation_group','request_sha256','raw_record_sha256')),'ALL_FIELDS_DATA_BINDING')
    cfg['model']=str(MODEL)
    prior_path=Path('/mnt/raid5/janghj/ODE-edit/local/checkpoint-mechanism-audit/20260920-v1/attempt-v1/inputs/model-asset-verification.json')
    prior={Path(r['path']).name:r for r in read(prior_path)['members']}
    cfg['model_identity_receipt']=record(prior_path);members=[]
    for old in cfg['model_members']:
        p=MODEL/Path(old['path']).name;s=p.stat();r=prior[p.name]
        require(r['sha256']==old['sha256'] and (s.st_size,s.st_ino,s.st_mtime_ns)==(r['bytes'],r['inode'],r['mtime_ns']),'MODEL_REUSE_STAT')
        members.append(dict(r,verification='PRIOR_S2_FULLSHA_AND_CURRENT_STAT_MATCH'))
    cfg['model_members']=members
    cfg['native_dependencies']=[record(p) for p in sorted(NATIVE.rglob('*')) if p.is_file() and p.suffix in ('.py','.yml')]
    cfg['dependency_scope']='Exact received actual native import closure; S4 broad 1801-member unrelated manifest retained only in s4_provenance'
    cfg['deps']=str(DEPS);cfg['native_root']=str(NATIVE)
    cfg['official_native']=record(repo/'audits/global/2026-09-22-alphaedit-original-failure-audit/official/AlphaEdit/AlphaEdit_main.py')
    cp_rows=read(repo/'transfers/approvals/2026-09-29-joint-multilayer-bs10-sh4-inputs.json')['files']
    for m in cp_rows:
        p=Path(m['source_path']);name=p.parent.name;old=cfg['checkpoints'][name]
        r=record(p);require((r['bytes'],r['sha256'])==(m['bytes'],m['sha256']),'PARENT_FILE')
        cp=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
        require(set(cp['weights'])==set(old['weights']) and list(cp['cache_c'].shape)==[5,14336,14336],'PARENT_SCHEMA')
        for k,v in cp['weights'].items():
            require(v.dtype==torch.float32 and list(v.shape)==[4096,14336] and bool(torch.isfinite(v).all()) and tensor_sha(v)==old['weights'][k],'PARENT_WEIGHT')
        require(cp['cache_c'].dtype==torch.float32 and tensor_sha(cp['cache_c'])==old['history_sha256'],'PARENT_HISTORY')
        require(all(bool(torch.isfinite(x).all()) for x in cp['cache_c']),'PARENT_HISTORY_FINITE')
        require(cp['metadata']['contexts']==old['metadata']['contexts'] and 'rng' in cp['metadata'],'PARENT_CONTEXT_RNG')
        cfg['checkpoints'][name]=dict(old,**r);cfg['checkpoints'][name]['verification']='S2_CURRENT_FULLSHA_CPU_W_M_CONTEXT_RNG_BINDING'
        del cp;gc.collect();print('BOUND',name,flush=True)
    expected=cfg['projector']['sha256'];p=Path('/mnt/raid5/janghj/EasyEdit/examples/null_space_project_Meta-Llama-3-8B-Instruct.pt')
    cfg['projector']=record(p);require(cfg['projector']['sha256']==expected,'PROJECTOR_FULLSHA')
    cfg['user_override']['authority']=record(landing/'s4-attempt/user-override.md')
    cfg['resources'].update(node='server2',gpu_peak_estimate_gib=44,host_peak_estimate_gib=48,
        memory_estimate_basis='FP32 model 29.92GiB; sequential FP64 raw-P geometry <=9.2GiB temporaries; single-row joint/native7-row short sequence activations separately, not simultaneous with solve; no model dense parameter gradients; CPU P/M/copies. Actual peaks NOT_MEASURED.',
        wall='7-00:00:00',memory_guarantee=False)
    cfg['disk']=dict(free_bytes=shutil.disk_usage(ROOT).free,exclusive_reservation=False)
    require(cfg['disk']['free_bytes']>=cfg['resources']['output_reserve_bytes'],'DISK_RESERVE')
    cfg['status']='S2_INPUTS_BOUND_CPU_ONLY_NOT_SUBMITTED'
    validate_execution(cfg)
    save(ROOT/'preflight/configuration.json',cfg)
    save(ROOT/'preflight/port-receipt.json',dict(configuration=record(ROOT/'preflight/configuration.json'),
        cancellation=cfg['cancellation'],model_receipt=cfg['model_identity_receipt'],checkpoint_files=[{k:r[k] for k in ('path','bytes','sha256','verification')} for r in cfg['checkpoints'].values()],
        original_scientific_files_changed=False,actual_gpu=False,large_transfer_bytes=0))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);prepare(p.parse_args().repo)
