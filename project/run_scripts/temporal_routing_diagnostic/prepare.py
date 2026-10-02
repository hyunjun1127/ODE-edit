"""CPU-only complete sealed-table validation and read-only asset binding."""
import argparse
import collections
import copy
import importlib
import os
from pathlib import Path
import sys
import time
from .common import *


def table_check(repo):
    base = repo/'plans/global/2026-09-29-temporal-routing-diagnostic-v1'
    manifest = read(repo/'audits/global/2026-09-29-temporal-routing-sh4-dispatch/input-manifest.json')
    members = []
    tables = {}
    for x in manifest['members']:
        p = repo/x['path']; r = record(p)
        require((r['bytes'], r['sha256']) == (x['size'], x['sha256']), 'AUTHORITY_BYTES:'+x['path'])
        if p.suffix == '.csv':
            raw = p.read_bytes(); require(raw.count(b'\r\n') == raw.count(b'\n'), 'CRLF')
            rows = csv_rows(p); require(all(None not in r and None not in r.values() for r in rows), 'CSV_SCHEMA')
            tables[p.name] = rows
            r.update(rows=len(rows), read_level='FULL_BYTE_READ_AND_ALL_ROW_FIELD_VALIDATION')
        else:
            p.read_text(); r['read_level'] = 'FULL_READ'
        members.append(r)
    stream = tables['continuation-ids.csv']; cells = tables['fit-cells.csv']
    routing = tables['stream-routing.csv']; panels = tables['panel-ids.csv']; snaps = tables['weight-snapshots.csv']
    require(len(stream)==100 and len(cells)==1500 and len(routing)==1500 and len(panels)==120 and len(snaps)==30, 'COUNTS')
    require([int(x['step']) for x in stream]==list(range(1,101)), 'ORDER')
    for b in ('B010','B050','B090'):
        for l in LAYERS:
            branch=f'{b}-L{l}'; br=[r for r in cells if r['branch']==branch]
            require([r['case_id'] for r in br]==[r['case_id'] for r in stream], 'BRANCH_ORDER')
            for i,r in enumerate(br,1):
                require(r['cell_id']==f'{branch}-S{i:03d}' and int(r['step'])==i and int(r['physical_layer'])==l and r['batch_size']=='1', 'CELL')
                require(r['parent_cell']==(f'{branch}-S{i-1:03d}' if i>1 else f'BASE_ALPHAEDIT:{b}'), 'PARENT')
                require(r['save_actual_weight']==str(i in (50,100)), 'SNAPSHOT_FLAG')
            require([int(r['step']) for r in snaps if r['branch']==branch]==[50,100], 'SNAPSHOT_ORDER')
    require(routing==[{k:r[k] for k in routing[0]} for r in cells], 'ROUTING')
    return base, members, tables


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--repo',required=True); ap.add_argument('--output',required=True)
    args=ap.parse_args(); repo=Path(args.repo).resolve(); out=Path(args.output).resolve(); start=time.monotonic()
    base,members,tables=table_check(repo)
    envelope=repo/'messages/head/2026-09-29-temporal-routing-diagnostic-sh4.md'
    require(sha(envelope)=='4df7b8fc4d7e5ad5710a188ecc944d72221fa19ed72155d7ed54a3c747c109bd','ENVELOPE')
    sys.path.insert(0,str(DEPS)); import torch; import transformers
    torch.set_num_threads(8)
    require(torch.__version__=='2.9.1+cu128' and transformers.__version__=='4.44.2','RUNTIME_VERSION')
    require(sha(DATA/'counterfact.json')=='3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1','DATA')
    require(sha(DATA/'source-sample.lock.json')=='a8d22c230611b6c14740d1d00658a53856bbde26d060189d5ddf30f2ffdbac92','SAMPLE')
    data=read(DATA/'counterfact.json'); byid={int(r['case_id']):r for r in data}
    locked=read(DATA/'source-sample.lock.json')['records']; order=sorted(locked,key=lambda r:r['ordinal'])
    norm=lambda x:' '.join(x.casefold().split())
    subject=lambda r:norm(byid[int(r['case_id'])]['requested_rewrite']['subject'])
    oldsubjects={subject(r) for r in order[:9000]}; stream=tables['continuation-ids.csv']
    for r in stream+tables['panel-ids.csv']:
        s=order[int(r['source_ordinal'])]
        require(all(str(s[k])==r[k] for k in ('case_id','subject_relation_group','request_sha256','raw_record_sha256')), 'ROW_BINDING')
    require(not {subject(r) for r in stream}&oldsubjects and len({subject(r) for r in stream})==100,'FRESH_SUBJECT')
    used={subject(r) for r in stream}; eligible=[]
    for r in order[9000:]:
        q=byid[int(r['case_id'])]
        if subject(r) in oldsubjects or subject(r) in {subject(x) for x in eligible}: continue
        if len(q['paraphrase_prompts'])<2 or len(q['neighborhood_prompts'])<10 or q['requested_rewrite']['target_new']['str']==q['requested_rewrite']['target_true']['str']:continue
        eligible.append(r)
        if len(eligible)==100:break
    require([int(r['case_id']) for r in stream]==[r['case_id'] for r in eligible],'METADATA_SELECTION')
    sensor={subject(r) for r in tables['panel-ids.csv'] if r['role'].endswith('sensor')}
    observer={subject(r) for r in tables['panel-ids.csv'] if r['role'].endswith('observer')}
    require(not sensor&observer and not (sensor|observer)&used,'FIREWALL')
    prior=Path('/data/janghj/ODE-edit/local/historical-update-timeaxis/20260924-v1/inputs/t0-v1/runtime-binding.json')
    require(sha(prior)=='717808cb2460b034d7f526bb14cf325e0ad817f678d6fa7cd98844649f051e41','PRIOR_T0')
    t0=read(prior); cps={}
    for b in (10,50,90):
        r=copy.deepcopy(t0['checkpoints']['BASE_ALPHAEDIT'][str(b)]); p=Path(r['path']); st=p.stat()
        expected=next(x for x in tables['checkpoint-bindings.csv'] if x['checkpoint']==f'B{b:03d}')
        require((st.st_size,st.st_ino,st.st_mtime_ns)==(r['bytes'],r['inode'],r['mtime_ns']), 'CP_STAT_CHANGED')
        require(r['sha256']==expected['recorded_sha256'] and r['bytes']==int(expected['recorded_bytes']),'CP_MANIFEST')
        cp=torch.load(p,map_location='cpu',weights_only=True,mmap=True); md=cp['metadata']
        require(set(cp)=={'weights','cache_c','metadata'} and md['batch']==b and md['method']=='AlphaEdit','CP_SCHEMA')
        require(md['base_model_revision']==REVISION and md['seen_ids']==[x['case_id'] for x in order[:b*100]],'CP_PARENT')
        require(cp['cache_c'].shape==(5,14336,14336) and cp['cache_c'].dtype==torch.float32,'HISTORY_SCHEMA')
        for l in LAYERS:
            name=f'model.layers.{l}.mlp.down_proj.weight'; w=cp['weights'][name]
            require(w.shape==(4096,14336) and w.dtype==torch.float32 and bool(torch.isfinite(w).all()),'WEIGHT_SCHEMA')
            require(tensor_sha(w)==r['weights'][name],'WEIGHT_HASH')
        require(bool(torch.isfinite(cp['cache_c']).all()),'HISTORY_FINITE')
        r.update(verification='PRIOR_FULL_FILE_SHA_PLUS_CURRENT_STAT_AND_FRESH_W_HASH_M_SCHEMA_FINITE',
            history_sha256=tensor_sha(cp['cache_c']), metadata=md)
        cps[f'B{b:03d}']=r; del cp
    oldroot=Path('/data/janghj/ODE-edit/local/alpha-key-concentration-causal/20260923-r1')
    old=read(oldroot/'inputs/design/evidence/audits/global/2026-09-22-alphaedit-native-criticality-audit/target-native-execution.lock.json')
    binding=read(oldroot/'receipts/native-input-binding-r2.json')
    deps=[]
    for r in binding['members']:
        p=Path(r['resolved_path'])
        if p.suffix in ('.py','.json','.yml') and p.is_file():
            now=record(p); require(now['sha256']==r['sha256'],'DEPENDENCY_CHANGED:'+str(p)); deps.append(now)
    projector=Path(old['projector']); pr=record(projector)
    require(pr['sha256']=='6d356468c6408dca694c1907ffde31cddb99f7e67fa2e74502910d5afbede5ec','PROJECTOR')
    P=torch.load(projector,map_location='cpu',weights_only=True,mmap=True)
    require(P.shape==(5,14336,14336) and P.dtype==torch.float32 and bool(torch.isfinite(P).all()),'P_SCHEMA')
    pr['tensor_sha256']=tensor_sha(P); del P
    hp=read(oldroot/'inputs/design/contract.json')['baseline']['hparams']; hp['blue']=False
    model_members=[]
    for p in sorted(MODEL.iterdir()):
        if p.is_file() and p.suffix in ('.json','.safetensors'):
            prior_member=next((r for r in t0['model_shards'] if Path(r['path']).name==p.name),None) if p.suffix=='.safetensors' else None
            if prior_member:
                st=p.stat(); require(st.st_size==prior_member['bytes'],'MODEL_SIZE'); model_members.append(dict(prior_member,verification='PRIOR_FULLSHA_CURRENT_SIZE'))
            else:model_members.append(record(p))
    stat=os.statvfs(ROOT); free=stat.f_bavail*stat.f_frsize
    require(free>64*2**30,'STORAGE_BLOCKED_64GIB_NEW_OUTPUT_RESERVE')
    config=dict(instruction_id=NONCE, authority=members,envelope=record(envelope),design=str(base),
        checkpoints=cps,prior_T0=record(prior),dataset=record(DATA/'counterfact.json'),sample=record(DATA/'source-sample.lock.json'),
        hparams=hp,projector=pr,native_root=str(NATIVE),deps=str(DEPS),native_dependencies=deps,
        official_native=record(repo/'audits/global/2026-09-22-alphaedit-original-failure-audit/official/AlphaEdit/AlphaEdit_main.py'),
        model=str(MODEL),model_members=model_members,writer_tokenizer=dict(add_bos_token_attribute=False,padding='right',native_backend_POSTPROCESSOR_UNCHANGED=True),
        observer=dict(microbatch=1,tokenization='BOS prompt + separately encoded leading-space target, TF input excludes final target; all valid TF positions',
            capture_dtype='float32',kl_reduction='FP64 p0 exp times logp diff; answer token mean then question mean',epsilon=1e-12,
            patch_seed=20260929,parity_nll_atol=2.5e-4,key_atol=1e-6,key_rtol=1e-6,zero_hook_logits_atol=1e-6),
        resources=dict(gpu_cap=2,cpus=8,mem_mib=60416,output_reserve_bytes=64*2**30,observed_free_bytes=free,
            snapshots_payload_bytes=7046430720,estimated_host_peak_gib=30,estimated_gpu_peak_gib=48,
            solve_dtype='original native FP32, not changed to FP64',fp64_diagnostic_scratch='bounded selected weight and response only'),
        snapshots=dict(steps=[50,100],count=30,save_actual_full_selected_weight=True,exact_editor_resume='NOT_AVAILABLE'),
        scientific_fits=1500,technical_duplicate_fits=0,max_total_fits=1502,technical='INTEGRATED_FIRST_SCIENTIFIC_WRITE_NO_DUPLICATE_FIT',
        status='INPUT_CPU_READY_ACTUAL_GPU_NOT_RUN',seconds=time.monotonic()-start)
    print(save(out/'input-binding.json',config),flush=True)
    print(save(out/'full-read.json',dict(members=members,table_rows={k:len(v) for k,v in tables.items()},
        semantic_docs='FULL_READ by owner; all CSV rows parsed and exact bindings validated, not visual per-row inspection',GPU_calls=0)),flush=True)


if __name__=='__main__':main()
