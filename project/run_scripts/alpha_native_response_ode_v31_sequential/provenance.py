"""Outcome-blind 1000-request seal; immutable pilot/reserved inventories only."""
import argparse,hashlib,json,subprocess
from pathlib import Path
from project.run_scripts.native_response_ode_v31.provenance import save,git
from project.run_scripts.ordered_response_barrier_ode import preflight as assets
from .contracts import SCIENCE,NAMESPACE,PILOT,CHAINS


def select_samples(repo,dataset):
    data=Path(dataset).read_bytes()
    if hashlib.sha256(data).hexdigest()!=assets.DATASET_SHA256 or len(data)!=assets.DATASET_BYTES:raise RuntimeError('DATASET_IDENTITY')
    rows=json.loads(data);pilot=json.loads((repo/PILOT).read_text())
    if assets.canonical_hash(pilot['records'])!=pilot['ordered_root'] or len(pilot['records'])!=38:raise RuntimeError('PILOT_EXCLUSION_IDENTITY')
    reserved=json.loads((repo/assets.STREAM_SEAL_RELATIVE).read_text());body=dict(reserved);root=body.pop('root_digest')
    if root!=assets.STREAM_ROOT or assets.canonical_hash(body)!=root or reserved['all_request_order_sha256']!=assets.ORDER_ROOT:raise RuntimeError('RESERVED_EXCLUSION_IDENTITY')
    excluded={int(x['case_id']) for x in pilot['records']}|{int(x['case_id']) for x in reserved['requests']}
    ranked=sorted((hashlib.sha256((NAMESPACE+'|'+str(int(r['case_id']))).encode()).hexdigest(),int(r['case_id']),r)
        for r in rows if int(r['case_id']) not in excluded)
    records=[];groups={}
    for i,(rank,case,r) in enumerate(ranked[:1000]):
        rw=r['requested_rewrite'];key=assets.canonical_hash([rw.get('relation_id'),rw['subject']])
        groups.setdefault(key,[]).append(case)
        records.append(dict(ordinal=i,batch_index=i//100+1,batch_ordinal=i%100,case_id=case,rank_sha256=rank,
            raw_record_sha256=assets.canonical_hash(r),request_sha256=assets.canonical_hash(rw),
            subject_relation_group=key,target_new_sha256=assets.canonical_hash(rw['target_new']),
            target_true_sha256=assets.canonical_hash(rw['target_true'])))
    if len(records)!=1000 or len({r['case_id'] for r in records})!=1000:raise RuntimeError('MAIN_SAMPLE_COUNT')
    return dict(namespace=NAMESPACE,records=records,ordered_root=assets.canonical_hash(records),
        dataset_sha256=assets.DATASET_SHA256,exclusion_pilot_sha256=assets.sha256_file(repo/PILOT),
        pilot_ids=[r['case_id'] for r in pilot['records']],reserved_stream_root=root,reserved_order_root=assets.ORDER_ROOT,
        reserved_manifest_sha256=assets.sha256_file(repo/assets.STREAM_SEAL_RELATIVE),
        pilot_overlap_count=0,reserved_overlap_count=0,outcome_selection_count=0,
        repeated_subject_relation_groups={k:v for k,v in groups.items() if len(v)>1},
        overwrite_status='PREDECLARED_GROUP_AND_TARGET_IDENTITIES_NO_OUTCOME_FILTERING',
        smoke_ids=[r['case_id'] for r in pilot['records'] if r['fixture'] in ('G0A','G0B')])


def prepare(repo,root,official):
    repo,root,official=map(lambda p:Path(p).absolute(),(repo,root,official))
    if root.exists():raise RuntimeError('CREATE_ONCE_RUN_ROOT')
    if git(repo,'status','--porcelain','--untracked-files=all'):raise RuntimeError('SOURCE_MUST_BE_CLEAN')
    source=dict(head=git(repo,'rev-parse','HEAD'),tree=git(repo,'rev-parse','HEAD^{tree}'),repo=str(repo),
        parent=git(repo,'rev-parse','HEAD^'),official=assets.validate_easyedit_source(official))
    paths=git(repo,'ls-files','project/run_scripts/alpha_native_response_ode_v31_sequential',
        'project/run_scripts/native_response_ode_v31','project/run_scripts/ordered_response_barrier_ode',
        'project/run_scripts/alphaedit_strength_neutral_barrier').splitlines()
    source['members']=[dict(path=p,bytes=(repo/p).stat().st_size,sha256=assets.sha256_file(repo/p)) for p in paths]
    source['members_root']=assets.canonical_hash(source['members'])
    source['official_shallow_snapshot']=True
    source['original_contract_text_sha256']='8f69ba38aabef627a7e37547e8b2bf78224054046e9bc42bf400e2995a7efb0e'
    source['contract_file']='/mnt/raid5/janghj/ODE-edit/local/alpha-native-response-v31-sequential-routing/handoff-20260906-v1/authoritative-contract.txt'
    source['contract_file_sha256']=assets.sha256_file(Path(source['contract_file']))
    if source['contract_file_sha256']!='c192ec7e6ca0ebad5554cf02d4c95ccc1fa4ec4d53a8f9db56479080680e61f5':raise RuntimeError('CONTRACT_FILE_DRIFT')
    sample=select_samples(repo,assets.EASYEDIT_ARTIFACT_ROOT/assets.DATASET_RELATIVE)
    # Full source-backed asset checks reused; asset bytes never regenerated.
    asset=assets.validate_model_artifacts(repo,assets.EASYEDIT_ARTIFACT_ROOT,assets.HF_HUB_CACHE_ROOT,deep_hash=True)
    root.mkdir(parents=True,mode=0o700)
    save(root/'source.lock.json',source);save(root/'science.lock.json',SCIENCE);save(root/'sample.lock.json',sample)
    save(root/'assets.lock.json',asset)
    save(root/'resource.lock.json',dict(server='server2',cap=3,gpu_per_process=1,mem_mib=60416,cpus=8,
        maximum_hours=48,pilot_gpu_hour_cap_inherited=False,chains=CHAINS))
    save(root/'dry-plan.json',dict(smoke=dict(indices=[0,1],requests_per_batch=1,batches=2,arm='JV_NATIVE',
        smoke_sample_ids=sample['smoke_ids'],main_initialization='fresh W0/M0; smoke discarded'),
        main=dict(indices=[0,1,2,3],array_throttle_max=3,batches=10,requests_per_batch=100),
        secondary=dict(indices=[4,5],requires='MAIN_TABLE_READY',batches=10,requests_per_batch=100),
        sample_root=sample['ordered_root'],existing_primitive_tests_reused=True,
        cold_model_loads=8,model_loads_detail='two smoke + six chains; no extra scientific arm',
        full_checkpoint_batches=[1,5,10],W0_reference='once per model after smoke W0 restoration',
        no_main_merge=True,resource_recount_required=True))
    return str(root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--official',type=Path,required=True);a=p.parse_args();print(prepare(a.repo,a.root,a.official))
