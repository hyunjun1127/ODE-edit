"""CPU authority/table closure. Reuses explicitly identified asset evidence, not old authority."""
import argparse
import copy
import collections
import shutil
import sys
from .common import *

PRIOR=Path('/data/janghj/ODE-edit/local/temporal-routing-diagnostic/20260929-v1/preflight-v2/configuration.json')

def prepare(repo,output):
    repo=Path(repo).resolve(); output=Path(output)
    old=read(PRIOR)
    allowed=('checkpoints','dataset','sample','model','model_members','native_dependencies','deps','native_root','hparams','projector','prior_T0')
    config={k:copy.deepcopy(old[k]) for k in allowed}
    config['reuse_bridge']=record(PRIOR)
    config['instruction_id']=NONCE
    config['design']=str(repo/'plans/global/2026-09-29-joint-multilayer-bs10-diagnostic-v1')
    config['envelope']=record(repo/'messages/head/2026-09-29-joint-multilayer-bs10-sh4.md')
    require(config['envelope']['sha256']=='e84c87038d8cbc03bdcb65a25e1adeb7a72cc8e3b6b1d97011abe36333f4b634','ENVELOPE')
    config['authority']=[]; tables={}
    for m in read(repo/'audits/global/2026-09-29-joint-multilayer-bs10-sh4-dispatch/input-manifest.json')['members']:
        p=repo/m['path']; r=record(p)
        require((r['bytes'],r['sha256'])==(m['size'],m['sha256']),'AUTHORITY:'+str(p))
        r['read_level']='FULL_READ'
        if p.suffix=='.csv':
            b=p.read_bytes();require(b.count(b'\n')==b.count(b'\r\n'),'SEALED_CRLF')
            tables[p.name]=csv_rows(p);require(all(None not in x and None not in x.values() for x in tables[p.name]),'CSV_FIELDS')
            r.update(read_level='FULL_BYTE_READ_ALL_FIELDS_VALIDATED',rows=len(tables[p.name]))
        config['authority'].append(r)
    config['official_native']=record(repo/'audits/global/2026-09-22-alphaedit-original-failure-audit/official/AlphaEdit/AlphaEdit_main.py')
    config['contract']=read(Path(config['design'])/'contract.json')
    for k in ('dataset','sample'):
        require(sha(config[k]['path'])==config[k]['sha256'],'DATA:'+k)
    data={int(r['case_id']):r for r in read(config['dataset']['path'])}
    sample=read(config['sample']['path']); order=sorted(sample['records'],key=lambda r:r['ordinal'])
    require(sample['ordered_root']==config['contract']['source']['ordered_root'],'ORDERED_ROOT')
    subject=lambda r:' '.join(data[int(r['case_id'])]['requested_rewrite']['subject'].casefold().split())
    stream=tables['continuation-ids.csv']; panels=tables['panel-ids.csv']; cells=tables['batch-cells.csv']
    require((len(stream),len(panels),len(cells),len(tables['weight-snapshots.csv']))==(500,272,450,18),'COUNTS')
    for r in stream+panels:
        s=order[int(r['source_ordinal'])]
        require(all(str(s[k])==r[k] for k in ('case_id','subject_relation_group','request_sha256','raw_record_sha256')),'ROW_IDENTITY')
    used={subject(r) for r in order[:9000]};eligible=[]
    for r in order[9000:]:
        q=data[r['case_id']];s=subject(r)
        if s in used or len(q['paraphrase_prompts'])<2 or len(q['neighborhood_prompts'])<10 or q['requested_rewrite']['target_new']['str']==q['requested_rewrite']['target_true']['str']:continue
        used.add(s);eligible.append(r['case_id'])
    require(len(eligible)==925 and eligible[:500]==[int(r['case_id']) for r in stream],'METADATA_SELECTION')
    control={subject(r) for r in panels if r['role'].endswith('control')};observer={subject(r) for r in panels if r['role'].endswith('observer')}
    require(not control&observer and not (control|observer)&{subject(r) for r in stream},'FIREWALL')
    for b,br in enumerate(tables['batches.csv'],1):
        expected=' '.join(r['case_id'] for r in stream[(b-1)*10:b*10])
        require(br['case_ids']==expected and int(br['batch'])==b,'MEMBERSHIP')
    config['trajectories']=[]
    for cp in ('B010','B050','B090'):
        # Fixed scheduling order before results: expose one joint first-step path early.
        for arm in ('JOINT_STEP','NATIVE','JOINT_CUM'):
            name=f'{cp}-{arm}'; rows=[r for r in cells if r['trajectory']==name]
            require([r['case_ids'] for r in rows]==[r['case_ids'] for r in tables['batches.csv']],'CHAIN')
            require([int(r['batch']) for r in rows]==list(range(1,51)),'BATCH_ORDER')
            require([int(r['offered_batches']) for r in tables['weight-snapshots.csv'] if r['trajectory']==name]==[25,50],'SNAPSHOTS')
            config['trajectories'].append(dict(checkpoint=cp,arm=arm,name=name))
    for name,r in config['checkpoints'].items():
        st=Path(r['path']).stat(); row=next(x for x in tables['checkpoint-bindings.csv'] if x['checkpoint']==name)
        require((st.st_size,st.st_ino,st.st_mtime_ns)==(r['bytes'],r['inode'],r['mtime_ns']),'CP_STAT')
        require(row['recorded_sha256']==r['sha256'] and int(row['recorded_bytes'])==st.st_size,'CP_REUSE_IDENTITY')
        r['verification']='PRIOR_FULLSHA_AND_W_M_PAYLOAD_VALIDATION_REUSED_CURRENT_STAT_MATCH'
    for r in config['model_members']+[config['projector']]:
        st=Path(r['path']).stat();require(st.st_size==r['bytes'] and st.st_mtime_ns==r['mtime_ns'],'ASSET_STAT')
    for r in config['native_dependencies']:
        require(sha(r['path'])==r['sha256'],'IMPORT_DEPENDENCY')
    config['settings']=dict(seed=20260929,diagnostic_epsilon=1e-12,greedy_max_new_tokens=32,
        greedy_normalization='strip whitespace, casefold; no punctuation removal',microbatch=1,
        full_guard_max=5,materialization_parity_nll_abs=2.5e-4,materialization_parity_margin_abs=5e-4,
        parity_policy='actual values first; structural failure on exceed, no waiver',exact_editor_resume='NOT_AVAILABLE')
    config['resources']=dict(gpu=1,cpus=8,mem_mib=60416,project_cap=2,task_cap=2,
        host_peak_estimate_gib=48,gpu_peak_estimate_gib=75,wall='7-00:00:00',
        wall_is_runtime_measurement=False,save_snapshots=18,snapshot_payload=21139292160,
        output_reserve_bytes=160*2**30,checkpoint_retransfer_bytes=0)
    # Direct USER override. Never mutate GH's sealed BS10 tables or the stopped singleton task.
    override=record(ROOT/'user-bs1-override-to-gh.md')
    original_contract=copy.deepcopy(config['contract']);config['sealed_parent_contract']=original_contract
    c=config['contract'];c.update(status='EXECUTION_USER_BS1_OVERRIDE',batch_size=BATCH_SIZE,batches_per_trajectory=STEPS,edits_per_trajectory=STEPS)
    c['source']['policy']='first100 of the original sealed500; same source order'
    c['geometry']['max_rank_per_layer']=1
    c['name']='joint-multilayer-user-bs1-100-save25-v1'
    c['objective']['edit_guard']='each BS1 request: mean target NLL across six native contexts <= 1.0 nat/token; canonical mean target NLL <= canonical mean original-answer NLL'
    c['joint_solver']['outcome']='smallest-energy actually feasible five-layer candidate; no feasible candidate => atomic reject the offered request'
    c['joint_solver']['microbatch']='USER BS1: one request and one simultaneous five-layer optimizer/commit'
    c['damage_interpretation']='USER BS1 and 100-edit continuation; no promised damage or improvement'
    c['measurements']['detailed_batches']=list(MILESTONES);c['measurements']['detailed_offered_edits']=list(MILESTONES)
    c['weight_snapshots'].update(offered_batches=list(SAVE_STEPS),offered_edits=list(SAVE_STEPS),snapshots=36,total_tensor_bytes=42278584320,total_tensor_gib=39.375,
        technical_early_stop='last finite replaces next scheduled; at most four per trajectory')
    c['budgets'].update(scientific_batch_attempts=900,total_offered_request_exposures=900,unique_continuation_requests=100,
         native_batch_calls=300,native_target_fits=300,joint_batch_solves=600,max_joint_primal_proposals=24000,
         max_joint_backtracking_trial_forwards=144000,max_joint_full_history_guard_passes=3000)
    config['user_override']=dict(nonce=OVERRIDE_NONCE,authority=override,batch_size=1,steps=100,
         edit_layers=list(LAYERS),save_steps=list(SAVE_STEPS),prior_single_layer_task='STOPPED_NOT_RESUMED')
    config['execution_ids']=[int(r['case_id']) for r in stream[:STEPS]]
    config['execution_batches']=[dict(step=i+1,case_ids=[cid]) for i,cid in enumerate(config['execution_ids'])]
    config['resources'].update(save_snapshots=36,snapshot_payload=42278584320,output_reserve_bytes=192*2**30)
    config['resources']['launch_order']='per CP: JOINT_STEP, NATIVE, JOINT_CUM; fixed before outcomes for representative initial observation'
    validate_execution(config)
    require(shutil.disk_usage(ROOT).free>=config['resources']['output_reserve_bytes'],'STORAGE_RESERVE')
    config['disk']=dict(free_bytes=shutil.disk_usage(ROOT).free,exclusive_reservation=False)
    config['status']='INPUTS_BOUND_CPU_ONLY_IMPLEMENTING_NOT_SUBMITTED'
    save(output/'configuration.json',config)
    save(output/'full-read.json',dict(instruction_id=NONCE,authority=config['authority'],
         csv_read='full bytes/all fields validated; not per-row visual reading',job_ids=[],actual_gpu=False,
         reuse_bridge=config['reuse_bridge'],prior_single_layer_stop_preserved=True))
    print(json.dumps(dict(status=config['status'],members=len(config['authority']),trajectories=9,requests=100,panels=272,snapshots=36,job_ids=[])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();prepare(a.repo,a.output)
