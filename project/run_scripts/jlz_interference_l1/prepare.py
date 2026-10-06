"""CPU-token-only fresh full native/evaluator/runtime/resource binding."""
import argparse,copy,csv,json,os,platform,shutil,unicodedata
from pathlib import Path
from scripts.fixed_counterfact import load_prefix
from project.run_scripts.jlz_writer_coupled.prepare import runtime_binding,asset_binding
from project.run_scripts.jlz_realization.observe import active_flags
from . import *
from .profile import ARMS,arm_profile

ASSET_REFERENCE=Path('/data/janghj/ODE-edit/local/jlz-v13-mdcd-sequential-2k/20261005-v1/attempt-r1')
MANIFEST_SHA='b7c7e7644bd1878143ce6df43f58b384978b10e01a235617d29081f5052fb78b'
SCHEDULE_SHA='dccb4da4896aa7c3f653f7192bd01ab5d8799a0b439101417d02ee287d3fd2e2'

def verify_schedule(path,records):
    require(sha(path)==SCHEDULE_SHA,'SCHEDULE_SHA')
    with path.open(newline='') as f:r=csv.DictReader(f);fields=r.fieldnames;rows=list(r)
    require(fields==['stream_index0','batch1','slot0','case_id','claim_sha256','target_new_sha256','active_at_W20']
            and len(rows)==len(records)==2000,'FULL_CSV_FIELDS_ROWS')
    flags=active_flags(records)
    for i,(row,record) in enumerate(zip(rows,records)):
        rw=record['requested_rewrite'];values=[i,i//100+1,i%100,record['case_id'],
            digest([unicodedata.normalize('NFC',' '.join(rw['subject'].split())),rw['relation_id']]),
            digest(rw['target_new']),int(flags[record['case_id']])]
        require([row[k] for k in fields]==list(map(str,values)),'CSV_FIXED_IDENTITY:'+str(i))
        require(len(record['paraphrase_prompts'])==2 and len(record['neighborhood_prompts'])==10,'OBSERVATION_DENOMINATORS')

def native_binding(prior,records,prior_alignment):
    from transformers import AutoTokenizer
    from project.run_scripts.jlz_realization.inputs import CounterFactAdapter
    tokenizer=AutoTokenizer.from_pretrained(prior['model'],local_files_only=True)
    tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='right'
    bench=CounterFactAdapter(tokenizer,json.loads(Path(prior['contexts']).read_text()))
    packs=[];bindings=[]
    for i in range(20):
        pack=bench.prepare(records[i*100:(i+1)*100]);old=prior_alignment['packs'][i]
        require(pack['identity']==old['identity'] and pack['record_ids']==old['ids'],'FRESH_20_NATIVE_PACK')
        rows=[];widths=[]
        for j,kind in enumerate(pack['row_kind']):
            n=int(pack['tokens']['attention_mask'][j].sum())
            rows.append(dict(row=j,owner=pack['row_request'][j],kind=kind,lookup=pack['lookup'][j],
                input_ids=pack['tokens']['input_ids'][j,:n].tolist(),
                attention_mask=pack['tokens']['attention_mask'][j,:n].tolist(),targets=pack['targets'][j,:n].tolist()))
        for owner in range(pack['n_requests']):
            own=[r for r in rows if r['owner']==owner]
            require(len(own)==pack['n_rw']+1 and sum(r['kind']=='kl' for r in own)==1,'COMPLETE_NATIVE_OWNER_ROWS')
            widths.append(max(len(r['input_ids']) for r in own))
        packs.append(copy.deepcopy(old));bindings.append(dict(batch=i+1,pack=pack['identity'],rows=len(rows),
            valid_tokens=sum(len(r['input_ids']) for r in rows),owner_padded_tokens=sum(widths)*(pack['n_rw']+1),
            max_owner_width=max(widths),full_native_tokens_sha256=digest(rows),KL_future_tokens_retained=True))
    T=max(b['owner_padded_tokens'] for b in bindings);GiB=1024**3
    # Fixed dense H/A and selected RAM snapshots, plus entry/current BUILD
    # CPU offload. No old measured fit-time or memory waiver is inherited.
    fixed=27.034;entry=(14336+4096)*4*T/GiB;build=4*(14336+4096)*4*T/GiB
    extra=dict(entry_prefix=entry,candidate_BUILD_boundaries=build,subject_RAMScratch=4096*4*T/GiB)
    peak=fixed+sum(extra.values());require(peak<58,'RESOURCE_BLOCKED_HOST_ESTIMATE')
    # Declared architecture plus bounded one-owner graph workspace; this is a
    # planning estimate, not an actual GPU peak or a qualification assertion.
    model_config=json.loads((Path(prior['model'])/'config.json').read_text())
    h=model_config['hidden_size'];d=model_config['intermediate_size'];n=model_config['num_hidden_layers']
    vocab=model_config['vocab_size'];kv=h*model_config['num_key_value_heads']//model_config['num_attention_heads']
    parameters=2*vocab*h+n*(2*h*h+2*h*kv+3*h*d+2*h)+h
    gpu_parts=dict(model_FP32=parameters*4/GiB,five_FP64_factors=5*d*d*8/GiB,
        factor_transients=2*d*d*8/GiB,selected_entry_and_candidate=2*5*h*d*4/GiB,
        single_owner_checkpoint_head_allocator_workspace_reserve=24.)
    gpu_peak=sum(gpu_parts.values());require(gpu_peak<95,'RESOURCE_BLOCKED_VRAM_ESTIMATE')
    return packs,dict(scope='Fresh CPU native full-token pack hashes only; no model load/forward/GPU PASS',
        all20=bindings,rows=sum(b['rows'] for b in bindings),max_owner_padded_tokens=T,
        fixed_host_GiB=fixed,host_extra_GiB=extra,host_peak_estimate_GiB=peak,estimate_not_measured=True,
        GPU_estimate_parts_GiB=gpu_parts,GPU_peak_planning_GiB=gpu_peak,
        GPU_workspace_status='planning reserve; actual peak NOT_MEASURED for this method',
        new_price_auxiliary_bytes=5*100*100*8+32*5*100*8,
        inherited_receipt_scope='Prior immutable full native packs/runtime/assets only; no old planner/qualification inheritance')

def prepare(out,attempt,cpu_preflight,cpus=8):
    out,attempt,cpu_preflight=map(lambda p:Path(p).resolve(),(out,attempt,cpu_preflight))
    require(out.is_relative_to(LOCAL) and attempt.is_relative_to(LOCAL),'TASK_LOCAL_BOUNDARY')
    require(not attempt.exists() and not (out/'configuration.json').exists(),'CREATE_ONCE')
    require(type(cpus) is int and 1<=cpus<=8,'LEGAL_RESOURCE_CPU_BINDING')
    cpu=json.loads(verify(member(cpu_preflight)).read_text());require(cpu.get('passed') is True,'SOURCE_CPU_REQUIRED')
    envelope=json.loads((ROOT/ENVELOPE).read_text());require(envelope['nonce']==NONCE and envelope['task_id']==TASK,'AUTHORITY')
    manifest=ROOT/DESIGN/'artifact-manifest.json';authority=[member(manifest),member(ROOT/ENVELOPE)]
    require(sha(manifest)==MANIFEST_SHA,'MANIFEST_BYTE_IDENTITY')
    for row in json.loads(manifest.read_text())['files']:
        p=ROOT/DESIGN/row['path'];require(p.is_file() and not p.is_symlink() and p.stat().st_size==row['bytes']
            and sha(p)==row['sha256'],'CANONICAL_SHA_SIZE:'+row['path']);authority.append(member(p))
    prior=json.loads((ASSET_REFERENCE/'config.json').read_text());lock=json.loads((ASSET_REFERENCE/'execution.lock.json').read_text())
    require(sha(ASSET_REFERENCE/'config.json')==lock['config_sha256'],'ASSET_REFERENCE_CONFIG')
    for row in lock['native_reference']+lock['dependency_sources']:verify(row)
    runtime=runtime_binding();require((runtime['torch'],runtime['transformers'])==
        (prior['runtime']['torch'],prior['runtime']['transformers']),'RUNTIME_VERSIONS')
    old={r['module']:r['sha256'] for r in prior['runtime']['source_members']}
    require(all(old.get(r['module'])==r['sha256'] for r in runtime['source_members']),'RUNTIME_SOURCE')
    oldassets={r['path']:r for r in prior['assets']}
    assets=[asset_binding(r['path'],oldassets,member(ASSET_REFERENCE/'config.json')) for r in prior['assets']]
    main=load_prefix(Path(prior['stream']).parent,2000)
    require(digest([r['case_id'] for r in main])==ORDERED_SHA,'MAIN_ORDER_EXACT')
    verify_schedule(ROOT/SCHEDULE,main)
    alignment=verify(prior['native_input_alignment']);observer=verify(prior['observer_identity'])
    identities=json.loads(observer.read_text())['rows']
    require(len(identities)==26000 and len({r['identity'] for r in identities})==26000,'ALL_OBSERVER_IDENTITIES')
    readonly=[]
    for relative in ('project/run_scripts/jlz_realization/inputs.py','project/run_scripts/jlz_realization/observe.py',
                     'project/run_scripts/jlz_pilot/prompts.py','scripts/fixed_counterfact.py'):
        require(sha(ROOT/relative)==sha(ASSET_REFERENCE/'source'/relative),'INPUT_EVALUATOR_EXACT_SOURCE')
        readonly.append(member(ROOT/relative))
    packs,binding=native_binding(prior,main,json.loads(alignment.read_text()))
    write(out/'native-full-input-binding.json',binding)
    from .storage import storage_plan
    storage=storage_plan()
    free=shutil.disk_usage(LOCAL).free;inodes=os.statvfs(LOCAL).f_favail;reserve=storage['reserve_bytes']
    require(free>=reserve and inodes>=10000,'RESOURCE_BLOCKED_STORAGE')
    c={k:copy.deepcopy(prior[k]) for k in ('model','stream','contexts','stats','profile','native_root','stats_root')}
    c.update(instruction_id=NONCE,task_id=TASK,attempt=str(attempt),run_instance=dict(date='2026-10-06',attempt=attempt.name),
        seed=20261002,runtime=runtime,assets=assets,authority_members=authority,cpu_preflight=member(cpu_preflight),
        native_reference=lock['native_reference'],dependency_sources=lock['dependency_sources'],native_hparams=prior['native_hparams'],
        native_input_alignment=member(alignment),native_full_input_binding=member(out/'native-full-input-binding.json'),
        observer_identity=member(observer),readonly_input_evaluator_sources=readonly,packs=packs,ordered_ids_sha256=ORDERED_SHA,
        cold_W0_H0=prior['qualification_reuse']['cold_W0_H0'],arm_profiles={arm:arm_profile(prior['profile'],arm) for arm in ARMS},
        W0_reuse=dict(status='NOT_AVAILABLE',reason='No arbitrary old efficacy/raw read; each new cold arm observes samefirst2k'),
        settings=dict(B=100,batches=20,requests=2000,arms=list(ARMS),fit_requests_per_group=1,observer_microbatch=2,
            milestones=list(MILESTONES),build_cap=25,subject_forward_cap=25,subject_backward_cap=24,no_B21=True,
            save_checkpoints=False,new_baselines=0),
        integrated_B1_assertions=dict(pairs=[[0,1],[1,0]],layers=[4,5,6,7,8],new_solves=0,new_model_calls=0,
            new_fits=0,old_toy_PASS_inherited=False,standalone_qualification_job=False),storage=storage,
        resources=dict(task_cap=2,project_cap=3,gpu=1,cpu=cpus,host_mib=59392,hard_host_mib=60416,
            collector_cpu=8,collector_host_mib=24576,wall='2-00:00:00',collector_wall='04:00:00',
            reserve_bytes=reserve,startup_free_bytes_min=storage['next_batch_bytes']+storage['error_reserve_bytes'],free_bytes=free,free_inodes=inodes,
            serializer_bound=storage,
            quota_status='quota command unavailable; same filesystem statvfs bound; no inferred quota exemption',
            host_peak_estimate=binding,ETA='NOT_MEASURED_NEW_METHOD;48h requested ceiling not ETA'))
    write(out/'configuration.json',c)
    write(out/'input-runtime-binding.json',dict(nonce=NONCE,worktree=str(ROOT),host=platform.node(),canonical=authority,
        native_full_input_binding=member(out/'native-full-input-binding.json'),ordered_first2000_sha256=ORDERED_SHA,
        observer_rows=26000,integrated_actual_checks=c['integrated_B1_assertions'],assets=assets,runtime=runtime,
        synthetic_tests=0,toy_PASS_inherited=False,source_static_import_only=True,
        GPU_qualification='NOT_RUN',model_load=False,old_task_science_or_queue_polling=False))
    return dict(configuration=str(out/'configuration.json'),packs=20,requests=2000,arms=3,GPU='NOT_RUN')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--attempt',type=Path,required=True)
    p.add_argument('--cpu-preflight',type=Path,required=True);p.add_argument('--cpus',type=int,default=8)
    a=p.parse_args();print(json.dumps(prepare(a.out,a.attempt,a.cpu_preflight,a.cpus)))
