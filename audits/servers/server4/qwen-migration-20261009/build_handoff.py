"""Export small existing metadata only; never model/data/CP/credentials."""
import csv
import hashlib
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone

OUT=Path(__file__).parent;BUNDLE=OUT/'handoff'
BASE=Path('/data/janghj/ODE-edit/local/qwen-baselines-12-20261009')
CF=BASE/'execution-cf-display-r1';ZS=BASE/'execution-noqual-r2'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2)
cessation=read(OUT/'cessation.json');assert len(cessation['cancelled_pending_ids'])==23
assert not BUNDLE.exists()
BUNDLE.mkdir()
sources=[]
for root in (CF,ZS):
    lock=read(root/'source-lock.json')
    for m in lock['members']:assert sha(root/'source'/m['relative'])==m['sha256']
    for m in read(root/'input-lock.json')['members']:assert sha(Path(m['path']))==m['sha256']
    paths=['official/runners/server4/'+n for n in ('qwen_run.py','qwen_pipeline.py','qwen_assets.py','qwen_factual.py','qwen_native_state.py','qwen_plan.py')]
    paths+=['project/run_scripts/server4_qwen_archive.py','project/run_scripts/server4_qwen_submit.py']
    sources.append(dict(original_root=str(root),commit=lock['code_commit'],official_tree_sha256=lock['official_tree_sha256'],
                        source_lock_sha256=sha(root/'source-lock.json'),input_lock_sha256=sha(root/'input-lock.json'),
                        key_members=[dict(path=p,sha256=sha(root/'source'/p)) for p in paths]))
put(BUNDLE/'source-provenance.json',sources)
rows=[]
for row in cessation['rows']:
    if row['kind']!='gpu':continue
    root=Path(row['root']);path=root/'configs'/f"{row['logical_main_row']}.json"
    config=read(path)
    assert config['config_sha256']==row['config_sha256']
    dest=BUNDLE/'configs'/path.name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(path,dest)
    stream=read(root/'streams'/f"{row['dataset']}-stream.lock.json")
    rows.append(dict(model='qwen25',method=row['method'],dataset=row['dataset'],logical_main_row=row['logical_main_row'],
                     old_job_id=row['job_id'],old_state=row['state'],destination_job_id=None,
                     config_sha256=config['config_sha256'],config_file_sha256=sha(path),
                     source=row['source'],ordered_sample_sha256=stream['ordered_case_ids_sha256'],
                     stream_sha256=stream['stream_sha256'],completed_edits=0,W20_observed=False,
                     metric_status='NOT_MEASURED',metrics={},main_result_eligible=False,
                     migration_status='SOURCE_CEASED_DESTINATION_NOT_SUBMITTED_BY_SH4'))
for dataset,root in [('cf',CF),('zsre',ZS)]:
    shutil.copyfile(root/'streams'/f'{dataset}-stream.lock.json',BUNDLE/f'{dataset}-stream.lock.json')
assets=read(CF/'assets.json')
for key in ('wandb_env','wandb_sdk_python'):assets.pop(key,None)
put(BUNDLE/'server4-asset-paths-reference-only.json',assets)
preflight=read(CF/'asset-preflight.json')
put(BUNDLE/'asset-provenance.json',dict(asset_identity=preflight['asset_identity'],runtime=preflight['runtime'],
        model_identity=preflight['model_identity'],physical_provenance=preflight['physical_provenance'],
        provenance='Historical source preflight receipt; SH2 must bind its actual local paths/runtime/storage',
        original_receipt_sha256=sha(CF/'asset-preflight.json')))
gen=Path(assets['generation_reference_manifest']);shutil.copyfile(gen,BUNDLE/'generation-reference-metadata.json')
put(BUNDLE/'archive-host-binding-reference.json',dict(receiver=read(CF/'receiver.json'),
        policy_sha256=sha(CF/'archive-policy.json'),policy_path='control/final-checkpoint-archive-policy.json',
        source_owner_change='SH2 must use its own future-job adoption/cutover/actual submission proof; never relabel server4 trust or old actual IDs',
        consumers='writer/evaluation share GPU job; archive CPU reads payload bytes only after actual W20 and all consumers clear',
        admission='fresh per-payload receiver capacity/independent verify before source delete, failure KEEP',
        previous_transfers=0,previous_deletions=0))
put(BUNDLE/'host-diff.json',dict(source_server='server4',destination_server='server2',
    preserve=['original native configs/layers/L2/seed/order/B100x20/FP32/TF32off','BLUE L2=1 no grid',
              'CF W0_AND_W20_FIRST2000 generation; zsRE no CF generation','official request-macro metrics and native display companions',
              'checkpoint latest1/W20 retention; finite/shape/nonedited/commit guards; no-GPU-qualification USER_DISABLED'],
    must_rebind=['own Slurm node/QoS/GPU model/memory/CPU/cap and existing frontiers','actual model/tokenizer/C0/P/dataset/reference hashes and local paths',
                 'own runtime Python/SDK/credential path without copying credentials','source/config/content provenance and all new actual IDs',
                 'own archive source-owner proof and server1 receiver admission','W&B server/task/attempt/job identity'],
    context='Native context generator/state is in pinned native registry source; no context cache or W0 was generated on these attempts. Preserve edit seed0 and checkpoint context/RNG policy.',
    W0='No usable official W0 observation from these attempts. Perform only original required W0 or reuse independently exact compatible raw; do not relabel PRICE W0.',
    launcher='Do not run server4_qwen_submit unchanged: it hardcodes server4 and original IDs. Use SH2 own wrapper and its actual resource dependencies.',
    algorithms='official.baselines.registry; source references via Git, no private task algorithm fork',
    source_choice='CF corrected execution dc80ec529c940019d1bee27a67a4e908eb37cc64 includes native display companions. zsRE prior d614add5 is historical provenance; adopt common corrected source with unchanged zsRE science.',
    qualification='NOT_RUN_USER_DISABLED',old_checkpoint_resume=False))
put(OUT/'result-eligibility.json',dict(at=datetime.now(timezone.utc).isoformat(),
    scope='Current server4 approved official Qwen12 migration rows; prior Llama60917..60922 cancelled with elapsed0',
    rows=rows,new_completed_W20_main_results=[],eligible_numeric_updates=0,
    coverage_limit='No historical/tuning/heldout/PRICE results automatically promoted; existing README historical exceptions remain untouched',
    old_llama_main_ids=[60917,60918,60919,60920,60921,60922],old_llama_status='CANCELLED elapsed0; no revive',
    missing_metrics='Omitted, not zero; CF generation NOT_MEASURED (original schedule not changed to DEFERRED)'))
with (OUT/'result-eligibility.csv').open('x',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['model','method','dataset','old_job_id','old_state','destination_job_id','completed_edits','W20_observed','main_result_eligible','metric_status','source','config_sha256','ordered_sample_sha256'])
    writer.writeheader();writer.writerows({k:r.get(k) for k in writer.fieldnames} for r in rows)
members=[dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(BUNDLE.rglob('*')) if p.is_file()]
put(OUT/'handoff-manifest.json',dict(at=datetime.now(timezone.utc).isoformat(),members=members,
    total_bytes=sum(m['bytes'] for m in members),model_or_raw_transfer=False,
    cessation_sha256=sha(OUT/'cessation.json'),results_sha256=sha(OUT/'result-eligibility.json')))
print(json.dumps(dict(members=len(members),bytes=sum(m['bytes'] for m in members),eligible_numeric_updates=0)))
