"""Fresh CPU bindings; reuse exact protected native/model/C0/P assets."""
import argparse
import copy
import csv
import importlib
import platform
import shutil
import subprocess
from .common import *
from scripts.fixed_counterfact import load_prefix
from project.run_scripts.base_model_eval.gpt2xl_server1_prepare import runtime
from project.run_scripts.experiment_tracking.schema import load_env
from project.run_scripts.experiment_generation_eval.assets import json_read as asset_read,member as asset_member
from project.run_scripts.experiment_generation_eval.common import SCHEMA,PROFILE,EVAL_SEED

OLD={
 'stock':Path('/mnt/raid5/janghj/ODE-edit/local/gpt2xl-native-baselines/20261007-v1/attempt-v1/config.json'),
 'cake':Path('/mnt/raid5/janghj/ODE-edit/local/gpt2xl-cake-blue-2k/20261007-v1/attempt-v1/config.json'),
 'prune':Path('/mnt/raid5/janghj/ODE-edit/local/gpt2xl-prune-rect-2k/20261007-v1/attempt-v1/config.json')}

def dependency_members(configs):
    found={}
    def add(row):
        verify(row);found[row['path']]=row
    for c in configs.values():
        for row in c['runtime']['source_members']+c['evaluator_sources']:add(row)
        if 'native_bundle' in c:
            for bundle in c['native_bundle'].values():
                for row in bundle['files']:
                    add(row['original']);add(row['effective'])
        else:
            for row in c['native']['closure']:add(row)
            for row in c['native']['hparams'].values():add(row)
            for row in c['native']['context_reuse']['source_members']:add(row)
            add(c['native']['context_reuse']['contexts']);add(c['native']['context_reuse']['ready'])
    return list(found.values())

def prepare(out,attempt,assets_manifest,shared_W0_root):
    contract,policy=authority()
    require(not out.exists() and not attempt.exists(),'CREATE_ONCE_NEW_ATTEMPT')
    require(not list(LOCAL.glob('*/submission.json')),'NONCE_NOT_REGISTERED')
    old={key:read(path) for key,path in OLD.items()}
    c=copy.deepcopy(old['stock'])
    configs={arm:copy.deepcopy(old['stock' if arm.startswith('BASE_') else 'cake' if arm in ('CAKE','ALPHAEDIT_BLUE') else 'prune']) for arm in ARMS}
    native=dependency_members(old)
    for row in c['assets']:
        s=Path(row['path']).stat()
        require((s.st_size,s.st_ino,s.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']),'PROTECTED_ASSET_STAT')
    records=load_prefix(Path(c['stream']).parent,2000);chunks=list(batches(records))
    require(len({r['case_id'] for r in records})==2000,'EXISTING_EVALUATOR_OCCURRENCE_UNIQUENESS')
    require(sha(c['stream'])==contract['science']['dataset_sha256'],'EXACT_DATASET')
    schedule=ROOT/contract['science']['schedule']
    require(sha(schedule)==contract['science']['schedule_sha256'],'EXACT_SCHEDULE')
    with schedule.open(newline='') as f:schedule_rows=list(csv.DictReader(f))
    require([int(r['case_id']) for r in schedule_rows]==[r['case_id'] for r in records]
            and [int(r['stream_index0']) for r in schedule_rows]==list(range(2000)),'ORDERED_OCCURRENCES')
    model=Path(c['model'])
    revision=subprocess.check_output(['git','-C',str(model),'rev-parse','HEAD'],text=True).strip()
    require(revision=='15ea56dee5df4983c59b2538573817e1667135e2','GPT2_NATIVE_REVISION')
    rt=runtime()
    require((rt['torch'],rt['transformers'])==(c['runtime']['torch'],c['runtime']['transformers']),'RUNTIME_PIN_PRESERVED')
    load_env(c['tracking']['env_file'])
    reference=asset_read(assets_manifest)
    require(reference['status']=='READY','GENERATION_REFERENCE_READY')
    for row in reference['files'].values():
        actual=asset_member(row['path'])
        require(actual['sha256']==row['sha256'] and actual['bytes']==row['bytes'],'GENERATION_REFERENCE_EXACT')
    gen_sources=[member(p) for p in sorted((ROOT/'project/run_scripts/experiment_generation_eval').glob('*.py')) if not p.name.startswith('test_')]
    model_identity=dict(model='gpt2xl',revision=revision,asset_identity=c['model_asset_identity'],
        runtime=dict(torch=rt['torch'],transformers=rt['transformers'],python=platform.python_version()),
        tokenizer=dict(loader='AutoTokenizer',local_model_path=str(model),asset_identity=c['model_asset_identity']),
        precision='FP32_EAGER_AUTOCAST_OFF_TF32_OFF')
    require(shutil.disk_usage(LOCAL.parent).free>=16*1024**3,'NEW_OUTPUT_AND_REFERENCE_RESERVE')
    c.update(schema=1,instruction_id=NONCE,task_id=TASK,attempt=str(attempt),seed=20261002,
        source_configs=configs,source_config_members=[member(p) for p in OLD.values()],
        dependency_sources=native,runtime=rt,model_revision=revision,run_instance=dict(attempt=attempt.name),
        packs=[dict(batch=n,ids=[r['case_id'] for r in current]) for n,current,_ in chunks],
        generation=dict(schema=SCHEMA,profile=PROFILE,eval_seed=EVAL_SEED,
            assets_manifest=str(assets_manifest),assets_manifest_member=member(assets_manifest),
            reference_assets_sha256=reference['identity_sha256'],reference_assets_members=list(reference['files'].values()),
            shared_W0_root=str(shared_W0_root),primary_arm='BASE_MEMIT',model_identity=model_identity,
            source_identity=dict(generation_sources=gen_sources,policy_sha256=GENERATION_POLICY_SHA),
            generation_source_sha=digest(gen_sources),source_members=gen_sources,
            declared_route='UNPADDED_FULL_PREFIX_NO_CACHE',
            plan=dict(new_generation_case_observations_per_arm_max=8500,shared_W0_cases=2000,
                six_arm_case_observations_max=53000,quality_gate=False,ETA='NOT_MEASURED; reference route higher cost')),
        authority=member(ROOT/ENVELOPE),contract=member(ROOT/CONTRACT),generation_policy=member(ROOT/GENERATION_POLICY),
        noCP=True,exact_resume='NOT_AVAILABLE',z_disk_cache=False,ordered_ids_sha256=ORDERED_SHA,
        resources=dict(project_cap=2,task_cap=2,gpu=1,cpu=8,host_mib=65536,hard_host_mib=183296,
            wall='2-00:00:00',collector_cpu=8,collector_host_mib=24576,collector_wall='04:00:00',
            reserve_bytes=16*1024**3,ETA='NOT_MEASURED; wall request only'),
        cancellation_receipt=member(ROOT/'audits/servers/server1/gpt2xl-baselines-fluency-consistency-2k/cancellation.json'),
        broadcast='NO_BROADCAST_NOT_REQUIRED; same-host originals/raw KEEP; compact reference manifest shared')
    c.pop('cpu_preflight',None)
    write(out/'config.json',c)
    write(out/'preparation.json',dict(status='CPU_ASSET_BOUND_NOT_GPU_PASS',source_config_members=c['source_config_members'],
        generation_reference=member(assets_manifest),model_loads=0,native_apply=0,stats_P_recomputed=False,
        scientific_runtime_upgraded=False,large_asset_validation='prior SHA + unchanged size/inode/mtime',
        W0_generation='NEW_ONCE_IN_PRIMARY_ACTUAL_COLD_BASELINE_NOT_YET_MEASURED'))
    return out/'config.json'

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--attempt',type=Path,required=True)
    p.add_argument('--assets-manifest',type=Path,required=True);p.add_argument('--shared-W0-root',type=Path,required=True)
    a=p.parse_args();print(prepare(a.out.resolve(),a.attempt.resolve(),a.assets_manifest.resolve(),a.shared_W0_root.resolve()))
if __name__=='__main__':main()
