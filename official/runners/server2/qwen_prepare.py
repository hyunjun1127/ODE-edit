"""Exact SH4 scientific metadata to SH2 local assets; no download or model load."""
import argparse
import json
from pathlib import Path
from official.experiments.prepare import file_sha, prepare_stream, write_new
from official.runners.server2.qwen_plan import rows

ROOT = Path(__file__).resolve().parents[3]
HANDOFF = ROOT/'audits/servers/server4/qwen-migration-20261009'
LOCAL = Path('/mnt/raid5/janghj/ODE-edit')

def read(p): return json.loads(Path(p).read_text())

def prepare(output):
    output=Path(output).resolve()
    manifest=read(HANDOFF/'handoff-manifest.json')
    assert file_sha(HANDOFF/'cessation.json') == manifest['cessation_sha256']
    cessation=read(HANDOFF/'cessation.json')
    assert cessation['status']=='CEASED_READY_FOR_SOURCE_HANDOFF_NOT_DESTINATION_SUBMITTED'
    assert not cessation['successful_terminal_main_ids'] and not cessation['W0_observed']
    for row in manifest['members']:
        p=HANDOFF/row['path']; assert p.stat().st_size==row['bytes'] and file_sha(p)==row['sha256']
    for row in rows():
        config=read(HANDOFF/'handoff/configs'/f"{row['logical_main_row']}.json")
        assert config==row['config']
        write_new(output/'configs'/f"{row['logical_main_row']}.json",config)
    original=read(HANDOFF/'handoff/server4-asset-paths-reference-only.json')
    assets=dict(original)
    for key in ('cf_source','zsre_source','model_snapshot','runtime_python'):
        assets[key]=original[key].replace('/data/janghj/','/mnt/raid5/janghj/',1)
    assets['covariance']={k:dict(v,path=v['path'].replace('/data/janghj/','/mnt/raid5/janghj/',1)) for k,v in original['covariance'].items()}
    assets['projector']=dict(original['projector'],path=original['projector']['path'].replace('/data/janghj/','/mnt/raid5/janghj/',1))
    ref=LOCAL/'local/gptj-baselines-fluency-consistency-2k/inputs/reference-r1'
    original_manifest=ref/'reference-ready-r1/manifest.json'
    generation=read(original_manifest)
    assert file_sha(original_manifest)=='6d9a713ab7eaa10871f277e10a3974e0bd5b140c265be80052c279f258876ca8'
    generation['producer_manifest']={'path':str(original_manifest),'sha256':file_sha(original_manifest)}
    generation['producer_file_paths']={k:v['path'] for k,v in generation['files'].items()}
    for name,row in generation['files'].items(): row['path']=str(ref/'download-r1'/name)
    write_new(output/'generation-local.json',generation)
    assets.update(generation_reference_manifest=str(output/'generation-local.json'),
        nltk_data='/mnt/raid5/janghj/nltk_data', output_root=str(output),
        wandb_env=str(LOCAL/'servers/local/wandb.env'),
        wandb_sdk_python=str(LOCAL/'local/wandb-setup/sdk-env/bin/python'))
    write_new(output/'assets.candidate.json',assets)
    for dataset in ('cf','zsre'):
        lock=read(HANDOFF/'handoff'/f'{dataset}-stream.lock.json')
        prepare_stream(assets[dataset+'_source'],dataset,output/'streams',expected_source_sha=lock['source_sha256'])
        assert read(output/'streams'/f'{dataset}-stream.lock.json')==lock
    write_new(output/'handoff-received.json',dict(source_owner_manifest_sha256=file_sha(HANDOFF/'handoff-manifest.json'),
        cessation_sha256=file_sha(HANDOFF/'cessation.json'),source_owner='server4',destination='server2',
        cells=12,CP_or_model_transfers=0,scientific_configs_unchanged=True,job_ids=[]))
    print(json.dumps({'status':'INPUT_FILES_PREPARED_NOT_SUBMITTED','output':str(output)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);prepare(p.parse_args().output)
