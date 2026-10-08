"""Exact six native zsRE configs; existing local assets only, CPU preparation."""
import argparse
import copy
from pathlib import Path
from official.experiments.prepare import build_matrix,digest,load_plan,write_new
from .common import read,member,require,validate_config,LOCAL_ROOT,ZSRE_PLAN
from .submit import ZSRE_METHODS


def prepare(output):
    root=Path(output).absolute()
    require(root.is_relative_to(LOCAL_ROOT) and not root.exists(),'FRESH_ZSRE_SIX_NAMESPACE')
    parent=LOCAL_ROOT/'alpha-sphere-cf-r1'
    assets=copy.deepcopy(read(parent/'assets.json'))
    from .assets import verify_manifest
    verify_manifest(assets,verify_preparation_source=False)
    require(member(assets['projector']['path'])['sha256']==assets['projector']['sha256'],'PROJECTOR_SHA_CHANGED')
    assets['assignment']['methods']=list(ZSRE_METHODS)
    assets['parent_asset_manifest']=member(parent/'assets.json')
    assets.pop('assets_sha256');assets['assets_sha256']=digest(assets)
    write_new(root/'assets.json',assets)
    streams=copy.deepcopy(read(parent/'streams.json'));streams['assets_sha256']=assets['assets_sha256']
    streams['parent_stream_bundle']=member(parent/'streams.json');write_new(root/'streams.json',streams)
    contract,profiles=load_plan();configs=[]
    for row in build_matrix(contract,profiles):
        if row['model']!='llama3' or row['dataset']!='zsre' or row['method'] not in ZSRE_METHODS:continue
        cfg=dict(copy.deepcopy(row),schema='official-server1-runtime-v1',zsre_six=True,
            scope_override='USER-DIRECT-SERVER1-ZSRE-SIX-20261009',assets_member=member(root/'assets.json'),
            stream_bundle_member=member(root/'streams.json'),qualification_plan=ZSRE_PLAN,
            zsre_W0_output=str(root/'runs'/'base-w0'),
            qualification_outputs={m:str(root/'runs'/('qualification-'+m.lower())) for m in ZSRE_METHODS},
            tracking=dict(module='official.tracking.client',entity='wkdguswns2256',project='layer allocation',
                env_file='/mnt/raid5/janghj/ODE-edit/servers/local/wandb.env',attempt='zsre-six-r1'),
            actual_GPU_qualification=False,actual_online_validation=False,automatic_retry=False,protected_old_jobs_keep=True)
        cfg.pop('config_sha256',None);cfg['config_sha256']=digest(cfg);validate_config(cfg)
        path=root/'configs'/(row['method'].lower()+'.json');write_new(path,cfg)
        configs.append(dict(method=row['method'],dataset='zsre',member=member(path)))
    require({v['method'] for v in configs}==set(ZSRE_METHODS),'EXACT_ZSRE_SIX_CONFIGS')
    result=dict(configs=configs,output_root=str(root/'runs'),assets_member=member(root/'assets.json'),
        stream_bundle_member=member(root/'streams.json'),actual_GPU_qualification=False,submitted_jobs=[])
    write_new(root/'preparation.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    result=prepare(p.parse_args().output);print({'configs':len(result['configs']),'GPU_qualification':False})
