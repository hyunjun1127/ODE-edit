"""CPU-only exact AlphaEdit/SPHERE CF preparation; no download or recomputation."""
import argparse
import copy
from pathlib import Path
import torch
from official.experiments.prepare import build_matrix, load_plan, digest, write_new
from .common import read, member, validate_config, DEFERRED_W20, CF_CHECKPOINT_AUTHORITY
from .prepare_runtime import QUALIFICATION_PLAN


def prepare(output):
    root = Path('/mnt/raid5/janghj/ODE-edit/local/official-baselines/server1')
    output = Path(output).absolute()
    if not output.is_relative_to(root) or output.exists():
        raise ValueError('FRESH_OWN_PROJECTED_PREPARATION_REQUIRED')
    assets = copy.deepcopy(read(root/'assets-r1.json'))
    from .assets import verify_manifest
    verify_manifest(assets, verify_preparation_source=False)
    original = member(root/'assets-r1.json')
    path = Path(assets['protected_projector']['path'])
    projector = member(path)
    if projector['sha256'] != '6d356468c6408dca694c1907ffde31cddb99f7e67fa2e74502910d5afbede5ec':
        raise ValueError('PREPARED_PROJECTOR_SHA_CHANGED')
    value = torch.load(path, weights_only=True, mmap=True, map_location='cpu')
    if not torch.is_tensor(value) or value.dtype != torch.float32 or list(value.shape) != [5,14336,14336]:
        raise ValueError('PROJECTOR_PHYSICAL_SHAPE_DTYPE')
    for layer in value:
        if not torch.isfinite(layer).all().item():
            raise ValueError('PROJECTOR_NONFINITE')
    del value
    assets['assignment']['methods'] = ['ALPHAEDIT','SPHERE']
    assets['projector'] = dict(projector, physical_layers=[4,5,6,7,8], threshold=.02,
        validation='EXISTING_ASSET_FULL_SHA_CPU_FP32_SHAPE_FINITE', recomputed=False)
    assets['parent_asset_manifest'] = original
    assets.pop('assets_sha256')
    assets['assets_sha256'] = digest(assets)
    write_new(output/'assets.json', assets)
    streams = copy.deepcopy(read(root/'stream-bundle-r1.json'))
    streams['assets_sha256'] = assets['assets_sha256']
    # Original exact record members are retained; only binding/provenance changes.
    streams['parent_stream_bundle'] = member(root/'stream-bundle-r1.json')
    write_new(output/'streams.json', streams)
    contract,profiles = load_plan()
    configs=[]
    for row in build_matrix(contract,profiles):
        if row['model']!='llama3' or row['dataset']!='cf' or row['method'] not in ('ALPHAEDIT','SPHERE'):
            continue
        config=copy.deepcopy(row)
        config.update(schema='official-server1-runtime-v1', assets_member=member(output/'assets.json'),
            stream_bundle_member=member(output/'streams.json'), qualification_plan=QUALIFICATION_PLAN,
            qualification_outputs={m:str(output/'runs'/('qualification-'+m.lower())) for m in ('ALPHAEDIT','SPHERE')},
            projected_CF_addition=True, cf_W20_generation=DEFERRED_W20, scope_override=CF_CHECKPOINT_AUTHORITY,
            tracking=dict(module='official.tracking.client',entity='wkdguswns2256',project='layer allocation',
                          env_file='/mnt/raid5/janghj/ODE-edit/servers/local/wandb.env',attempt='projected-cf-r1'),
            actual_GPU_qualification=False, actual_online_validation=False,automatic_retry=False,
            protected_old_jobs_keep=True)
        config['evaluation']['generation']['edited_endpoints']=[]
        config['evaluation']['generation']['deferred_to_checkpoint']=20
        config.pop('config_sha256',None)
        config['config_sha256']=digest(config)
        validate_config(config)
        target=output/'configs'/(row['method'].lower()+'.json')
        write_new(target,config)
        configs.append(dict(method=row['method'],dataset='cf',member=member(target)))
    if len(configs)!=2: raise ValueError('PROJECTED_EXACT_TWO_CONFIGS')
    result=dict(configs=configs,output_root=str(output/'runs'),assets_member=member(output/'assets.json'),
                stream_bundle_member=member(output/'streams.json'),GPU_qualification=False,submitted_jobs=[])
    write_new(output/'preparation.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    result=prepare(p.parse_args().output)
    print({'methods':[v['method'] for v in result['configs']],'GPU_qualification':False})
