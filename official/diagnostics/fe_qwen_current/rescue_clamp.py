"""Prospective single L2=100/clamp=.5 rescue; no old proof or CP reuse."""
import argparse
import copy
from contextlib import contextmanager,ExitStack
from pathlib import Path
from unittest.mock import patch
import torch
from official.experiments.prepare import digest,write_new
from official.runners.server1.common import read,verify,member,require
from official.diagnostics.fe_qwen_current import rescue as old,prepare as prep,launch as launcher,benchmark

SPEC=copy.deepcopy(old.SPEC)
SPEC.update(protocol='rescue-L2-100-clamp-0p5-v1',clamp=.5,arm_suffix='-clamp-0p5',
    control='previous_L2_100_clamp1_QUALITY_EARLY_STOP_B9_not_selected',
    superseded_job='janghj-feq-r100-20261012-r1',superseded_uid='072c9d96-58a6-4e4d-b9f7-0f5e372bda2f')


def tracking_config(c,source):
    v=old.tracking_config(c,source)
    v.update(task_id='fe-qwen-current-z-rescue-L2-100-clamp-0p5-20261012',
        baseline='FE-author-hparam-tuned-L2-100-clamp-0p5',
        evaluation_profile='heldout-rescue-L2-100-clamp-0p5-not-main',
        arm='heldout4000-6000-L2-100-clamp-0p5')
    if c.get('stage')=='final_benchmark':
        v.update(task_id='fe-qwen-current-z-rescue-clamp-0p5-selected-final-20261012',
            evaluation_profile='rescue-clamp-0p5-selected-current-z-final',arm='final-'+c['dataset']+'-L2-100-clamp-0p5')
    return v


def native_source(text):
    return old.native_source(text).replace('fe_qwen_current.rescue_observer import','fe_qwen_current.rescue_clamp_observer import')


def main_source(text):
    return old.main_source(text).replace('fe_qwen_current.rescue_observer import','fe_qwen_current.rescue_clamp_observer import')


def validate_profile(config):
    require(config['L2']==100 and config.get('clamp_norm_factor')==.5
        and config.get('author_hparam_profile')==SPEC['protocol']
        and config['selection_contract_sha256']==digest(SPEC),'CLAMP_RESCUE_PROFILE')


@contextmanager
def bindings():
    with old.bindings(),ExitStack() as s:
        for module in (old,prep,launcher,benchmark):s.enter_context(patch.object(module,'SPEC',SPEC))
        s.enter_context(patch.object(prep,'native_source',native_source))
        s.enter_context(patch.object(prep,'main_source',main_source))
        s.enter_context(patch.object(prep,'tracking_config',tracking_config))
        s.enter_context(patch.object(benchmark,'validated_selection',validated_selection))
        yield


def collect(root,*,write_result=True):
    root=Path(root);lock=read(root/'lock.json')
    require(len(lock['configs'])==1,'ONE_CLAMP_CANDIDATE')
    c=read(verify(lock['configs'][0]));validate_profile(c);out=Path(c['output'])
    # Quality stop is authoritative even if Hydra returns process exit1.
    require(not (out/'QUALITY_EARLY_STOP.json').exists(),'QUALITY_EARLY_STOP_NO_PROMOTION')
    with bindings():result=old.collect(root,write_result=False)
    runtime=read(out/'runtime.json');done=read(out/'COMPLETE.json')
    require(runtime['identity']==done['identity'] and runtime['resolved']['llms']['clamp_norm_factor']==.5
        and runtime['resolved']['algs']['L2']==100
        and done['identity']['clamp_norm_factor']==.5
        and done['identity']['author_hparam_profile']==SPEC['protocol'],'CLAMP_RUNTIME_IDENTITY')
    cp=result['checkpoint'];payload=torch.load(verify(cp),map_location='cpu',mmap=True,weights_only=False)
    require(payload['identity']==done['identity'] and payload['batch']==20 and payload['cursor']['edits']==2000
        and payload['cursor']['native_config']['llms']['clamp_norm_factor']==.5
        and payload['cursor']['native_config']['algs']['L2']==100,'CLAMP_CHECKPOINT_IDENTITY')
    del payload;verify(cp)
    result['selection']['selected_clamp_norm_factor']=.5 if result['selection']['status']=='SELECTED' else None
    result['author_hparam_profile']=SPEC['protocol']
    if write_result:write_new(root/'selection-result.json',result)
    return result


def validated_selection(m):
    saved=read(verify(m));actual=collect(Path(m['path']).parent,write_result=False)
    require(saved==actual and actual['selection']['status']=='SELECTED'
        and actual['selection']['selected_L2']==100 and actual['selection']['selected_clamp_norm_factor']==.5,
        'SELECTED_L2_100_CLAMP_0P5_REQUIRED')
    return actual


def prepare(output,inputs):
    require(read(inputs)['server']=='rent','ROOT_RENT_ONLY')
    with bindings():return prep.prepare(output,inputs)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','launch','collect','final-collect']);p.add_argument('first');p.add_argument('second',nargs='?');a=p.parse_args()
    if a.mode=='prepare':print(prepare(a.first,a.second))
    elif a.mode=='collect':print(collect(a.first))
    elif a.mode=='final-collect':
        with bindings():print(benchmark.collect_final(a.first))
    else:
        with bindings():launcher.launch(a.first,a.second)
