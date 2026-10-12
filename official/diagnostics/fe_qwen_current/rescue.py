"""Explicit new USER rescue contract; old four-arm source/attempts unchanged.

Process-local bindings reuse the native preparation/launcher and final benchmark
machinery. They are scoped to this entrypoint and restored on return; no files
in an existing attempt are patched and no job is submitted.
"""
import argparse
import copy
import math
from contextlib import contextmanager,ExitStack
from pathlib import Path
from unittest.mock import patch
from official.experiments.prepare import digest,write_new
from official.runners.server1.common import read,verify,member,require
from official.diagnostics.fe_qwen_current import contract,prepare as prep,launch as launcher,benchmark
from official.diagnostics.fe_qwen_current.patch import native_source as old_native,main_source as old_main
from official.evaluation.factual import validate_retained_observation
from official.evaluation.reduce import counterfact

SPEC=copy.deepcopy(contract.SPEC)
SPEC.update(protocol='rescue-L2-100-v1',L2_arms=[100],promotable_L2=[100],
    ranking='single_candidate_absolute_W20_gate',veto='no_best_failed_promotion',
    control='prior_four_arm_contract_SUPERSEDED_USER_EARLY_STOP',
    superseded_job='janghj-feq-l210-20261012-r2',
    superseded_uid='c3ffd8c8-f465-468c-8da4-64121d4d8d02',
    early_stop=dict(Efficacy=80,Generalization=70,Specificity=65),
    failure_policy='QUALITY_EARLY_STOP_is_not_numeric_failure; incomplete_or_technical_blocks_promotion')


def quality_stop(summary):
    values={k:summary[k] for k in SPEC['early_stop']}
    require(summary['requests']==100 and all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=100 for v in values.values()),'RESCUE_CURRENT_METRICS')
    return {k:v for k,v in values.items() if v<SPEC['early_stop'][k]}


def select_complete(metrics):
    require(metrics['requests']==2000 and metrics['committed_batches']==20,'RESCUE_W20_REQUIRED')
    vals=[metrics[k] for k in ('Efficacy','Generalization','Specificity','Score')]
    require(all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=100 for v in vals),'RESCUE_FINITE_METRICS')
    score=0 if min(vals[:3])==0 else 3/sum(1/v for v in vals[:3])
    require(abs(score-vals[3])<1e-9,'RESCUE_HARMONIC')
    passed=all(metrics[k]>=v for k,v in SPEC['gate'].items())
    return dict(status='SELECTED' if passed else 'NO_PROMOTION',selected_L2=100 if passed else None,
        contract_sha256=digest(SPEC),scientific_recovery_claim=False)


def tracking_config(c,source):
    v=contract.tracking_config(c,source)
    v.update(task_id='fe-qwen-current-z-rescue-L2-100-20261012',baseline='FE-author-hparam-tuned-rescue-L2-100',
        evaluation_profile='heldout-rescue-not-main',arm='heldout4000-6000-L2-100')
    if c.get('stage')=='final_benchmark':
        v.update(task_id='fe-qwen-current-z-rescue-selected-final-20261012',
            evaluation_profile='rescue-selected-native-current-z-final-first2000',arm='final-'+c['dataset']+'-L2-100')
    return v


def native_source(text):
    return old_native(text).replace('fe_qwen_current.observer import','fe_qwen_current.rescue_observer import')


def main_source(text):
    return old_main(text).replace('fe_qwen_current.observer import','fe_qwen_current.rescue_observer import')


def collect(root,*,write_result=True):
    root=Path(root);lock=read(root/'lock.json');frozen=read(verify(lock['selection_lock']))
    require(frozen['spec']==SPEC and frozen['spec_sha256']==digest(SPEC) and len(lock['configs'])==1,'RESCUE_LOCK')
    cm=lock['configs'][0];c=read(verify(cm));out=Path(c['output'])
    require(c['L2']==100 and c.get('stage')!='final_benchmark' and c['source']==lock['source_commit']
        and c['selection_contract_sha256']==digest(SPEC)
        and c['config_sha256']==digest({k:v for k,v in c.items() if k!='config_sha256'}),'RESCUE_CONFIG')
    require(not (out/'QUALITY_EARLY_STOP.json').exists() and not (out/'FAILURE.json').exists(),'RESCUE_STOPPED_NO_PROMOTION')
    done=read(out/'COMPLETE.json');ident=done['identity'];rows=read(verify(c['stream']))
    require(done['status']=='W20_COMPLETE' and (done['edits'],done['batches'])==(2000,20)
        and ident['source']==c['source'] and ident['config_sha256']==c['config_sha256']
        and ident['selection_contract_sha256']==digest(SPEC) and ident['L2']==100
        and ident['z_reference']=='batch_entry' and ident['split']=='heldout_fixed10k_4000_6000','RESCUE_COMPLETE')
    verify(done['checkpoint']);audits={}
    for b in range(21):
        commit=read(out/'commits'/f'B{b:02d}.json')
        require(commit['identity']==ident and (commit['batch'],commit['edits'])==(b,b*100),'RESCUE_COMMIT')
        labels=['W0'] if b==0 else [f'B{b:02d}-current']
        if b in SPEC['milestones']:labels.append(f'B{b:02d}-all-seen')
        require(len(labels)==len(commit['raw']),'RESCUE_ENDPOINTS')
        for label,rm in zip(labels,commit['raw']):
            raw=read(verify(rm));receipt=read(out/'evaluations'/f'{label}.json');plan=read(verify(receipt['queries']))
            ei=raw['identity'].get('external_identity',raw['identity'])
            require(all(ei[k]==v for k,v in ident.items()) and receipt['identity']==ident and receipt['raw']==rm,'RESCUE_RAW_IDENTITY')
            expected=rows if b==0 else rows[:b*100] if label.endswith('all-seen') else rows[(b-1)*100:b*100]
            require([r['case_id'] for r in raw['cases']]==[r['case_id'] for r in expected],'RESCUE_RAW_ORDER')
            validation=validate_retained_observation(raw,'cf',input_signature=plan);values=counterfact(raw['cases'])
            require(all(values[k]==raw['summary'][k] for k in ('Efficacy','Generalization','Specificity','Score','requests')),'RESCUE_REDUCTION')
            if label.endswith('current'):require(not quality_stop(values),'RESCUE_MISSED_QUALITY_STOP')
            audits[label]=dict(raw=rm,queries=receipt['queries'],validation=validation,metrics=values)
    require(done['endpoint']==audits['B20-all-seen']['raw'],'RESCUE_W20_RAW')
    pointer=read(out/'checkpoints/latest.json')
    require(pointer['sha256']==done['checkpoint']['sha256'] and pointer['bytes']==done['checkpoint']['bytes']
        and pointer['batch']==20 and pointer['final_W20'] is True and pointer['identity']==ident,'RESCUE_CP_POINTER')
    result=dict(audits['B20-all-seen']['metrics'],committed_batches=20,status='W20_COMPLETE')
    receipt=dict(protocol=SPEC['protocol'],selection=select_complete(result),results={'100':result},endpoints=audits,
        checkpoint=done['checkpoint'],selection_lock=member(root/'selection-lock.json'),source=lock['source_commit'],
        benchmark_submitted=False,no_GPU_or_model_calls=True)
    if write_result:write_new(root/'selection-result.json',receipt)
    return receipt


def validated_selection(m):
    saved=read(verify(m));actual=collect(Path(m['path']).parent,write_result=False)
    require(saved==actual and actual['selection']['status']=='SELECTED' and actual['selection']['selected_L2']==100,'RESCUE_SELECTED_PROOF_REQUIRED')
    return actual


@contextmanager
def bindings():
    # Only this new process/entrypoint adopts the new prospective contract.
    with ExitStack() as s:
        for module in (prep,launcher,benchmark):s.enter_context(patch.object(module,'SPEC',SPEC))
        s.enter_context(patch.object(prep,'native_source',native_source))
        s.enter_context(patch.object(prep,'main_source',main_source))
        s.enter_context(patch.object(prep,'tracking_config',tracking_config))
        s.enter_context(patch.object(benchmark,'validated_selection',validated_selection))
        yield


def prepare(output,inputs):
    b=read(inputs);require(b['server']=='rent','RENT_ONLY_NO_SH_SUBMISSION')
    with bindings():return prep.prepare(output,inputs)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','launch','collect','final-collect']);p.add_argument('first');p.add_argument('second',nargs='?');a=p.parse_args()
    if a.mode=='prepare':print(prepare(a.first,a.second))
    elif a.mode=='collect':print(collect(a.first))
    elif a.mode=='final-collect':
        with bindings():print(benchmark.collect_final(a.first))
    else:
        with bindings():launcher.launch(a.first,a.second)
