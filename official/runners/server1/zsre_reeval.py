"""Restore immutable Llama W20 selected weights; one GH public-query evaluation."""
import argparse
import math
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from official.evaluation import zsre_paper
from official.evaluation.zsre_query_parity import compare_queries
from official.experiments.prepare import digest,file_sha,write_new
from official.experiments.checkpoint import rng_snapshot
from official.tracking import init,official_zsre_metrics
from official.tracking.schema import config as validate_tracking
from .common import read,member,verify,load_model,rng_content
from .native import tensor_sha
from .zsre_reeval_inventory import INSTRUCTION,TASK,LOCAL
from .zsre_reeval_restore import restore

def prepare(inventory_path,output):
    output=Path(output).absolute();assert output.is_relative_to(LOCAL) and not output.exists()
    inv=read(inventory_path);assert len(inv['rows'])==6 and all(r['status']=='FINAL_W20_CHECKPOINT_FULL_SHA_VERIFIED' for r in inv['rows'])
    assets=read(verify(inv['base']['manifest']))
    tok=AutoTokenizer.from_pretrained(assets['model']['tokenizer_path'],local_files_only=True)
    tok.pad_token=tok.eos_token;tok.padding_side='right'
    records=read(verify(inv['rows'][0]['stream']));assert len(records)==2000
    proof=compare_queries(tok,records,model_family='llama3')
    assert proof['requests']==2000 and proof['input_mismatches']==proof['target_mismatches']==0
    parity=output/'query-parity.json';write_new(parity,proof)
    configs=[]
    reviewed=read(Path(__file__).resolve().parents[3]/'audits/servers/server1/official-baselines-20261008/results-review-20261009/results.json')
    for row in inv['rows']:
        old=read(verify(row['config']))
        previous=next(r for r in reviewed['rows'] if r['job_id']==row['job_id'])
        verify(previous['endpoint'])
        c=dict(schema='official-server1-zsre-eval-only-v1',instruction=INSTRUCTION,original=row,
               previous_endpoint=previous['endpoint'],
               assets=inv['base']['manifest'],stream=row['stream'],parity=member(parity),
               evaluator=member(zsre_paper.__file__),microbatch=16,model='llama3',dataset='zsre',
               method=row['method'],output=str(output/'runs'/row['method'].lower()),
               tracking_env_file=old['tracking']['env_file'],no_edits=True,no_generation=True,no_W0=True)
        c['config_sha256']=digest(c);path=output/'configs'/(row['method'].lower()+'.json');write_new(path,c);configs.append(member(path))
    value=dict(instruction=INSTRUCTION,configs=configs,parity=member(parity),inventory=member(inventory_path))
    write_new(output/'preparation.json',value);return value

def tracking_values(c,source):
    old=c['original']
    return validate_tracking(dict(server='server1',task_id=TASK,arm=old['method'],attempt='reeval-r1-'+old['job_id'],
        source_sha=source,config_sha=c['config_sha256'],model='llama3',model_family='llama',writer=old['method'].lower(),
        baseline=old['method'],role='eval_only',metric_schema='official-baselines-scalar-v1',dataset='zsre',
        instruction_id=INSTRUCTION,evaluation_profile='zsre-public-query-W20-only-v1',
        checkpoint_sha256=old['checkpoint']['sha256'],evaluator_sha256=c['evaluator']['sha256'],
        stream_sha256=c['stream']['sha256'],tokenizer_sha256=old['identity']['tokenizer_sha256'],source_run_id=old['job_id']))

def run(config_path,lock_path):
    c=read(config_path);lock=read(lock_path)
    assert c['instruction']==INSTRUCTION and c['schema']=='official-server1-zsre-eval-only-v1'
    assert digest({k:v for k,v in c.items() if k!='config_sha256'})==c['config_sha256']
    assert member(config_path) in lock['configs'] and Path(__file__).resolve().is_relative_to(Path(lock['source_directory']))
    for m in lock['source_members']:verify(m)
    assert file_sha(zsre_paper.__file__)==c['evaluator']['sha256']
    proof=read(verify(c['parity']));assert proof['status']=='PASS_CPU_QUERY_ONLY' and proof['requests']==2000
    records=read(verify(c['stream']));assert digest(records)==proof['stream_sha256']
    assets=read(verify(c['assets']));old=c['original']
    for m in assets['model']['weights']+list(assets['model']['tokenizer_files'].values()):
        path=Path(m['path']);s=path.stat()
        assert str(path.resolve())==m['real_path']
        assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==(m['device'],m['inode'],m['bytes'],m['mtime_ns'],m['ctime_ns'])
    verify(old['checkpoint']);verify(old['pointer']);verify(old['terminal']);verify(old['config'])
    cfg=tracking_values(c,lock['source_commit']);output=Path(c['output']);output.mkdir(parents=True,exist_ok=True)
    assert not (output/'COMPLETE.json').exists(),'NO_DUPLICATE_EVALUATION'
    tracker=init(env_file=c['tracking_env_file'],spool=output/'tracking',config=cfg)
    exit_code=1
    def log(v):
        if tracker.log(v) is False:raise ValueError('EVAL_ONLY_TRACKING_REJECTED')
    try:
        model,tok=load_model(assets)
        restore_receipt=restore(model,old);write_new(output/'restore.json',restore_receipt)
        # Exact selected state+RNG evidence, no edit engine or history forward.
        hp=old['hparams'];selected={n:p for n,p in model.named_parameters() if any(hp['rewrite_module_tmp'].format(l) in n for l in hp['layers'])}
        before={n:tensor_sha(p) for n,p in selected.items()};rng_before=rng_content(rng_snapshot())
        binding=dict(instruction=INSTRUCTION,original_job=old['job_id'],original_checkpoint=old['checkpoint'],
                     original_identity=old['identity'],new_source=lock['source_commit'],config_sha256=c['config_sha256'],
                     evaluator_sha256=c['evaluator']['sha256'],query_parity_sha256=c['parity']['sha256'])
        previous=[0.]
        def progress(value):
            now=time.monotonic()
            if value['completed_queries']==value['total_queries'] or now-previous[0]>=15:
                log({'eval_progress/'+k:v for k,v in value.items()});previous[0]=now
        result=zsre_paper.evaluate(model,tok,records,model_family='llama3',batch_size=c['microbatch'],device='cuda:0',identity=binding,progress=progress)
        assert result['query_sha256']==proof['query_sha256'] and result['token_denominators']==proof['token_denominators']
        assert result['summary']['requests']==2000 and len(result['cases'])==2000
        assert before=={n:tensor_sha(p) for n,p in selected.items()} and rng_before==rng_content(rng_snapshot())
        verify(old['checkpoint']);verify(old['pointer'])
        path=output/'W20-evaluation.json';write_new(path,result)
        log(official_zsre_metrics(result['summary'],config_values=cfg,endpoint='all_seen/post',edits=2000,post_state_edits=2000))
        write_new(output/'COMPLETE.json',dict(instruction=INSTRUCTION,config=member(config_path),source=lock['source_commit'],
            original_job=old['job_id'],original_checkpoint=old['checkpoint'],endpoint=member(path),requests=2000,
            weights_unchanged=True,RNG_unchanged=True,checkpoint_unchanged=True,edits=0,generation=0))
        exit_code=0
    except BaseException as e:
        write_new(output/'FAILURE.json',dict(type=type(e).__name__,error=str(e),original_job=old['job_id'],checkpoint_keep=True));raise
    finally:tracker.finish(exit_code=exit_code,timeout=45)

def collect(preparation,output,lock_path):
    lock=read(lock_path)
    for m in lock['source_members']:verify(m)
    rows=[]
    for cm in read(preparation)['configs']:
        c=read(verify(cm));folder=Path(c['output']);row=dict(method=c['method'],original_job=c['original']['job_id'])
        try:
            done=read(folder/'COMPLETE.json');assert done['config']==cm
            raw=read(verify(done['endpoint']));assert len(raw['cases'])==raw['summary']['requests']==2000
            assert done['source']==lock['source_commit']==raw['identity']['new_source']
            assert raw['identity']['original_checkpoint']==c['original']['checkpoint'] and raw['identity']['config_sha256']==c['config_sha256']
            assert raw['query_sha256']==read(verify(c['parity']))['query_sha256']
            assert digest([r['case_id'] for r in raw['cases']])==c['original']['ordered_case_ids_sha256']
            scores={};counts={}
            previous=read(verify(c['previous_endpoint']));assert len(previous['cases'])==2000
            old_metrics={}
            for group,label in [('rewrite','Efficacy'),('paraphrase','Generalization'),('neighborhood','Specificity')]:
                rates=[];tokens=correct=0
                for case in raw['cases']:
                    obs=case[group+'_observations'];bits=[r['predicted_token_id']==r['target_token_id'] for r in obs]
                    assert bits and bits==case[group+'_prompts_correct'] and all(r['correct']==b for r,b in zip(obs,bits))
                    rates.append(sum(bits)/len(bits));tokens+=len(bits);correct+=sum(bits)
                scores[label]=100*math.fsum(rates)/2000;counts[group]=dict(requests=2000,tokens=tokens,correct=correct)
                assert abs(scores[label]-raw['summary'][label])<1e-10 and tokens==raw['token_denominators'][group]
                key=group+'_prompts_correct'
                old_metrics[label]=100*math.fsum(sum(case[key])/len(case[key]) for case in previous['cases'])/2000
            row.update(status='W20_EVAL_ONLY_CPU_REDUCED',metrics=scores,counts=counts,endpoint=done['endpoint'],
                       source=done['source'],config=cm,checkpoint=c['original']['checkpoint'],
                       previous_metrics=old_metrics,delta_pp={k:scores[k]-old_metrics[k] for k in scores},
                       previous_endpoint=c['previous_endpoint'],old_query_not_native_equivalent=True)
        except Exception as error:row.update(status='INCOMPLETE_OR_FAILED',error=str(error))
        rows.append(row)
    write_new(output,dict(instruction=INSTRUCTION,rows=rows,all_complete=all(r['status']=='W20_EVAL_ONLY_CPU_REDUCED' for r in rows)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','collect']);p.add_argument('--inventory');p.add_argument('--output');p.add_argument('--config');p.add_argument('--lock');p.add_argument('--preparation');a=p.parse_args()
    if a.mode=='prepare':prepare(a.inventory,a.output)
    elif a.mode=='collect':collect(a.preparation,a.output,a.lock)
    else:run(a.config,a.lock)
