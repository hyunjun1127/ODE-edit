"""SH1 author-hparams CF/zsRE two-chain CPU preparation; existing assets only."""
import argparse
from pathlib import Path
from official.experiments.prepare import digest,write_new,file_sha
from official.baselines.fe_author_profile import INSTRUCTION,TASK,METHOD,PROFILE,profile,resolve
from official.runners.fe_author_history import validate_config,tracking_values
from official.tracking.schema import config as tracking_config
from official.evaluation.zsre_query_parity import compare_queries
from transformers import AutoTokenizer
from .fe_history_prepare import runtime
from .common import read,verify,member
from .assets import member as asset_member

LOCAL=Path('/mnt/raid5/janghj/ODE-edit/local/official-baselines/server1')/TASK

def prepare(output):
    output=Path(output).absolute();assert output.is_relative_to(LOCAL) and not output.exists()
    root=LOCAL.parent
    old=read(root/'memit-fe-history-three-model-2k/preparation-r1/configs/llama3.json')
    assets=read(verify(old['assets']))
    for m in assets['model']['members']:asset_member(m['path'],allow_symlink=True,expected_sha=m['sha256'],expected_bytes=m['bytes'])
    for row in assets['C0'].values():verify(row['member'])
    bundle=read(root/'stream-bundle-r1.json');members={d:bundle['datasets'][d]['stream'] for d in ('cf','zsre')}
    records={d:read(verify(m)) for d,m in members.items()}
    records={d:(r['records'] if isinstance(r,dict) else r) for d,r in records.items()}
    tok=AutoTokenizer.from_pretrained(assets['model']['snapshot'],local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='right'
    proof=compare_queries(tok,records['zsre'],model_family='llama3')
    assert proof['requests']==2000 and proof['input_mismatches']==proof['target_mismatches']==0
    pp=output/'zsre-query-parity.json';write_new(pp,proof)
    configs=[]
    for dataset in ('cf','zsre'):
        assert len(records[dataset])==2000 and [r['occurrence_index'] for r in records[dataset]]==list(range(1,2001))
        c=dict(old);c.pop('config_sha256');c.pop('generation_schedule')
        c.update(schema='official-fe-author-history-v1',instruction_id=INSTRUCTION,task_id=TASK,server='server1',dataset=dataset,
            model='llama3',method=METHOD,arm=dataset+'-MEMIT_FE_HISTORY_FE_AUTHOR_HPARAMS',hparams=profile('llama3')['hparams'],
            author_profile_sha256=file_sha(PROFILE),stream_member=members[dataset],stream_sha256=digest(records[dataset]),
            ordered_case_ids_sha256=digest([r['case_id'] for r in records[dataset]]),
            storage_min_free_bytes=64*(1<<30),runtime=runtime(),output=str(output/'runs'/dataset),
            checkpoint_policy='W0_THEN_LATEST_EACH_BATCH_KEEP_W20',future_consumer_pending=dataset=='cf',
            author_only_changes=['clamp_norm_factor','v_num_grad_steps'])
        if dataset=='cf':c['generation_schedule']='DEFERRED_CHECKPOINT_EVALUATION'
        else:c['zsre_query_parity']=member(pp)
        c['config_sha256']=digest(c);validate_config(c);tracking_config(tracking_values(c,{'source_commit':'a'*40}))
        hp=resolve('llama3');assert (hp.clamp_norm_factor,hp.v_num_grad_steps)==(.75,35)
        path=output/'configs'/f'{dataset}.json';write_new(path,c);configs.append(member(path))
    result=dict(instruction=INSTRUCTION,server='server1',model='llama3',datasets=['cf','zsre'],configs=configs,
        author_profile=member(PROFILE),zsre_query_parity=member(pp),GPU_qualification='NOT_RUN_USER_DISABLED')
    write_new(output/'preparation.json',result);print(result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();prepare(a.output)
