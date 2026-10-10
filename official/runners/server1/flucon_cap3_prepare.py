"""Bounded read-only eligibility and dedup for latest cap3 evaluation authority."""
import argparse
from pathlib import Path
from official.experiments.prepare import digest,write_new
from official.baselines import registry
from official.evaluation.generation.assets import load_assets
from .common import read,member,verify
from .assets import member as asset_member
from .fe_history_prepare import runtime
from . import flucon_eval as e
from .flucon_submit import CAP3_AUTHORITY,CAP3_LOCAL

def prepare(output):
    output=Path(output).absolute();assert output.is_relative_to(CAP3_LOCAL) and not output.exists()
    old=e.BASE/'flucon-paper-scale-20261010/registration-r1/submission.json'
    submission=read(old);dedup=[member(old)];registered={}
    for key,j in submission['jobs'].items():
        if key=='collector':continue
        c=read(verify(j['config']));registered[c['original']['checkpoint']['sha256']]=(j,c)
    # Explicit cancelled endpoints were never observed; the new authority explicitly
    # includes valid FE_HISTORY W20 CP. Preserve cancelled IDs and use fresh identities.
    accounting=read('/mnt/raid5/janghj/ODE-edit/local/completed-table-flucon-cap3-20261010/server1/accounting.json')
    states={s.split('|')[0]:s.split('|')[2] for s in accounting['rows'].splitlines() if s}
    refs=load_assets(e.REFERENCE);configs=[];eligible=[];seen=set()
    candidates=[('gptj','61927',e.BASE/'memit-fe-history-three-model-2k/preparation-r1/configs/gptj.json'),
                ('llama3','61928',e.BASE/'memit-fe-history-three-model-2k/preparation-r1/configs/llama3.json'),
                ('qwen25','62061',Path('/mnt/raid5/janghj/ODE-edit/local/qwen-baseline-mask-cold-rerun-20261010/server1/preparation-r1/configs/qwen25.json'))]
    # Bind Qwen from its actual submission rather than assume its config filename.
    qs=read('/mnt/raid5/janghj/ODE-edit/local/qwen-baseline-mask-cold-rerun-20261010/server1/registration-r1/submission.json')
    candidates[-1]=('qwen25','62061',Path(qs['jobs']['qwen25']['config']['path']))
    for model,jid,cpth in candidates:
        c=read(cpth);folder=Path(c['output']);done=read(folder/'COMPLETE.json');ident=done['identity']
        assert done['requests']==2000 and done['native_apply_calls']==done['history_appends_per_layer']==20
        assert c['model']==model and c['method']=='MEMIT_FE_HISTORY' and c['generation_schedule']=='DEFERRED_CHECKPOINT_EVALUATION'
        assert ident['config_sha256']==c['config_sha256'] and states[jid]=='COMPLETED'
        records=read(verify(c['stream_member']));records=records['records'] if isinstance(records,dict) else records
        assert len(records)==2000 and digest(records)==c['stream_sha256']
        previous=None
        for b in range(1,21):
            cm=read(folder/'commits'/f'batch-{b:02d}.json');edit=cm['cursor']['edit']
            assert cm['identity']==ident and cm['batch']==cm['cursor']['completed_batch']==b
            assert edit['requests']==100 and edit['history_appends_per_layer']==1
            assert edit['request_sha256']==digest(registry.requests(records[(b-1)*100:b*100],'MEMIT_FE_HISTORY',model))
            if previous is not None:assert edit['before']==previous
            previous=edit['after']
        pointer=read(folder/'checkpoint/latest.json');assert pointer==cm['checkpoint'] and pointer['batch']==20 and pointer['final_W20']
        assert pointer['identity_sha256']==digest(ident) and done['final_cursor']==cm['cursor']
        assert Path(pointer['file']).name==pointer['file']
        cp=member(folder/'checkpoint'/pointer['file']);assert cp['sha256']==pointer['sha256'] and cp['sha256'] not in seen;seen.add(cp['sha256'])
        raw=read(verify(cm['cursor']['factual']));assert len(raw['cases'])==raw['summary']['requests']==2000
        prior=registered.get(cp['sha256']);prior_id=None
        if prior:
            j,prev=prior;prior_id=j['job_id']
            assert states[prior_id].startswith('CANCELLED') and not (Path(prev['output'])/'COMPLETE.json').exists()
            assert not (Path(prev['output'])/'generation').exists(),'PARTIAL_GENERATION_NEEDS_EXACT_REUSE_REVIEW'
            dedup.append(j['config'])
        a=read(verify(c['assets']))
        for m in a['model']['members']:asset_member(m['path'],allow_symlink=True,expected_sha=m['sha256'],expected_bytes=m['bytes'])
        key=model+'-memit_fe_history-cap3'
        orig=dict(job_id=jid,method='MEMIT_FE_HISTORY',identity=ident,checkpoint=cp,pointer=member(folder/'checkpoint/latest.json'),
            terminal=member(folder/'COMPLETE.json'),config=member(cpth),hparams=c['hparams'],expected_history=True,
            ordered_case_ids_sha256=digest([r['case_id'] for r in records]))
        new=dict(instruction=e.INSTRUCTION,registration_authority=CAP3_AUTHORITY,key=key,model=model,original=orig,
            assets=c['assets'],asset_schema='history',stream=c['stream_member'],reference=member(e.REFERENCE),reference_identity=refs.sha,
            evaluator=member(Path(e.__file__).parents[2]/'evaluation/generation/native_observer.py'),runtime=runtime(),
            output=str(output/'runs'/key),tracking_env_file=c['tracking_env_file'])
        new['config_sha256']=digest(new);e.tracking_values(new,'a'*40)
        path=output/'configs'/f'{key}.json';write_new(path,new);configs.append(member(path))
        eligible.append(dict(model=model,original_job=jid,checkpoint=cp,prior_cancelled_eval=prior_id,commits=20,requests=2000,
                             latest_explicit_authority=CAP3_AUTHORITY))
        print(model,'W20_CP_AND_20_COMMITS_VERIFIED',flush=True)
    value=dict(instruction=e.INSTRUCTION,registration_authority=CAP3_AUTHORITY,configs=configs,reference=member(e.REFERENCE),
               eligible_checkpoints=eligible,dedup_receipts=dedup,dedup_verified=True)
    write_new(output/'preparation.json',value)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();prepare(a.output)
