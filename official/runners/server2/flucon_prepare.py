"""Exact existing local W20 allowlist, CPU-only preparation; no checkpoint copy."""
import argparse
from pathlib import Path
from official.experiments.prepare import digest,write_new
from official.runners.server1.common import read,member,verify
from official.runners.server1.assets import member as asset_member
from official.runners.server1.flucon_eval import INSTRUCTION
from official.runners.server1.fe_history_prepare import runtime
from official.evaluation.generation.assets import load_assets
from .flucon_eval import AUTHORITY,LOCAL,tracking_values

BASE=Path('/mnt/raid5/janghj/ODE-edit/local')
GPTJ=BASE/'official-baselines-server2/20261008-r1'
QWEN=BASE/'qwen-baseline-mask-cold-rerun-20261010/registration-r1'
REFERENCE=BASE/'qwen-baselines-server2-20261009/preparation-r1/generation-local.json'
GJ_ROWS=[('FT','61650','registration-cf-checkpoint-r1','FT/chain'),
 ('MEMIT','61725','registration-no-gpu-qual-r1','CF_MEMIT'),
 ('ALPHAEDIT','61778','registration-cf-display-r1','CF_ALPHAEDIT'),
 ('ALPHAEDIT_BLUE','61779','registration-cf-display-r1','CF_ALPHAEDIT_BLUE'),
 ('MEMIT_FE','61780','registration-cf-display-r1','CF_MEMIT_FE'),
 ('SPHERE','61781','registration-cf-display-r1','CF_SPHERE')]

def original_read(m):
    current=member(m['path']);assert current['sha256']==m['sha256']
    if 'bytes' in m:assert current['bytes']==m['bytes']
    return read(m['path'])

def checked_pointer(folder,identity,last):
    pointer=read(folder/'latest.json')
    assert pointer['batch']==20 and pointer['final_W20']
    assert pointer['identity_sha256']==digest(identity) and pointer==last
    assert Path(pointer['file']).name==pointer['file']
    cp=member(folder/pointer['file'])
    assert cp['sha256']==pointer['sha256']
    return cp,member(folder/'latest.json')

def normalize_assets(output,tag,snapshot,members):
    # Full hash against original immutable members once per model; no transfer.
    unique={m.get('snapshot_path',m['path']):m for m in members}
    checked=[asset_member(path,allow_symlink=True,expected_sha=m['sha256'],expected_bytes=m['bytes'])
             for path,m in unique.items()]
    # Shared generic loader chooses safetensors using member suffix. Preserve the
    # verified snapshot filename as well as the physical full-stat identity.
    for item,path in zip(checked,unique):item['path']=path
    path=output/'assets'/f'{tag}.json'
    write_new(path,dict(model=dict(snapshot=str(snapshot),members=checked)))
    return member(path)

def candidates():
    for method,jid,registration,relative in GJ_ROWS:
        root=GPTJ/registration;folder=root/relative
        yield 'gptj',method,jid,root,folder
    for method,jid in [('MEMIT','62073'),('ALPHAEDIT','62075'),('MEMIT_FE','62077'),('SPHERE','62079')]:
        yield 'qwen25',method,jid,QWEN,QWEN/'runs'/('qwen25-cf-'+method.lower())

def prepare(output):
    output=Path(output).absolute();assert output.is_relative_to(LOCAL) and not output.exists()
    refs=load_assets(REFERENCE);assets_cache={};configs=[];inventory=[];seen=set()
    for tag,method,jid,root,folder in candidates():
        row=dict(model=tag,method=method,original_job_id=jid)
        try:
            if tag=='gptj':
                terminal=folder/'result.json';r=read(terminal)
                assert r['model']=='gptj' and r['dataset']=='cf' and r['method']==method and r['batches']==20
                assert r['generation_endpoint'] is None
                oldpath=root/'manifest.json';old=read(oldpath);identity=r['checkpoint_identity']
                stream=member(verify(old['streams']['cf']['member']))
                hp=read(root/'source/official/hparams'/method/'gptj.json')
                commits=r['commits'];assert len(commits)==20
                for n,m in enumerate(commits,1):
                    c=read(verify(m));assert c['batch']==n and c['requests']==100
                    assert c['checkpoint_identity']==identity and c['method']==method
                raw=read(verify(r['factual_endpoints']['W20']))
                assert raw['summary']['requests']==len(raw['cases'])==2000
                cp,pointer=checked_pointer(Path(r['checkpoint_folder']),identity,c['checkpoint'])
                if tag not in assets_cache:
                    assets_cache[tag]=normalize_assets(output,tag,old['model_snapshot'],old['model_assets'])
            else:
                terminal=folder/'terminal.json';r=read(terminal)
                assert r['actual_job_id']==jid and r['status']=='W20_COMPLETE' and r['completed_edits']==2000
                assert r['generation_schedule']=='DEFERRED_CHECKPOINT_EVALUATION'
                oldpath=root/'configs'/f'qwen25-cf-{method.lower()}.json';old=read(oldpath)
                identity=r['checkpoint_identity'];assert old['config_sha256']==identity['config_sha256']
                hp=old['hparams'];stream=member(root/'streams/cf-stream.json')
                assert len(r['commits'])==20
                for n,m in enumerate(r['commits'],1):
                    c=original_read(m);assert c['completed_batch']==n and c['checkpoint_identity']==identity
                raw=original_read(r['calculation_evidence']['factual']);assert len(raw)==2000
                assert c['factual']['summary']['requests']==2000
                cp,pointer=checked_pointer(folder/'checkpoint',identity,r['checkpoint'])
                assert c['checkpoint_sha256']==cp['sha256']
                if tag not in assets_cache:
                    a=read(root/'asset-preflight.json')['assets']['model_snapshot']
                    members=[]
                    for key in ('metadata_files','tokenizer_files','weight_shards'):
                        value=a[key];members.extend(value.values() if isinstance(value,dict) else value)
                    assets_cache[tag]=normalize_assets(output,tag,a['path'],members)
            assert stream['sha256']==identity['stream_sha256']
            records=read(verify(stream));assert len(records)==2000
            cases=raw['cases'] if isinstance(raw,dict) else raw
            assert [x['case_id'] for x in cases]==[x['case_id'] for x in records]
            assert cp['sha256'] not in seen;seen.add(cp['sha256'])
            original=dict(job_id=jid,method=method,identity=identity,checkpoint=cp,pointer=pointer,
                terminal=member(terminal),config=member(oldpath),hparams=hp,
                expected_history=method in ('ALPHAEDIT','ALPHAEDIT_BLUE','SPHERE'),
                ordered_case_ids_sha256=digest([r['case_id'] for r in records]))
            key=tag+'-'+method.lower()+'-'+jid
            config=dict(instruction=INSTRUCTION,registration_authority=AUTHORITY,key=key,model=tag,
                original=original,assets=assets_cache[tag],asset_schema='history',stream=stream,
                reference=member(REFERENCE),reference_identity=refs.sha,
                evaluator=member(Path(__file__).parents[2]/'evaluation/generation/native_observer.py'),
                runtime=runtime(),output=str(output/'runs'/key),
                tracking_env_file='/mnt/raid5/janghj/ODE-edit/servers/local/wandb.env')
            config['config_sha256']=digest(config);tracking_values(config,'a'*40)
            path=output/'configs'/f'{key}.json';write_new(path,config);configs.append(member(path))
            row.update(status='VALID_W20_2000_FULL_SHA',checkpoint=cp,identity=identity,config=member(path))
        except Exception as e:row.update(status='BLOCKED',error_type=type(e).__name__,reason=str(e))
        inventory.append(row);print(jid,row['status'],row.get('reason',''),flush=True)
    result=dict(instruction=INSTRUCTION,registration_authority=AUTHORITY,configs=configs,inventory=inventory,
        reference=member(REFERENCE),dedup_excluded=[dict(jobs=['61898','61962'],reason='EXISTING_QWEN_FT_BLUE_COMPLETE_GENERATION'),
          dict(jobs=['62581','62582','62583'],reason='SH1_EXISTING_FE_HISTORY_EVALUATIONS'),
          dict(jobs=['62531'],reason='AUTHOR_CF_W20_NOT_AVAILABLE_AT_INPUT_SNAPSHOT')],
        no_model_forward=True,no_checkpoint_copy=True,no_edit=True)
    write_new(output/'preparation.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();prepare(a.output)
