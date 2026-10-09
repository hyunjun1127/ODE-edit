"""Bounded six completed Llama W20 checkpoint inventory; never load tensors."""
import argparse
import json
from pathlib import Path
from official.experiments.prepare import digest,write_new,file_sha
from .common import read,member,verify
from .assets import member as asset_member

INSTRUCTION='USER-GH-ZSRE-SAVED-WEIGHTS-2K-REEVAL-20261009-R1'
TASK='zsre-2k-reeval-20261009'
LOCAL=Path('/mnt/raid5/janghj/ODE-edit/local/official-baselines/server1')/TASK
JOBS={'61716','61717','61718','61719','61720','61721'}

def inventory(output):
    root=Path(__file__).resolve().parents[3]
    source=root/'audits/servers/server1/official-baselines-20261008/results-review-20261009/results.json'
    prior=read(source);rows=[];seen=set();base=None;stream=None
    candidates=[r for r in prior['rows'] if r['dataset']=='zsre' and r['complete']]
    assert {r['job_id'] for r in candidates}==JOBS
    for old in candidates:
        row={k:old[k] for k in ('method','model','dataset','job_id','source','config','identity','ordered_case_ids_sha256')}
        try:
            cfg=read(verify(old['config']));terminal=read(verify(old['terminal']))
            assert cfg['dataset']=='zsre' and cfg['model']=='llama3' and cfg['method']==old['method']
            assert terminal['requests']==2000 and terminal['native_apply_calls']==20 and terminal['checkpoint_W20_preserved']
            assert terminal['identity']==old['identity']
            pointer_path=verify(old['checkpoint_metadata']);pointer=read(pointer_path)
            assert pointer['batch']==20 and pointer['final_W20'] and pointer['identity_sha256']==digest(old['identity'])
            assert Path(pointer['file']).name==pointer['file']
            payload=member(pointer_path.parent/pointer['file']);assert payload['sha256']==pointer['sha256']
            stat=Path(payload['path']).stat();replica=(stat.st_dev,stat.st_ino,payload['sha256'])
            assert replica not in seen,'DUPLICATE_REPLICA';seen.add(replica)
            if base is None:
                assets=read(verify(cfg['assets_member']));base=dict(manifest=cfg['assets_member'],model=assets['model'])
                for m in assets['model']['weights']:
                    asset_member(m['path'],expected_sha=m['sha256'],expected_bytes=m['bytes'],allow_symlink=True)
                for m in assets['model']['tokenizer_files'].values():
                    asset_member(m['path'],expected_sha=m['sha256'],expected_bytes=m['bytes'],allow_symlink=True)
                bundle=read(verify(cfg['stream_bundle_member']));stream=bundle['datasets']['zsre']['stream']
                records=read(verify(stream));assert len(records)==2000
                assert digest([r['case_id'] for r in records])==old['ordered_case_ids_sha256']
            else:assert cfg['assets_member']==base['manifest']
            row.update(status='FINAL_W20_CHECKPOINT_FULL_SHA_VERIFIED',checkpoint=payload,pointer=member(pointer_path),
                       terminal=old['terminal'],hparams=cfg['hparams'],stream=stream,
                       expected_history=old['method'] in ('ALPHAEDIT','ALPHAEDIT_BLUE','SPHERE'),
                       base_revision=base['model']['identity']['revision'])
        except Exception as error:row.update(status='MISSING_OR_INCOMPLETE',error=str(error),error_type=type(error).__name__)
        rows.append(row);print(row['method'],row['status'],flush=True)
    value=dict(instruction=INSTRUCTION,server='server1',model='llama3',rows=rows,base=base,
               approved_baseline_candidates=6,approved_ours_completed=0,replica_duplicates=0,
               evaluator_status='SOURCE_INPUT_PENDING',actual_restore='NOT_RUN',actual_GPU='NOT_RUN',new_jobs=[],
               CP_transfer_delete=False,raw_modified=False,source_inventory_reducer_sha256=file_sha(__file__))
    write_new(output,value);return value

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();inventory(a.output)
