"""독립 재검산한 기존72의 장기보존 catalog와 수신자 확인서를 봉인한다."""
import json
import os
from pathlib import Path
from verify_initial import CONTROL,INITIAL,save,sha

def main():
    payload=json.loads((CONTROL/'initial72-payload-rehash.json').read_text())
    closure=json.loads((CONTROL/'initial72-closure-rehash.json').read_text())
    base=json.loads((CONTROL/'initial72-base-reference-rehash.json').read_text())
    assert payload['count']==72 and payload['bytes']==62011141768
    for row in payload['members']:
        m=row['destination'];s=Path(m['path']).stat()
        assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_uid,s.st_nlink)==(m['dev'],m['inode'],m['bytes'],m['mtime_ns'],os.getuid(),1)
    catalog=dict(bundle='existing-initial72',retention='KEEP_UNTIL_SEPARATE_USER_AUTHORIZED_RETIREMENT',
       checkpoint_root=str(INITIAL/'payload'),companion_root=str(INITIAL/'source'),
       members=payload['members'],source_closure=closure['members'],shared_asset_references=base['members'],
       restore=dict(base_revision='8afb486c1db24fe5011ec46dfbe5b5dccdb575c2',
           operation='Restore exact W0 then overwrite exactly listed selected weight keys; retain cache_c as method history, not inference weights.',
           schema='weights/cache_c/metadata; RNG/context inventory bound by original CPU audit and unchanged whole-file SHA',
           full_model_checkpoint=False,source_head='1075540b45c29269e690ac63aae44758d8d63174',
           loader=str(INITIAL/'source/closure/003-checkpoint_loader.py'),gpu_replay='NOT_PERFORMED',
           continued_training_numerical_parity='NOT_CLAIMED'))
    cp=CONTROL/'initial72-retention-catalog.json'
    if cp.exists():assert json.loads(cp.read_text())==catalog
    else:save(cp,catalog)
    receipt=dict(instruction_id='ODEEDIT-S06-SERVER4-CHECKPOINT-MIGRATION-SERVER2-V1',status='VERIFIED_DESTINATION',
       bundle='existing-initial72',source_manifest_sha256=payload['manifest_sha256'],
       source_manifest_path='NOT_RECORDED_IN_ORIGINAL_SOURCE_SEAL; SH4 must bind its current source manifest',
       destination_original_manifest=str(INITIAL/'source/checkpoint-manifest.json'),
       destination_manifest=str(CONTROL/'initial72-retention-catalog.json'),destination_manifest_sha256=sha(CONTROL/'initial72-retention-catalog.json'),
       count=72,checkpoint_bytes=62011141768,closure_members=100,closure_bytes=2293389,
       shared_assets=13,shared_asset_bytes=21824307420,full_sha256=True,in_place_reuse=True,rsync_writer_actions=0,
       mapping='destination_manifest.members[].source_path -> destination.path; source_sha256==destination.sha256',
       cpu_schema='Reused exact-hash-bound original weights_only selected-tensor/history/RNG/metadata audit; no model load.',
       gpu_replay=0,source_deletion_owner='SH4',
       source_delete_condition='Only exact mapped checkpoint files, after SH4 CURRENT source SHA/size/dev/inode/mtime/owner/single-link and no-active-consumer check. Any changed/unmapped/in-use source stays HOLD. No directories/companions/shared assets deletion.',
       destination_retention='No move/rename/overwrite; keep payload, source closure and exact shared base/P/stats references.',
       source_current_stat_and_consumer_checks='SH4_REQUIRED_NOT_PERFORMED_BY_SH2',scientific_outcome_change=False,
       supersedes='initial72-VERIFIED_DESTINATION.json (never sent; unverified original manifest path removed)')
    save(CONTROL/'initial72-VERIFIED_DESTINATION-v2.json',receipt)
    print('VERIFIED_DESTINATION',sha(CONTROL/'initial72-VERIFIED_DESTINATION-v2.json'))

if __name__=='__main__':main()
