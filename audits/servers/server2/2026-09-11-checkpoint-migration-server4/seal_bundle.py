"""검증된 bundle만 no-replace 원자 rename하고 삭제 조건부 수신확인서를 만든다."""
import argparse
import ctypes
import json
import os
from pathlib import Path
from verify_initial import CONTROL,ARCHIVE,save,sha

def rename_noreplace(source,target):
    libc=ctypes.CDLL(None,use_errno=True)
    fn=libc.renameat2;fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];fn.restype=ctypes.c_int
    if fn(-100,os.fsencode(source),-100,os.fsencode(target),1)!=0:
        err=ctypes.get_errno();raise OSError(err,os.strerror(err),str(target))

def main(name,ready):
    assert ready.is_file() and ready.resolve().is_relative_to(CONTROL/'incoming')
    # The caller binds this exact source transfer completion receipt after SH4 READY, never a partial payload.
    records=json.loads((CONTROL/(name+'-full-rehash.json')).read_text())
    refs=json.loads((CONTROL/(name+'-reference-closure.json')).read_text())
    assert records['status']=='ALL_BYTES_VERIFIED_AWAITING_REFERENCE_CLOSURE'
    assert refs['status']=='REFERENCE_CLOSURE_PASS' and not refs['missing']
    stage=ARCHIVE/(name+'.partial');final=ARCHIVE/name
    for item in refs['verified']:
        v=item['destination'];s=Path(v['path']).stat()
        assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(v['dev'],v['inode'],v['bytes'],v['mtime_ns']),('REFERENCE_CHANGED',v['path'])
    for m in records['members']+records['companions']:
        v=m['staging_verification'];p=Path(v['path']);s=p.stat()
        assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(v['dev'],v['inode'],v['bytes'],v['mtime_ns'])
        assert not p.is_symlink() and s.st_nlink==1 and s.st_uid==os.getuid()
        fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
        try:os.fsync(fd)
        finally:os.close(fd)
    for directory in sorted([stage]+[p for p in stage.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True):
        fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:os.fsync(fd)
        finally:os.close(fd)
    assert not final.exists()
    rename_noreplace(stage,final)
    fd=os.open(ARCHIVE,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    catalog=dict(bundle=name,retention='KEEP_UNTIL_SEPARATE_USER_AUTHORIZED_RETIREMENT',retained_root=str(final),
         checkpoint_members=records['members'],copy_only_companions=records['companions'],shared_references=refs,
         source_transfer_ready=dict(path=str(ready),sha256=sha(ready)),
         full_rehash_receipt=dict(path=str(CONTROL/(name+'-full-rehash.json')),sha256=sha(CONTROL/(name+'-full-rehash.json'))),
         gpu_continuation_replay=0,full_model_checkpoint=False,
         restoration='Exact pretrained W0 + recorded selected keys overwrite + saved method state/context. Historical BLUE1k RNG not saved; no bitwise continuation claim.')
    save(CONTROL/(name+'-retention-catalog.json'),catalog)
    receipt=dict(status='VERIFIED_DESTINATION',instruction_id='ODEEDIT-S06-SERVER4-CHECKPOINT-MIGRATION-SERVER2-V1',
      bundle=name,checkpoint_count=records['checkpoint_count'],checkpoint_bytes=records['checkpoint_bytes'],
      companion_count=records['companion_count'],companion_bytes=records['companion_bytes'],retained_root=str(final),
      source_manifest_sha256=records['source_manifest_sha256'],
      destination_manifest=str(CONTROL/(name+'-retention-catalog.json')),
      destination_manifest_sha256=sha(CONTROL/(name+'-retention-catalog.json')),
      atomic_noreplace_seal=True,payload_full_sha256=True,source_transfer_ready=dict(path=str(ready),sha256=sha(ready)),
      source_deletion_owner='SH4',source_delete_condition='Only mapped exact checkpoint files after CURRENT source identity/no-consumer recheck; companions/shared assets/directories never delete.',
      raw_source_evaluation_log_mutation=0,rsync_writer_actions=0,gpu_model_evaluator_slurm_actions=0,
      scientific_outcome_changed=False)
    save(CONTROL/(name+'-VERIFIED_DESTINATION.json'),receipt)
    print('VERIFIED_DESTINATION',name,sha(CONTROL/(name+'-VERIFIED_DESTINATION.json')))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('bundle',choices=['new177-v1','jvp1k-v1']);ap.add_argument('--ready-receipt',type=Path,required=True);a=ap.parse_args();main(a.bundle,a.ready_receipt)
