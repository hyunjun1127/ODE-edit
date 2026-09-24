"""Sole receiver, exact approved package and safe 29-file extraction."""
import datetime
import os
from pathlib import PurePosixPath
import shutil
import subprocess
import tarfile
from .common import *

def main():
    approval=REPO/'transfers/approvals/2026-09-24-historical-update-timeaxis-sh4.json';a=read(approval)
    assert a['instruction_id']==INSTRUCTION and a['receiver_session']==SESSION
    assert subprocess.check_output(['hostname'],text=True).strip()=='server4' and os.environ['CODEX_THREAD_ID']==SESSION
    peer=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',a['design_source_alias'],'hostname; id -un'],text=True).strip().splitlines()
    assert peer==['devbox','janghj'],peer
    assert shutil.disk_usage(ROOT).free>64*2**20
    received=[]
    for r in a['files']:
        p=Path(r['destination']);assert p.is_relative_to(ROOT/'inputs/design-package');p.parent.mkdir(parents=True,exist_ok=True)
        status='REUSED_VERIFIED' if p.exists() else 'RECEIVED'
        if not p.exists():
            tmp=p.with_name(p.name+'.partial');assert not tmp.exists()
            with tmp.open('xb') as out:
                subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',a['design_source_alias'],'cat -- '+r['path']],stdout=out,check=True)
                out.flush();os.fsync(out.fileno())
            assert tmp.stat().st_size==r['bytes'] and sha(tmp)==r['sha256'],('TRANSFER_MISMATCH',r['path'])
            os.link(tmp,p);tmp.unlink()
        assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'];received.append(dict(**record(p),status=status,source=r['path']))
    archive=Path(a['files'][0]['destination']);allow={r['relative_path']:r for r in a['archive_members']}
    members=[]
    prefix=PurePosixPath(read(ROOT/'inputs/design-package/dispatch-input-manifest.json')['plan_root']).name
    def member_key(m):
        name=PurePosixPath(m.name)
        assert not name.is_absolute() and '..' not in name.parts
        assert name.parts[0]==prefix and len(name.parts)>1
        return str(PurePosixPath(*name.parts[1:]))
    with tarfile.open(archive,'r:gz') as tf:
        ts=tf.getmembers();seen=set()
        for m in ts:
            name=PurePosixPath(m.name)
            assert not name.is_absolute() and '..' not in name.parts and m.isfile() and not m.issym() and not m.islnk(),('UNSAFE_TAR_MEMBER',m.name)
            key=member_key(m);assert key in allow and key not in seen;seen.add(key)
            assert m.size==allow[key]['bytes']
        assert seen==set(allow) and len(ts)==29
        for m in ts:
            key=member_key(m);p=DESIGN/key;p.parent.mkdir(parents=True,exist_ok=True)
            data=tf.extractfile(m).read();assert len(data)==allow[key]['bytes'] and hashlib.sha256(data).hexdigest()==allow[key]['sha256']
            if p.exists():assert p.read_bytes()==data
            else:
                with p.open('xb') as f:f.write(data)
            assert sha(p)==allow[key]['sha256'];members.append(dict(relative_path=key,**record(p)))
    # Existing E3 inputs are candidates only; T0 will freshly verify whole payload/tensors.
    prior=read('/data/janghj/ODE-edit/local/native-delayed-write-e3/20260924-v1/attempt-v1/execution.lock.json');cps=[]
    for cp in a['checkpoints']:
        m=prior['checkpoints'][cp['family']][str(cp['batch'])];p=Path(m['path'])
        assert m['sha256']==cp['sha256'] and m['bytes']==cp['bytes']
        cps.append(dict(family=cp['family'],batch=cp['batch'],path=str(p),expected_sha256=cp['sha256'],expected_bytes=cp['bytes'],
            availability='PRESENT_SIZE_MATCH' if p.exists() and p.stat().st_size==cp['bytes'] else 'MISSING_OR_SIZE_MISMATCH',prior_SHA_binding=True,new_T0_fullSHA='NOT_RUN'))
    result=dict(instruction_id=INSTRUCTION,status='RECEIVER_3_FILES_29_MEMBERS_VERIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        approval=record(approval),source_peer=peer,files=received,members=members,member_bytes=sum(r['bytes'] for r in members),
        source_keep=True,overwrite=False,checkpoint_candidates=cps,new_CP_transfer=0,implementation='IMPLEMENTING_NOT_SUBMITTED',job_ids=[],actual_model_validation='NOT_RUN')
    save(ROOT/'receipts/design-received.json',result)
    save(REPO/'transfers/verifications/2026-09-24-historical-update-timeaxis-sh4/design-received.json',result)
    print(json.dumps(dict(receipt=record(ROOT/'receipts/design-received.json'),members=len(members),member_bytes=result['member_bytes'],
        checkpoints_present=sum(r['availability']=='PRESENT_SIZE_MATCH' for r in cps),status=result['implementation'],job_ids=[])))

if __name__=='__main__':main()
