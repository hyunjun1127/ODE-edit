"""Compact current-status record; preserves the original storage-block package."""
import datetime
from pathlib import Path
from .common import ROOT, INSTRUCTION, read, save, sha


def main():
    repo=Path(__file__).resolve().parents[3]
    attempt=ROOT/'attempt-v1'
    out=repo/'audits/servers/server4/native-delayed-write-e3-20260924-v1/resumption-r1'
    records={}
    for name in ('submission.json','admission.json','held-inspection.json'):
        value=read(attempt/name)
        # Held inspection is a compact Slurm command/config receipt, not science raw.
        records[name]=save(out/name,value)
    direct=read(ROOT/'receipts/gh-submission-direct.json')
    assert direct['status']=='DELIVERED_COMPLETED'
    records['gh-direct.json']=save(out/'gh-direct.json',direct)
    gates={}
    for stage in ('G00','G10'):
        p=attempt/'output'/stage/'gate-result.json';g=read(p)
        assert g['status']=='PASS' and g['binding']['instruction_id']==INSTRUCTION
        records[stage]=save(out/f'{stage}.json',g);gates[stage]='PASS'
    save(out/'resumption.json',dict(instruction_id=INSTRUCTION,
        user_recall='저장공간 확보했으니 task 이어서 진행하고 GH에게 최종보고까지 완료해',
        recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        state='RUNNING_E1_G20',jobs=read(attempt/'submission.json')['jobs'],
        prior_state='BLOCKED_STORAGE_NOT_SUBMITTED',prior_receipt_preserved=True,
        no_files_deleted=True,no_new_checkpoint=True,source_unchanged=True,
        execution_source='3ebe0b07078940c2d46f9ea2226ccc20c0446162',
        lock_sha256=sha(attempt/'execution.lock.json'),actual_gates=gates,
        allocated_not_utilization=True,monitoring_until_authorized_terminal=True,
        records={k:dict(path=str(Path(v['path']).relative_to(repo)),sha256=v['sha256'],bytes=v['bytes']) for k,v in records.items()}))


if __name__=='__main__':main()
