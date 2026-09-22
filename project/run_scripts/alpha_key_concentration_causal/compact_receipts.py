"""Copy only compact preparation receipts into the authorized Git audit scope."""
import argparse
import json
from pathlib import Path
from .common import save,file_sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--repo',required=True);p.add_argument('--cpu',required=True)
    a=p.parse_args();root=Path(a.root);repo=Path(a.repo)
    audit=repo/'audits/servers/server4/alpha-key-causal-20260923-r1/preparation-r1'
    for src,name in [(Path(a.cpu),'cpu-tests.json'),(root/'receipts/preflight-r1/resource-plan.json','resource-plan.json')]:
        save(audit/name,json.loads(src.read_text()))
    original=json.loads((root/'receipts/preflight-r1/checkpoint-content/checkpoint-content.json').read_text())
    save(audit/'checkpoint-content-summary.json',dict(status=original['status'],checkpoints=original['checkpoints'],seconds=original['wall_seconds'],
        full_receipt_path=str(root/'receipts/preflight-r1/checkpoint-content/checkpoint-content.json'),
        full_receipt_sha256=file_sha(root/'receipts/preflight-r1/checkpoint-content/checkpoint-content.json'),
        new_GPU_calls=0,rows=[dict(batch=r['batch'],file_bytes=r['file_bytes'],file_sha256=r['file_sha256'],
            content_sha256=r['metadata_sha256'],state=r['original_commit_endpoint'],verification=r['tensor_content_verification']) for r in original['rows']]))
    transfers=[]
    for name in ('design-transfer-r1.json','checkpoint-transfer-r1.json'):
        path=root/'receipts'/name;value=json.loads(path.read_text())
        transfers.append(dict(receipt=str(path),sha256=file_sha(path),status=value['status'],members=len(value['members']),
            bytes=sum(x['bytes'] for x in value['members']),source_keep=True))
    save(audit/'transfer-summary.json',transfers)
    approval=json.loads((repo/'transfers/approvals/2026-09-23-alpha-key-causal-sh4-inputs.json').read_text())
    files=[]
    for item in approval['members']:
        path=root/'inputs/design'/item['destination_relative_path']
        assert file_sha(path)==item['sha256']
        files.append(dict(relative_path=item['destination_relative_path'],bytes=item['bytes'],sha256=item['sha256']))
    save(audit/'full-read-binding.json',dict(status='FULL_READ',input_members=files,
        authority=['4da5514323ad58fce709e4f8fe2c8238b15a8b1e','33e7bb7f7d428f34c475f2ebe8e5e7beb4539f7a'],
        actual_model_execution='NOT_RUN',PROTOCOL_sha256=file_sha(repo/'PROTOCOL.md'),
        primary_design_contract_cells_and_proposal='FULL_READ',panels='FULL_PARSE_IDENTITY_INTEGRITY_CHECKED',
        frozen_native_source='FULL_READ',GH_preparation_scripts='READ_ONLY_NOT_EXECUTED'))

if __name__=='__main__':main()
