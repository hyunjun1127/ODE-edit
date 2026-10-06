"""CPU-only compact publication checks, without scheduler/model/raw result reads."""
import argparse
import hashlib
import json
import re
from pathlib import Path

def check(root, task):
    audit = root / 'audits/servers/server1' / task
    receipt = json.loads((audit/'submission-receipt.json').read_text())
    manifest = json.loads((audit/'artifact-manifest.json').read_text())
    status = json.loads((root/'tasks/status'/task/'server1.json').read_text())
    head = json.loads((root/'messages/server-heads/server1'/(task+'.json')).read_text())
    report = root/'experiment-reports/servers/server1'/task/'report-ko.md'
    for obj in (status,head):
        assert obj['jobs']==receipt['jobs']
        assert obj['execution_source']==receipt['source_commit']
        assert obj['lock_sha256']==receipt['lock']['sha256']
    assert status['scientific_completion']=='NOT_ESTABLISHED'
    assert manifest['raw_git'] is False and manifest['checkpoint_saved'] is False
    checked=[]
    for member in manifest['compact_receipt_members']+manifest['own_source_members']+[manifest['archive']]:
        path=Path(member['path'])
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size==member['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==member['sha256']
        checked.append(str(path))
    local=json.loads((Path(manifest['local_attempt'])/'submission.json').read_text())
    def omit_stat(obj):
        if isinstance(obj,dict):
            return {key:omit_stat(value) for key,value in obj.items() if key not in ('inode','mtime_ns')}
        if isinstance(obj,list):return [omit_stat(value) for value in obj]
        return obj
    assert all(receipt[key]==omit_stat(value) for key,value in local.items())
    text=report.read_text()
    assert any('\uac00'<=char<='\ud7a3' for char in text)
    links=[]
    for target in re.findall(r'\]\(([^)]+)\)',text):
        if not target.startswith(('https://','http://')):
            path=(report.parent/target.split('#')[0]).resolve()
            assert path.is_file(),target
            links.append(str(path))
    columns=None
    for line in text.splitlines()+['']:
        if line.startswith('|'):
            count=len(line.split('|'))-2
            if columns is None:columns=count
            assert count==columns,'GFM table column count'
        else:columns=None
    if task=='base-model-gpt2xl-w0':
        assert receipt['dependencies']['GPU'] is None
        assert status['USER_SCOPED_CAP_EXCEPTION']
    return dict(task=task,CPU_publication_check='PASS',metadata_hashes_checked=len(checked),
        relative_links_checked=len(links),GFM_table_columns='PASS',JSON='PASS',
        no_scheduler_or_science_poll=True,source_bytes_preserved=True,
        scientific_GPU_validation='NOT_OBSERVED',online_validation='NOT_OBSERVED',
        render='NOT_RUN; markdown_it/mistune/markdown unavailable in pinned Python',
        report=dict(path=str(report),bytes=report.stat().st_size,
            sha256=hashlib.sha256(report.read_bytes()).hexdigest()))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('task')
    args=parser.parse_args();print(json.dumps(check(args.root.resolve(),args.task),ensure_ascii=False))
