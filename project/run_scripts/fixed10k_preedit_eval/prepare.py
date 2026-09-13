"""Bind exact SH4 evaluation assets to server2, without loading a model."""
import argparse, hashlib, json, os, subprocess, tarfile
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def save(path,value):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False)

def main(root,repo):
    original=root/'support-sh4-v1/execution.lock.json'
    assert sha(original)=='f6d1d40170fb0df7bea9dd7101282ce046dc8f4dccac3fd49a43a220f69c227a'
    assert sha(root/'support-sh4-v1/local-source.tar')=='582a378906fbcc3304de92011d67d54f28840d76cfb59ee1f6fa00166bda0560'
    old=json.loads(original.read_text());transfer=json.loads((root/'support-sh4-v1/preedit-transfer-receipt.json').read_text())
    assert transfer['preedit_job']=='42656' and transfer['preedit_started'] is False and transfer['preedit_scientific_denominator']==0
    assert 'JobState=CANCELLED' in transfer['preedit_terminal'] and 'AllocTRES=(null)' in transfer['preedit_terminal']
    snapshot=Path('/mnt/raid5/janghj/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots')/old['revision']
    mappings={old['source_root']:root/'helper-source',old['dependencies']:root/'deps-transformers-4.44.2',old['snapshot']:snapshot}
    members=[]
    for x in old['members']:
        for source,dest in mappings.items():
            if x['path'].startswith(source+'/'):
                p=dest/x['path'][len(source)+1:]
                assert p.is_file() and p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],str(p)
                members.append(dict(x,path=str(p),source_path=x['path']))
                break
    assert len(members)==1757
    with tarfile.open(root/'support-sh4-v1/local-source.tar') as archive:
        for item in archive.getmembers():
            assert item.isfile() and '..' not in Path(item.name).parts and not item.name.startswith('/')
            p=root/'native-source'/item.name
            expected=hashlib.sha256(archive.extractfile(item).read()).hexdigest()
            assert not p.is_symlink() and sha(p)==expected
            members.append(dict(path=str(p),bytes=p.stat().st_size,sha256=expected,source_archive_member=item.name))
    assert sha(root/'native-source/native/preedit.py')=='d913c5409a9d5faa8b8e2b0ffa3c4bf4a0152cd3a8e1e5bb090e0cf10c40006e'
    local_files=sorted(Path(__file__).parent.glob('*'))+[repo/'scripts/fixed_counterfact.py']
    for p in local_files:
        if p.is_file():members.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    fixed=dict(old['fixed_dataset_binding']);dataset=Path('/mnt/raid5/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1')
    fixed.update(root=str(dataset),loader=str(repo/'scripts/fixed_counterfact.py'),members=[dict(x,path=str(dataset/Path(x['path']).name)) for x in fixed['members']])
    for x in fixed['members']:assert sha(x['path'])==x['sha256']
    head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    tree=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD^{tree}'],text=True).strip()
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
    lock=dict(instruction_id='ODEEDIT-S06-FIXED10K-PREEDIT-EVALUATION-SH2-V1',evaluation_type='PRE_EDIT_W0_FULL10000',
              source_head=head,source_tree=tree,source_root=str(repo),snapshot=str(snapshot),revision=old['revision'],
              dependencies=str(root/'deps-transformers-4.44.2'),sample=str(dataset/'source-sample.lock.json'),
              sample_root=old['sample_root'],seed=old['seed'],fixed_dataset_binding=fixed,members=members,
              resource=dict(cap=2,gpus=1,cpus=8,memory_mib=60416,wall_hours=48),scientific_promotion=False,
              sh4_parent_lock_sha256=sha(original),sh4_helper_head=old['source_head'],sh4_helper_tree=old['source_tree'],
              sh4_transfer_receipt_sha256=sha(root/'support-sh4-v1/preedit-transfer-receipt.json'),
              sh4_archive_sha256=old['local_source_sha256'],excluded_unused_members='BLUE/native writer/P/stats/hparams not loaded or copied; evaluation dependency closure only',
              native_evaluator_unchanged=True,microbatch=16,parts=100,requests_per_part=100,
              full_nonselected_weight_byte_hash='NOT_CLAIMED; all parameter pointer/version guard and selected5 byte hashes',
              output=str(root/'output'))
    source_archive=root/'server2-source.tar'
    with source_archive.open('xb') as f:
        with tarfile.open(fileobj=f,mode='w') as t:
            for p in local_files:
                if p.is_file():t.add(p,arcname=str(p.relative_to(repo)),recursive=False)
    lock['server2_archive_sha256']=sha(source_archive)
    save(root/'execution.lock.json',lock)
    print(json.dumps(dict(status='SOURCE_INPUT_ASSET_GATE_PASS',head=head,tree=tree,members=len(members),lock_sha256=sha(root/'execution.lock.json'),duplicate_gate='SH4_CANCELLED_BEFORE_EXECUTION')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);a=p.parse_args();main(a.root,a.repo)
