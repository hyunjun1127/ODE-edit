"""SH2 전용 imports boundary 안에서만 검산/atomic seal. 모델 로딩 없음."""
import argparse,hashlib,json,os
from pathlib import Path
BASE=Path('/mnt/raid5/janghj/ODE-edit/local/blue-checkpoint-downstream/20260909-v1/imports')
ROOT=BASE/'initial6-v1'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def safe(p):
    assert p.is_relative_to(BASE) and p.resolve()==p
    assert p.stat().st_uid==os.getuid()
def save(p,obj):
    with p.open('x') as f:json.dump(obj,f,sort_keys=True,indent=2)
    p.chmod(0o600)
def verifyfile(p,n,h):
    safe(p);assert p.is_file() and not p.is_symlink()
    assert p.stat().st_size==n and sha(p)==h,str(p)
def main(mode,expected):
    safe(BASE)
    if mode=='init':
        v=os.statvfs(BASE);assert v.f_bavail*v.f_frsize>100*(1<<30) and v.f_favail>1000
        ROOT.mkdir(mode=0o700,exist_ok=False);(ROOT/'source.partial').mkdir(mode=0o700);(ROOT/'payload.partial').mkdir(mode=0o700)
        receipt=dict(status='CREATE_ONCE_LANDING',free_bytes=v.f_bavail*v.f_frsize,free_inodes=v.f_favail,writer='SH4',original_overwrite=False)
        save(ROOT/'landing.json',receipt);print(json.dumps(receipt));return
    safe(ROOT)
    if mode=='source':
        partial=ROOT/'source.partial';seal=partial/'source-seal.json'
        assert sha(seal)==expected
        data=json.loads(seal.read_text())
        for m in data['members']:verifyfile(partial/m['relative'],m['bytes'],m['sha256'])
        target=ROOT/'source';assert not target.exists();partial.rename(target)
        receipt=dict(status='SOURCE_READY',files=len(data['members'])+1,source_seal_sha256=expected,manifest_sha256=sha(target/'checkpoint-manifest.json'))
        save(ROOT/'source-ready.json',receipt);print(json.dumps(receipt));return
    assert mode=='payload'
    manifest=ROOT/'source/checkpoint-manifest.json';assert sha(manifest)==expected
    data=json.loads(manifest.read_text());partial=ROOT/'payload.partial'
    for c in data['checkpoints']:
        m=c['file'];rel=Path(m['destination_relative']);assert rel.parts[0]=='payload'
        verifyfile(partial/Path(*rel.parts[1:]),m['bytes'],m['sha256'])
        print('SHA_OK',c['arm'],c['batch'],flush=True)
    target=ROOT/'payload';assert not target.exists();partial.rename(target)
    receipt=dict(status='READY',checkpoints=len(data['checkpoints']),bytes=data['total_checkpoint_bytes'],manifest_sha256=expected,source_seal_sha256=sha(ROOT/'source/source-seal.json'),verification='remote full file size/SHA; source CPU tensor/schema hashes in manifest; SH2 independent loader audit separate',GPU=0,model_load=0)
    save(ROOT/'transfer-ready.json',receipt);print(json.dumps(receipt),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode');p.add_argument('--expected',default='');a=p.parse_args();main(a.mode,a.expected)
