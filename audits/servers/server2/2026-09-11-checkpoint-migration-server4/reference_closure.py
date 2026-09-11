"""전송된 lock의 shared 모델/P/stats 참조를 로컬 정확한 bytes에 결속한다."""
import argparse
import json
from pathlib import Path
from verify_initial import CONTROL,ARCHIVE,save,sha,stable_verify

def candidates(value):
    if isinstance(value,dict):
        if all(k in value for k in ['absolute_path','bytes']):
            digest=value.get('sha256',value.get('expected_sha256'))
            if digest:
                if 'observed_sha256' in value:assert digest==value['observed_sha256']
                yield from candidates(dict(path=value['absolute_path'],bytes=value['bytes'],sha256=digest))
        if all(k in value for k in ['snapshot','snapshot_config_sha256','snapshot_tokenizer_sha256']):
            for name,key in [('config.json','snapshot_config_sha256'),('tokenizer.json','snapshot_tokenizer_sha256')]:
                p=str(Path(value['snapshot'])/name)
                local=Path(p.replace('/data/janghj/','/mnt/raid5/janghj/',1))
                yield dict(path=p,bytes=local.stat().st_size if local.is_file() else -1,sha256=value[key])
        if all(k in value for k in ['path','bytes','sha256']):
            p=value['path']
            if isinstance(p,str) and ('/.cache/huggingface/hub/' in p or '/examples/null_space_project_' in p or '/examples/data/stats/' in p):
                yield value
        for v in value.values():yield from candidates(v)
    elif isinstance(value,list):
        for v in value:yield from candidates(v)

def main(name):
    stage=ARCHIVE/(name+'.partial')
    # Only exact manifest-listed closure paths, not arbitrary scientific outputs.
    r=json.loads((CONTROL/(name+'-full-rehash.json')).read_text())
    assets={};locks=[]
    for m in r['companions']:
        p=Path(m['staging_verification']['path'])
        if p.name.endswith('.lock.json'):
            assert sha(p)==m['staging_verification']['sha256']
            d=json.loads(p.read_text());locks.append(dict(path=str(p),sha256=sha(p)))
            for a in candidates(d):
                key=a['path']
                if key in assets:assert (assets[key]['bytes'],assets[key]['sha256'])==(a['bytes'],a['sha256']),('CONFLICTING_ASSET_LOCK',key)
                assets[key]=a
    assert assets, 'NO_EXACT_SHARED_ASSET_REFERENCES_IN_LOCKS'
    # Snapshot shard byte identities are derived only from real content-addressed HF blob links;
    # the source lock pins revision/config/tokenizer, not a source-side full shard manifest.
    for a in list(assets.values()):
        if '/snapshots/' in a['path']:
            snapshot=Path(a['path']).parent
            local=Path(str(snapshot).replace('/data/janghj/','/mnt/raid5/janghj/',1))
            index=local/'model.safetensors.index.json'
            if index.is_file():
                names=set(json.loads(index.read_text())['weight_map'].values())|{'model.safetensors.index.json','tokenizer_config.json','generation_config.json','special_tokens_map.json'}
                for n in names:
                    p=local/n;source=str(snapshot/n)
                    if source in assets or not p.exists():continue
                    blob=p.resolve().name
                    if p.is_symlink() and len(blob)==64 and all(c in '0123456789abcdef' for c in blob):
                        assets[source]=dict(path=source,bytes=p.stat().st_size,sha256=blob,identity_basis='LOCAL_HF_CONTENT_ADDRESS_UNDER_SOURCE_PINNED_REVISION')
    verified=[];missing=[]
    cached=json.loads((CONTROL/'initial72-base-reference-rehash.json').read_text())['members']
    prior_jvp=CONTROL/'jvp1k-v1-reference-closure.json'
    if name!='jvp1k-v1' and prior_jvp.is_file():
        previous=json.loads(prior_jvp.read_text())
        assert previous['status']=='REFERENCE_CLOSURE_PASS'
        cached += [x['destination'] for x in previous['verified']]
    cached_by={(x['path'],x['sha256']):x for x in cached}
    for a in assets.values():
        source=a['path'];p=Path(source.replace('/data/janghj/','/mnt/raid5/janghj/',1))
        if not p.is_file():missing.append(dict(source_path=source,expected_sha256=a['sha256'],expected_bytes=a['bytes'],local_path=str(p),status='MISSING_REFERENCE'));continue
        old=cached_by.get((str(p),a['sha256']))
        if old:
            s=p.stat()
            if (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(old['dev'],old['inode'],old['bytes'],old['mtime_ns']):
                verified.append(dict(source_path=source,destination=old,verification='THIS_TASK_PRIOR_FULL_SHA_STABLE_STAT_REUSE'));continue
        try:v=stable_verify(p,a,True)
        except AssertionError as e:
            missing.append(dict(source_path=source,expected_sha256=a['sha256'],local_path=str(p),status='REFERENCE_MISMATCH',detail=str(e)));continue
        verified.append(dict(source_path=source,destination=v,verification='FULL_SHA256',identity_basis=a.get('identity_basis','SOURCE_LOCK_MEMBER')))
        print('SHARED_REFERENCE_VERIFIED',name,p.name,flush=True)
    save(CONTROL/(name+'-reference-closure.json'),dict(status='REFERENCE_CLOSURE_PASS' if not missing else 'HOLD_REFERENCE_CLOSURE',
        bundle=name,locks=locks,verified=verified,missing=missing,model_gpu_replay=0,
        limitations='W and saved M restoration supported; RNG absent in historical BLUE1k checkpoints; no promise of bitwise continuation.'))
    print('REFERENCE_RESULT',name,len(verified),len(missing),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('bundle',choices=['new177-v1','jvp1k-v1']);main(ap.parse_args().bundle)
