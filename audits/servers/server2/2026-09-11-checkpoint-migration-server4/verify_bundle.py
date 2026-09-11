"""SH4 완료 통지 후 고정 allowlist payload/closure를 전수 검산. 원본·GPU 접근 없음."""
import argparse
import json
import os
from pathlib import Path
import time
from verify_initial import CONTROL,ARCHIVE,save,sha,stable_verify
from admit import EXPECTED

def load(name):
    p=CONTROL/'incoming'/name;assert sha(p)==EXPECTED[name]
    return json.loads(p.read_text())

def schema_cpu(path, arm):
    import torch
    import hashlib
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    cp=torch.load(path,map_location='cpu',weights_only=True)
    assert set(cp)=={'weights','cache_c','metadata'},list(cp)
    layers={'BLUE_1K':[4,8],'BLUE_L4_1K':[4],'BLUE_L8_1K':[8]}[arm]
    assert set(cp['weights'])=={f'model.layers.{l}.mlp.down_proj.weight' for l in layers}
    def tensor(t):
        assert t.dtype==torch.float32 and bool(torch.isfinite(t).all())
        x=t.detach().contiguous();h=hashlib.sha256(str((str(x.dtype),list(x.shape))).encode())
        raw=x.view(torch.uint8).numpy().reshape(-1)
        for i in range(0,raw.size,8<<20):h.update(memoryview(raw[i:i+(8<<20)]))
        return dict(dtype=str(t.dtype),shape=list(t.shape),sha256=h.hexdigest(),finite=True)
    weights={k:tensor(t) for k,t in cp['weights'].items()}
    for t in cp['weights'].values():assert tuple(t.shape)==(4096,14336)
    history=tensor(cp['cache_c']);assert list(cp['cache_c'].shape)==[len(layers),14336,14336]
    metadata=cp['metadata'];assert isinstance(metadata,dict)
    result=dict(status='CPU_WEIGHTS_ONLY_FULL_TENSOR_HASH_FINITE_PASS',weights=weights,history=history,
        metadata_keys=sorted(metadata),rng='NOT_SAVED; source contract seed only',gpu_replay=0,model_load=0)
    # Metadata values/prompts remain inside the unchanged local checkpoint, not in the public receipt.
    assert not torch.cuda.is_initialized()
    return result

def verify(name):
    source_name='source-manifest.json' if name=='new177-v1' else 'jvp-source-manifest.json'
    source=load(source_name);members=[m for m in source['members'] if not m['reuse']]
    final=ARCHIVE/name;stage=ARCHIVE/(name+'.partial')
    assert stage.is_dir() and not final.exists() and not stage.is_symlink()
    records=[];expected=set();start=time.monotonic()
    for i,m in enumerate(members):
        rel=Path(m['retained_path']).relative_to(final);assert '..' not in rel.parts
        p=stage/rel;expected.add(p)
        v=stable_verify(p,m)
        schema=m['schema'] or schema_cpu(p,m['arm'])
        records.append(dict(source_path=m['path'],destination_path=m['retained_path'],staging_verification=v,
          source_stat={k:m[k] for k in ['realpath','owner','uid','type','nlink','dev','inode','bytes','mtime_ns','sha256']},
          arm=m['arm'],batch=m['batch'],schema=schema,source_manifest_sha256=EXPECTED[source_name]))
        if (i+1)%12==0 or i+1==len(members):print('BUNDLE_PAYLOAD_FULL_SHA',name,i+1,len(members),flush=True)
    companions=[]
    sets=[('companion-manifest.json','closure'),('supplement-manifest.json','supplement')] if name=='new177-v1' else [('jvp-companion-manifest.json','closure')]
    for mf,sub in sets:
        for m in load(mf)['members']:
            rel=Path(m['relative']);assert not rel.is_absolute() and '..' not in rel.parts
            p=stage/sub/rel;expected.add(p);v=stable_verify(p,m)
            companions.append(dict(source_path=m['path'],destination_path=str(final/sub/rel),staging_verification=v,disposition='COPY_ONLY_SOURCE_PRESERVED'))
    observed={p for p in stage.rglob('*') if not p.is_dir()}
    assert observed==expected,dict(unexpected=[str(x) for x in observed-expected][:10],missing=[str(x) for x in expected-observed][:10])
    assert not any(p.is_symlink() for p in stage.rglob('*'))
    save(CONTROL/(name+'-full-rehash.json'),dict(status='ALL_BYTES_VERIFIED_AWAITING_REFERENCE_CLOSURE',
       bundle=name,source_manifest=source_name,source_manifest_sha256=EXPECTED[source_name],members=records,companions=companions,
       checkpoint_count=len(records),checkpoint_bytes=sum(r['source_stat']['bytes'] for r in records),
       companion_count=len(companions),companion_bytes=sum(r['staging_verification']['bytes'] for r in companions),
       elapsed_seconds=time.monotonic()-start,source_delete_allowed=False))
    print('BUNDLE_FULL_BYTES_PASS',name,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('bundle',choices=['new177-v1','jvp1k-v1']);a=ap.parse_args();verify(a.bundle)
