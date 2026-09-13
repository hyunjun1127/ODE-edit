"""Approved source-only exact Git-object acquisition; no S4 live/raw reads."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

REV='311b076a92e4ed0f14f5c8b4909732da781bc5f7'
TREE='f3c933c31cba2fe979c5c34546a99a72e6beb763'
DEST=Path('/mnt/raid5/janghj/ODE-edit/local/multilayer-joint-compensation/20260911-v1/imports/native-source-v1')
FILES=[
 'AlphaEdit/AlphaEdit_main.py','AlphaEdit/AlphaEdit_hparams.py','AlphaEdit/compute_ks.py','AlphaEdit/compute_z.py','AlphaEdit/__init__.py',
 'memit/memit_main.py','memit/memit_hparams.py','memit/compute_ks.py','memit/compute_z.py','memit/__init__.py',
 'rome/layer_stats.py','rome/tok_dataset.py','rome/repr_tools.py',
 'util/__init__.py','util/logit_lens.py','util/nethook.py','util/generate.py','util/globals.py','util/hparams.py','util/runningstats.py',
 'hparams/AlphaEdit/Llama3-8B.json','hparams/MEMIT/Llama3-8B.json','globals.yml','LICENSE','README.md']


def remote(command):
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','rke-server4',command],capture_output=True,check=True,timeout=120)
    return r.stdout


def main():
    if DEST.exists():raise FileExistsError(f'CREATE_ONCE_SOURCE_EXISTS: {DEST}')
    identity=remote(f'git -C /data/janghj/BLUE show -s --format="%H %T" {REV}').decode().strip()
    if identity!=f'{REV} {TREE}':raise ValueError('SOURCE_GIT_IDENTITY_MISMATCH')
    listing=remote(f'git -C /data/janghj/BLUE ls-tree -r {REV} -- '+ ' '.join(FILES)).decode().splitlines()
    blobs={}
    for line in listing:
        fields,path=line.split('\t');mode,kind,oid=fields.split()
        if kind!='blob' or mode not in ('100644','100755'):raise ValueError('NONREGULAR_GIT_SOURCE')
        blobs[path]=(mode,oid)
    if set(blobs)!=set(FILES):raise ValueError('GIT_ALLOWLIST_MISSING')
    payload=remote(f'git -C /data/janghj/BLUE archive --format=tar {REV} -- '+ ' '.join(FILES))
    DEST.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    stage=Path(tempfile.mkdtemp(prefix='.native-source-v1-',dir=DEST.parent))
    members=[]
    with tarfile.open(fileobj=io.BytesIO(payload),mode='r:') as archive:
        actual=[m for m in archive.getmembers() if not m.isdir()]
        if sorted(m.name for m in actual)!=sorted(FILES):raise ValueError('ARCHIVE_ALLOWLIST_MISMATCH')
        for m in actual:
            if not m.isfile():raise ValueError('ARCHIVE_NONREGULAR')
            data=archive.extractfile(m).read();mode,oid=blobs[m.name]
            observed_blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            if observed_blob!=oid:raise ValueError('GIT_OBJECT_BYTES_MISMATCH')
            path=stage/m.name;path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
            with path.open('xb') as f:f.write(data)
            path.chmod(0o400)
            digest=hashlib.sha256(data).hexdigest()
            if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('DESTINATION_SHA_MISMATCH')
            members.append(dict(path=m.name,bytes=len(data),sha256=digest,git_blob=oid,git_mode=mode))
    manifest=dict(status='SOURCE_GIT_BYTES_VERIFIED',repository='rke-server4:/data/janghj/BLUE',revision=REV,tree=TREE,
        members=members,member_count=len(members),total_bytes=sum(m['bytes'] for m in members),
        acquisition='EXPLICIT_GIT_ARCHIVE_ALLOWLIST',live_source_read=0,raw_checkpoint_access=0,
        model_or_GPU=0,Slurm=0,shared_source_mutation=0,
        imports_policy='namespace wrappers may bypass unrelated __init__; record exact executed closure separately',
        globals_policy='original bytes preserved; platform-path binding only in private adapter')
    with (stage/'source-manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
    (stage/'source-manifest.json').chmod(0o400)
    if DEST.exists():raise FileExistsError('DESTINATION_PUBLISH_CONFLICT')
    os.rename(stage,DEST)
    print(json.dumps(dict(path=str(DEST),revision=REV,tree=TREE,count=len(members),
        manifest_sha256=hashlib.sha256((DEST/'source-manifest.json').read_bytes()).hexdigest())))


if __name__=='__main__':main()
