"""CPU-only exact common/input/source binding; no asset regeneration."""
import argparse
import dataclasses
import json
from pathlib import Path
import stat
import subprocess
import time
import torch
from ..contracts import MODEL,DATA,Science,sha,digest,save,tensor_sha
from .runtime import verify_bundle

OLD_INPUT=Path('/mnt/raid5/janghj/ODE-edit/local/blue-l4-progress-barrier/attempt-v1/input.lock.json')
ASSET_ROOT=Path('/mnt/raid5/janghj/ODE-edit/local/blue-l4-progress-barrier/attempt-v1/imports')

def main():
    p=argparse.ArgumentParser();p.add_argument('--common',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8);start=time.monotonic()
    source,receiver=verify_bundle(a.common)
    source_ready=json.loads((source/'READY.json').read_text())
    if digest(source_ready['members'])!=source_ready['members_root']:raise ValueError('COMMON_MEMBER_ROOT')
    entry=torch.load(source/'entry.pt',map_location='cpu',weights_only=True,mmap=True)
    if entry['schema']!='common-entry-v1':raise ValueError('COMMON_SCHEMA')
    weights={}
    for key in ('We','WN','W0'):
        if len(entry[key])!=2:raise ValueError('COMMON_TWO_WEIGHTS')
        weights[key]=[]
        for w in entry[key]:
            if w.dtype!=torch.float32 or tuple(w.shape)!=(4096,14336) or not torch.isfinite(w).all():raise ValueError('COMMON_WEIGHT')
            weights[key].append(tensor_sha(w))
    if weights['We'][1]!=weights['WN'][1] or weights['W0'][1]!=weights['WN'][1]:raise ValueError('WN8_UNEDITED')
    histories={}
    for key in ('M4','M8'):
        t=entry[key]
        if t.dtype!=torch.float32 or tuple(t.shape)!=(14336,14336) or not torch.isfinite(t).all():raise ValueError('COMMON_HISTORY')
        histories[key]=tensor_sha(t)
    packed=torch.load(source/'prediction-rows.pt',map_location='cpu',weights_only=True)
    banks=json.loads((source/'bank-manifest.json').read_text())
    for role,rows in packed.items():
        if digest(rows)!=banks['packing_identity'][role]:raise ValueError('PACKING_IDENTITY')
        teacher=torch.load(source/f'We-teacher-{role}.pt',map_location='cpu',weights_only=True)
        if [r['identity'] for r in rows]!=[r['identity'] for r in teacher]:raise ValueError('TEACHER_ORDER')
        if abs(sum(r['context_weight'] for r in rows)-1)>1e-10:raise ValueError('GLOBAL_REDUCTION')
        for r,t in zip(rows,teacher):
            if list(t['target_ids'])!=r['target_ids'] or not torch.isfinite(t['logp']).all():raise ValueError('TEACHER_ALIGNMENT_FINITE')
        del teacher
    from scripts.fixed_counterfact import load_prefix
    records=load_prefix(DATA,10000)
    inv=entry['raw_effective_inventory']
    if inv!=banks['inventory']:raise ValueError('COMMON_INVENTORY')
    calibration=json.loads((source/'calibration.json').read_text())
    if calibration!=source_ready['calibration']:raise ValueError('CALIBRATION_IDENTITY')
    old=json.loads(OLD_INPUT.read_text());assets=[]
    for m in old['model_members']+old['dependencies']:
        path=Path(m['path']);resolved=path.resolve(strict=True)
        if path.is_symlink():resolved.relative_to(MODEL.parent.parent/'blobs')
        expected=m['sha256'];inherited=False
        if expected=='INHERITED_SNAPSHOT_REVISION_NO_DUPLICATE_MULTIGB_HASH':
            # Historical receipt intentionally did not hash the weight shards.
            # Independently verify the current HF LFS blob digest, not that text.
            if not path.is_symlink() or len(resolved.name)!=64 or any(c not in '0123456789abcdef' for c in resolved.name):
                raise ValueError('UNRESOLVED_HF_LFS_BLOB_IDENTITY')
            expected=resolved.name;inherited=True
        actual=sha(path)
        if not stat.S_ISREG(resolved.stat().st_mode) or resolved.stat().st_size!=m['bytes'] or actual!=expected:
            raise ValueError('EXACT_PINNED_ASSET_CHANGED:'+str(path))
        assets.append(dict(m,sha256=actual,resolved_path=str(resolved),prior_revision_only_receipt=inherited))
    # Native key finalization imports only immutable previously transferred code.
    native=[]
    for path in sorted((ASSET_ROOT/'imports/blue-source').rglob('*')):
        if path.is_file() and path.suffix in ('.py','.yml'):
            if path.is_symlink():raise ValueError('NATIVE_SOURCE_SYMLINK')
            native.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)))
    path=ASSET_ROOT/'imports/config.json';native.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)))
    for n in ('compute_ks.py','compute_z.py','AlphaEdit_main.py'):
        path=ASSET_ROOT/'imports/blue-source/AlphaEdit'/n
        if sha(path)!=entry['source_identity']['native'][n]:raise ValueError('SH1_SH2_NATIVE_SOURCE_MISMATCH:'+n)
    receipt=dict(status='COMMON_CPU_SCHEMA_AND_LOCAL_ASSET_PASS',common=str(Path(a.common).absolute()),
        common_receiver_sha=sha(Path(a.common)/'receiver-ready.json'),source_ready_sha=receiver['source_ready_sha256'],
        members_root=source_ready['members_root'],common_source_head=receiver['source_head'],weights=weights,histories=histories,
        packing_identity=banks['packing_identity'],inventory=inv,calibration=calibration,
        prior_asset_lock=dict(path=str(OLD_INPUT),sha256=sha(OLD_INPUT)),assets=assets,native_source=native,
        science=dataclasses.asdict(Science()),model_revision=MODEL.name,dataset_records=len(records),
        local_torch=torch.__version__,torch_CUDA_initialized=torch.cuda.is_initialized(),
        full_space_geometry='EXACT_SOURCE_SHA_BOUND; from_state numerical checks at runtime',
        WN_current_teacher='NOT_IN_COMMON; B_LOCAL_CAPTURE_ONCE_AT_EXACT_WN',
        runtime_model_parity='PENDING_ACTUAL_FIRST_GATE',project_cap=2,mem_mib=60416,
        wall_seconds=time.monotonic()-start,scientific_promotion=False)
    save(out/'input.lock.json',receipt)
    print(json.dumps(dict(status=receipt['status'],path=str(out/'input.lock.json'),sha256=sha(out/'input.lock.json'))),flush=True)

if __name__=='__main__':main()
