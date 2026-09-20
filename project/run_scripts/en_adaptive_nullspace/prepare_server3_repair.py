"""Bind already verified SH3 assets and exact small RCA; CPU only."""
import argparse,json,os,sys,time
from pathlib import Path
from .prepare_server4 import sha
from .json_io import save
BASE=Path('/data/janghj/ODE-edit/local/en-adaptive-nullspace/20260920-v1')
READY=Path('/data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1')

def prepare(attempt):
    attempt=Path(attempt).absolute()
    if attempt != BASE/'server3-repair-r1/attempt-v1':raise ValueError('SCOPE')
    from .runtime import verify_ready_inputs
    from .metrics import contract_receipt
    from scripts.fixed_counterfact import load_prefix
    import numpy as np,torch,transformers,scipy
    manifest=json.loads((READY/'manifest.json').read_text())
    checked=verify_ready_inputs(manifest);contract=contract_receipt()
    records=load_prefix(manifest['dataset_root'],300)
    if (sys.version_info[:3],str(torch.__version__),transformers.__version__,np.__version__,scipy.__version__)!=((3,12,3),'2.9.1+cu128','4.44.2','2.2.6','1.15.3'):raise ValueError('VERSIONS')
    if torch.cuda.is_initialized():raise ValueError('CPU_ONLY')
    verified=[json.loads(s) for s in (BASE/'transfer-verified-members.jsonl').read_text().splitlines()]
    gp=BASE/'inputs/generated-v1';gm=json.loads((gp/'manifest.json').read_text());expect={}
    for d in gm['documents']:
        for kind in ('capsule','keys','residual','logp'):
            x=d[kind];expect[x['path']]=(x['bytes'],x['sha256'])
    if len(verified)!=2560 or len(expect)!=2560:raise ValueError('MEMBER_COUNT')
    for row in verified:
        p=gp/row['relative'];st=p.stat()
        if (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)!=(row['device'],row['inode'],row['size'],row['mtime_ns']):raise ValueError('STAT:'+str(p))
        if expect[row['relative']]!=(row['size'],row['sha256']):raise ValueError('MANIFEST_MEMBER')
    if gm['document_counts']!={'Dev128':128,'R512':512}:raise ValueError('ROLE_COUNTS')
    ri=BASE/'inputs/reference-inputs.json'
    if sha(ri)!='507b202108acc90e10810455f57da60df14e9e9e31ba567c8c75f51cf76afaeb':raise ValueError('REFERENCE_INPUT')
    bp=BASE/'inputs/pstar-derived-v1/basis.npy';br=json.loads(bp.with_name('receipt.json').read_text())
    basis=np.load(bp,mmap_mode='r',allow_pickle=False)
    if basis.shape!=(14336,14326) or basis.dtype!=np.float64 or bp.stat().st_size!=br['bytes']:raise ValueError('BASIS')
    # Single basis SHA binding for the new source path, no repeated heavy teacher hash.
    if sha(bp)!=br['sha256']:raise ValueError('BASIS_SHA')
    authority=Path(__file__).resolve().parents[3]/'audits/global/2026-09-20-sh3-en-adaptive-nullspace/authority-manifest.json'
    authority_value=json.loads(authority.read_text())
    for row in authority_value['members']:
        p=Path(__file__).resolve().parents[3]/row['path']
        if p.stat().st_size!=row['size'] or sha(p)!=row['sha256']:raise ValueError('AUTHORITY')
    manifest.update(node='ubuntu',python=sys.executable,python_version=list(sys.version_info[:3]),
        generated_root=str(gp),reference_inputs=str(ri),save_checkpoints=False,
        exact_crash_resume='NOT_AVAILABLE',maximum_batch=3,
        repair_provenance=dict(execution='b6e86234640ca127546aee094f2a67bbe684a490',
            analysis='1bb93e1d45d9144faf8af0aa9574a6d2c0dab438',RCA='9a371cb37112508fbdb5c366e0fe293da33d0bd9'))
    save(attempt/'inputs/manifest.json',manifest)
    save(attempt/'inputs/evaluator-map.json',{name:str(READY/name) for name in contract['contract']['source_files']})
    st=os.statvfs(attempt)
    if st.f_bavail*st.f_frsize<12*2**30 or st.f_favail<10000:raise ValueError('STORAGE')
    save(attempt/'cpu-binding.json',dict(status='CPU_PASS_NOT_MODEL_PATH_PASS',source_members=len(checked['source_members']),
        large_assets=checked['large_asset_count'],teacher_members=len(verified),teacher_bytes=sum(x['size'] for x in verified),
        verification='prior full SHA/size/shape plus unchanged destination stat and exact generated manifest members',
        generated_manifest_sha256=sha(gp/'manifest.json'),reference_inputs_sha256=sha(ri),
        basis=dict(path=str(bp),sha256=br['sha256'],bytes=br['bytes'],shape=list(basis.shape),dtype=str(basis.dtype)),
        document_counts=gm['document_counts'],position_counts=gm['position_counts'],
        first300_case_ids=[r['case_id'] for r in records],metrics_contract_sha256=contract['sha256'],
        authority_sha256=sha(authority),free_bytes=st.f_bavail*st.f_frsize,inodes_free=st.f_favail,
        CUDA_initialized=False,reference_transfer_bytes=0,new_checkpoint_bytes=0,time=time.time()))
    print(json.dumps(dict(status='CPU_PASS_NOT_MODEL_PATH_PASS',teacher_members=2560,free_bytes=st.f_bavail*st.f_frsize)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);a=p.parse_args();prepare(a.attempt)
