"""Immutable non-secret identity, separate from mutable delivery receipt."""
import json
import os
from pathlib import Path
import tempfile
from .schema import config, job_identity, require

def create(spool,cfg,startup,run_id):
    cfg=config(cfg)
    require(startup['run_id']==run_id and startup['config']==cfg,'STARTUP_CONFIG_IDENTITY')
    require(startup['job_identity']==job_identity(cfg),'STARTUP_JOB_IDENTITY')
    value=dict(run_id=run_id,url=startup['url'],run_name=startup['run_name'],config=cfg,
        job_identity=job_identity(cfg),startup_remote_identity_verified=True,
        scientific_completion_claim=False,transport_receipt='receipt.json')
    root=Path(spool);target=root/'identity.json'
    fd,tmp=tempfile.mkstemp(prefix='.identity-',dir=root)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(value,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.link(tmp,target) # Fail if any identity already exists; never replace.
    finally:os.unlink(tmp)
    return value
