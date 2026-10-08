"""Task-private final schedule extension; immutable SH1 helper is read-only.

The shared identity validator does not yet register generation_schedule. Only
the explicit final profile uses this strict extension. All previous callers
continue through the original shared create function, unchanged.
"""
import json
import os
from pathlib import Path
import tempfile
from project.run_scripts.experiment_tracking.identity import create as shared_create
from .generation_tracking_schema import config, job_identity, require


def create(spool,cfg,startup,run_id):
    if cfg.get('attempt')!='final-generation-v1':
        return shared_create(spool,cfg,startup,run_id)
    cfg=config(cfg)
    require(startup['run_id']==run_id and startup['config']==cfg,'STARTUP_CONFIG_IDENTITY')
    require(startup['job_identity']==job_identity(cfg),'STARTUP_JOB_IDENTITY')
    value=dict(run_id=run_id,url=startup['url'],run_name=startup['run_name'],config=cfg,
        job_identity=job_identity(cfg),startup_remote_identity_verified=True,
        scientific_completion_claim=False,transport_receipt='receipt.json',
        identity_schema='GPTJ_FINAL_GENERATION_PRIVATE_SCHEDULE_EXTENSION_V1')
    root=Path(spool);target=root/'identity.json'
    fd,tmp=tempfile.mkstemp(prefix='.identity-',dir=root)
    try:
        with os.fdopen(fd,'w') as stream:
            json.dump(value,stream,sort_keys=True,allow_nan=False)
            stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.link(tmp,target)
    finally:
        os.unlink(tmp)
    return value
