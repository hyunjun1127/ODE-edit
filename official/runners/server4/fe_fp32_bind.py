"""Bind actual server4 job to SH1's immutable standalone scientific core."""
import argparse
import json
import secrets
from pathlib import Path
from official.runners.server1.common import member, verify
from official.experiments.prepare import write_new


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--job-id',required=True)
    p.add_argument('--original-config',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--env-file',type=Path,required=True)
    p.add_argument('--attempt',type=Path,required=True)
    a=p.parse_args()
    if not a.job_id.isdigit() or int(a.job_id)<=0:raise ValueError('actual Slurm ID required')
    x=json.loads(a.inputs.read_text());assets=json.loads(a.assets.read_text())
    original=json.loads(a.original_config.read_text())
    if original['dataset']!='cf' or any(original['stream'][k]!=x['main_stream'][k] for k in ('sha256','bytes')):
        raise ValueError('original main provenance')
    # Resolve HF symlinks to existing regular blobs; never copy model assets.
    model_members=[]
    for m in assets['model']['members']:
        resolved=dict(path=str(Path(m['path']).resolve(strict=True)),bytes=m['bytes'],sha256=m['sha256'])
        verify(resolved);model_members.append(resolved)
    cfgrow=next(m for m in assets['model']['members'] if Path(m['path']).name=='config.json')
    snapshot=Path(cfgrow['path']).parent
    module=a.source/'official/diagnostics/fe_qwen_current/author_basic_eot_fp32.py'
    token=secrets.token_hex(32);out=a.attempt/'output'
    b=dict(schema='author-basic-eot-fp32-v1',dataset='cf',source=member(module),
           execution=dict(server='server4',job_id=a.job_id,source_commit=a.source_commit,
                          attempt=a.attempt.name,source_run_id='820f241a015b4397',admission_id=token),
           author_yaml=x['author_yaml'],native=x['native'],native_config=x['native_config'],
           main_stream=x['main_stream'],original_main_config=member(a.original_config),C0=x['C0'],
           official_members=[member(f) for f in sorted((a.source/'official').rglob('*.py'))],
           model=dict(snapshot=str(snapshot),members=model_members,config=member((snapshot/'config.json').resolve())),
           runtime=json.loads(a.runtime.read_text()),tracking_env_file=str(a.env_file),
           storage=dict(reserve_bytes=8*1024**3,payload_bound=8568963072,
                        raw_margin_bytes=5*1024**3,
                        lock_path=str(a.attempt.parent/'checkpoint-host.lock'),
                        admitted=[dict(output=str(out),config_sha256=token)]))
    write_new(a.attempt/'bindings.json',b)
    write_new(a.attempt/'host-binding-receipt.json',dict(job_id=a.job_id,bindings=member(a.attempt/'bindings.json'),
         raw_margin_own_bytes=3*1024**3,unrelated_nockpt_growth_reservation_bytes=2*1024**3,
         reference_run_id='820f241a015b4397',source_commit=a.source_commit,output=str(out)))
    print(json.dumps(dict(job_id=a.job_id,bindings=str(a.attempt/'bindings.json'),output=str(out))))


if __name__=='__main__':main()
