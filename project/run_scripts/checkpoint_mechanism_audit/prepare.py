"""Create an exact, isolated import tree; never execute native editing code."""
import argparse
import shutil
from pathlib import Path
from .common import *


def prepare():
    mapping=source_map(); out=ATTEMPT/'inputs/source'; out.mkdir(exist_ok=False)
    roots={'blue':S4OLD+'/blue-source/', 'helper':S4OLD+'/source-tech-r2/'}
    members=[]
    for source, local in mapping.items():
        for name, prefix in roots.items():
            if source.startswith(prefix):
                rel=Path(source[len(prefix):]); assert '..' not in rel.parts
                dst=out/name/rel; dst.parent.mkdir(parents=True,exist_ok=True)
                with open(local,'rb') as f, dst.open('xb') as g: shutil.copyfileobj(f,g)
                assert sha256(dst)==sha256(local)
                members.append(dict(source=source,path=str(dst),bytes=dst.stat().st_size,sha256=sha256(dst)))
    lock=read(mapped(S4RUN+'/execution.lock.json',mapping))
    cfg=mapped(lock['cells'][3]['config'],mapping)
    assert sha256(cfg)==lock['cells'][3]['config_sha256']
    receipt=dict(status='SOURCE_READY_NOT_MODEL_PASS',members=members,
        source_map_sha256=sha256(ATTEMPT/'inputs/source-map.json'),config=str(cfg),
        blue=str(out/'blue'),helper=str(out/'helper'),
        deps=str(ROOT/'local/fixed10k-preedit-eval/attempt-v1/deps-transformers-4.44.2'),
        authority_commit='4440e111251cc6a0a924e9e5b4589d61333f14dc',
        new_editing=0,z_optimization=0,save_checkpoints=False,
        cap_override_nonce='ODEEDIT-GH-SH2-CHECKPOINT-MECHANISM-CAP2-20260920-R1',
        lanes={'initial':'one common actual B1 prerequisite; CPU archival/geometry independent',
               'after_gate':['disjoint history LU/operator single-writer partitions',
                             'independent demand/activation after matching prerequisites']},
        resource=dict(project_cap=2,task_max_gpu=2,gpus_per_job=1,cpu_per_job=8,mem_mib_per_job=60416,
                      design_ram_gib=64,actual_request_gib=59,export='NONE',requeue=False))
    write_json(ATTEMPT/'inputs/source-ready.json',receipt)
    print('SOURCE_READY',len(members),flush=True)

if __name__=='__main__': prepare()
