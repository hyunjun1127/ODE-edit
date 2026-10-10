"""CPU host evidence, explicitly not shared native integration/GPU PASS."""
import json,os,subprocess
from pathlib import Path
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from official.experiments.prepare import read,write_new,file_sha
from official.evaluation.zsre_query_parity import compare_queries
from official.runners.server2.submit import inventory,dependency_ids
from official.runners.server1.submit import graph_width
B=Path('/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011');O=Path(__file__).parent
prep=read(B/'host-preparation.json');hp=OmegaConf.load(prep['author_yaml']['path']);alg=OmegaConf.load(prep['algs_yaml']['path'])
assert (hp.clamp_norm_factor,hp.v_num_grad_steps,hp.v_lr,hp.v_weight_decay)==(1,35,.5,.001)
assert hp.layers==[4,5,6,7,8] and alg.add_old_keys and alg.L2==0
model=read('/mnt/raid5/janghj/ODE-edit/local/qwen-baseline-mask-cold-rerun-20261010/registration-r1/assets.json')['model_snapshot']
tok=AutoTokenizer.from_pretrained(model,local_files_only=True);tok.padding_side='right';tok.pad_token=tok.eos_token
q=compare_queries(tok,read(prep['streams']['zsre']['path']),model_family='qwen25')
assert q['requests']==2000 and q['input_mismatches']==q['target_mismatches']==0
live=inventory();assert not live['ambiguous']
assert not {'62532','62868','62872'} & {r['job'] for r in live['project']}
width=graph_width([dict(key=r['job'],gpus=r['gpus'],parents=dependency_ids(r['dependency'])) for r in live['project']])
assert width<=2
caps=Path('/mnt/raid5/janghj/ODE-edit/servers/local/gpu-caps.tsv').read_text();assert '\t2\t60416\t' in caps
write_new(O/'host-checks.json',dict(author_yaml_PASS=True,zsre_query_parity=q,CPU_only=True,GPU_forward=0,
  native_W0_z_integration_tests='NOT_RUN_SHARED_SOURCE_PENDING',checkpoint_roundtrip='NOT_RUN_SHARED_SOURCE_PENDING',
  project_width=width,allocated_GPU=sum(r['allocated_GPUs'] for r in live['project']),live=live,
  exact_FE_cancelled=['62532','62868','62872'],cap2_unchanged=True,
  preparation=dict(path=str(B/'host-preparation.json'),sha256=file_sha(B/'host-preparation.json'))))
print(json.dumps(dict(CPU_query=q['status'],width=width,allocated=sum(r['allocated_GPUs'] for r in live['project']))))
