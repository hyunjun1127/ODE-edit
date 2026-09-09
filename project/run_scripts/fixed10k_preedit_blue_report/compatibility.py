"""Source and metadata identity comparison; no cross-hardware forward parity claim."""
import argparse
import csv
import json
import shlex
import subprocess
from pathlib import Path
from .reduce import read, save, write_csv, sha, require

REMOTE = r'''
import sys,json,hashlib
from pathlib import Path
out=[]
for m in json.load(sys.stdin):
 p=Path(m['path']);b=p.read_bytes()
 assert len(b)==int(m['bytes']) and hashlib.sha256(b).hexdigest()==m['sha256']
 r=json.loads(b)
 out.append(dict(arm=m['arm'],input=m,model_revision=r['model_revision'],W0=r['W0'],
 writer_tokenizer=r['writer_tokenizer'],evaluator_tokenizer=r['evaluator_tokenizer'],
 **{k:r[k] for k in ['gpu','torch','transformers','attention','tf32_matmul','tf32_cudnn','dtype','autocast']}))
print(json.dumps(out))
'''


def run(task, v2, reduction, out):
    require(not out.exists(),'CREATE_ONCE')
    inv=list(csv.DictReader((v2/'raw-member-inventory.csv').open()))
    refs=[{k:m[k] for k in ('arm','path','bytes','sha256')} for m in inv if m['path'].endswith('/runtime.json')]
    require(len(refs)==6,'RUNTIME_SIX')
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','rke-server4','python3 -c '+shlex.quote(REMOTE)],input=json.dumps(refs),text=True,capture_output=True,timeout=60)
    require(r.returncode==0,r.stderr)
    blue=json.loads(r.stdout);w0=read(reduction/'runtime-copy.json');lock=read(task/'execution.lock.json')
    source=list(csv.DictReader((v2/'source-member-inventory.csv').open()));by_path={r['path']:r for r in source}
    matched=[]
    for m in lock['members']:
        other=by_path.get(m.get('source_path',''))
        if other:
            require(m['sha256']==other['sha256'] and m['bytes']==int(other['bytes']),'COMMON_SOURCE_ASSET_MISMATCH '+m['path'])
            matched.append(dict(server2_path=m['path'],server4_path=other['path'],sha256=m['sha256'],bytes=m['bytes']))
    essential=['config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json',
      'alphaedit_strength_neutral_barrier/evaluator.py','blue_alphaedit_sequential_comparison/evaluation.py',
      'ordered_response_barrier_ode/counterfact_locality_evaluator.py']
    for suffix in essential:require(any(m['server4_path'].endswith(suffix) for m in matched),'UNBOUND_SOURCE '+suffix)
    require(sum(m['server4_path'].endswith('.safetensors') for m in matched)==4,'FOUR_WEIGHT_SHARD_IDENTITIES')
    rows=[]
    for b in blue:
        for key,sig in b['W0']['weights'].items():
            require(sig['sha256']==w0['selected_W0']['weights'][key]['sha256'],'WRONG_W0 '+b['arm']+' '+key)
        for key in ('model_revision','torch','transformers','attention','tf32_matmul','tf32_cudnn'):
            require(b[key]==w0[key],'RUNTIME_CONTRACT '+key)
        require(b['dtype']=='torch.float32' and b['autocast'] is False,'FP32')
        require(b['evaluator_tokenizer']['padding']==w0['tokenizer']['padding_side'] and b['evaluator_tokenizer']['pad']==w0['tokenizer']['pad'],'TOKENIZER_POLICY')
        rows.append(dict(arm=b['arm'],entry_selected_W0_SHA_match=True,selected_W0_keys=len(b['W0']['weights']),
            model_revision=b['model_revision'],shared_source_asset_matches=len(matched),evaluator_kernel_SHA='4f5af6dbf8854c79aedc5e36134fddfcb2995b60f05d270604f8ea735672d93e',
            padding='manual left padding; tokenizer property right; explicit attention_mask; position_ids native default',
            target_alignment='same sealed tokenizer/contracts and evaluator target mask',microbatch=16,seed=lock['seed'],
            dtype='torch.float32',torch=b['torch'],transformers=b['transformers'],attention=b['attention'],
            tf32_matmul=b['tf32_matmul'],tf32_cudnn=b['tf32_cudnn'],server2_gpu=w0['gpu'],server4_gpu=b['gpu'],
            cross_hardware_forward_parity='NOT_TESTED_WARN',nonselected_W0_full_bytes='NOT_CLAIMED',
            evaluator_add_bos='NOT_EXPOSED at runtime; tokenizer asset/code identity only',
            context_generation_W0=0,edit_contexts='BLUE native contexts; W0 evaluation uses sealed prompts only; not a W0 target context comparison'))
    out.mkdir(parents=True)
    save(out/'blue-runtime-extracts.json',blue)
    write_csv(out/'w0-blue-compatibility.csv',rows)
    write_csv(out/'shared-source-asset-identity.csv',matched)
    save(out/'compatibility-validation.json',dict(status='PASS_WITH_CROSS_HARDWARE_WARN',selected_W0_match_arms=6,
       common_source_asset_count=len(matched),cross_hardware_parity='NOT_TESTED',full_nonselected_parameter_bytes='NOT_CLAIMED',
       six_runtime_refs=refs,source_inventory_sha256=sha(v2/'source-member-inventory.csv'),
       new_model_GPU_evaluator_Slurm_actions=0))
    print(json.dumps(rows,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('task','v2','reduction','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.task,a.v2,a.reduction,a.out)
