"""CPU/tokenizer-only preparation for the Qwen PRICE eval-2K final run.

- Requests: fixed-10K first 2000 in file order, 20 packs of 100, exactly the
  baseline CF stream (server4 qwen-baselines-12 cf-stream.json) batch by batch.
- Config: resolver arm qwen25-Q3-beta250 (beta=c=beta_max=2.5, lr .1, lambda_N .001)
  on the QWEN_M1_CAP075 runtime axes, with M1 off.
- Rows: position-0 rule (eot.py). Unaffected rows must be byte-identical to the
  native pack; affected rows must be exactly [prefix] + native row, lookup + 1.
"""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from official.ours.config import resolve,profile,plain,PRICE_KEYS
from project.run_scripts.jlz_pilot import prompts
from project.run_scripts.jlz_realization.inputs import CounterFactAdapter
from project.run_scripts.jlz_realization.common import digest,member,require
from project.run_scripts.jlz_interference_l1.cap_storage import write,guard
from scripts.fixed_counterfact import load_prefix
from . import TASK,NONCE
from . import eot

OLD=Path('/data/janghj/ODE-edit/local/qwen-ours-m1-2k-20261008/preparation-v1')
CELL='QWEN_M1_CAP075'
BASELINE_STREAM=Path('/data/janghj/ODE-edit/local/qwen-baselines-12-20261009/execution-noqual-r1/streams/cf-stream.json')
BASELINE_LOCK=Path('/data/janghj/ODE-edit/local/qwen-baselines-12-20261009/execution-noqual-r1/streams/cf-stream.lock.json')
SERVER3_STREAM=Path('/data/janghj/ODE-edit/local/official-baselines-20261008/inputs/cf-stream.json')
ARM='Q3-beta250'
BATCHES=20
ALL_SEEN=(5,10,15,20)
RESERVE_BYTES=12*1024**3
OBSOLETE=PRICE_KEYS|{'beta_max_native_scale','clamp_factor','kl_factor','norm_factor','K_eval',
                     'eligible_layers','anchor_layer','nll_layer','lambda_C'}


def runtime_profile(cell):
    runtime={k:v for k,v in cell['arm_profiles'][CELL].items() if k not in OBSOLETE}
    runtime['price_m1_anchor_guard']=False
    return runtime


def unpadded(pack,key,row):
    n=int(pack[key]['attention_mask'][row].sum())
    return pack[key]['input_ids'][row,:n].tolist()


def compare(native,ruled,prefix_id):
    """Return affected rows; require all other rows unchanged."""
    affected=[]
    for row,(a,b) in enumerate(zip(native['lookup'],ruled['lookup'])):
        x,y=unpadded(native,'tokens',row),unpadded(ruled,'tokens',row)
        tn=native['targets'][row,:len(x)].tolist();tr=ruled['targets'][row,:len(y)].tolist()
        if x==y:
            require(a==b and tn==tr,'POS0_UNAFFECTED_ROW_CHANGED')
            continue
        require(a==0 and b==1 and y==[prefix_id]+x and tr==[-100]+tn,'POS0_AFFECTED_ROW_SHAPE')
        affected.append(dict(row=row,owner=native['row_request'][row],kind=native['row_kind'][row],
            canonical=row in native['canonical_rows'],case_id=native['specs'][native['row_request'][row]]['case_id']))
    for row,(a,b) in enumerate(zip(native['key_lookup'],ruled['key_lookup'])):
        x,y=unpadded(native,'key_tokens',row),unpadded(ruled,'key_tokens',row)
        if x==y:require(a==b,'POS0_UNAFFECTED_KEY_ROW_CHANGED')
        else:require(a==0 and b==1 and y==[prefix_id]+x,'POS0_AFFECTED_KEY_ROW_SHAPE')
    require(all(ruled['lookup'][r]>0 for r in range(len(ruled['lookup']))),'POS0_REMAINING')
    return affected


def prepare(out):
    out=Path(out);guard(out,RESERVE_BYTES)
    cell=json.loads((OLD/'config.json').read_text())['cells'][CELL]
    records=load_prefix(Path(cell['stream']).parent,2000)
    ids=[r['case_id'] for r in records]
    baseline=json.loads(BASELINE_STREAM.read_text())
    lock=json.loads(BASELINE_LOCK.read_text())
    require([r['case_id'] for r in baseline]==ids,'BASELINE_SAMPLE_ORDER')
    require(lock['batch_size']==100 and lock['requests']==2000 and
            lock['ordered_case_ids_sha256']==cell['ordered_ids_sha256'],'BASELINE_STREAM_LOCK')
    require([r['requested_rewrite'] for r in baseline]==[r['requested_rewrite'] for r in records],'BASELINE_REWRITES')
    official=json.loads(SERVER3_STREAM.read_text())
    require([r['case_id'] for r in official]==ids and
            [r['requested_rewrite'] for r in official]==[r['requested_rewrite'] for r in records],'SERVER3_BASELINE_SAMPLE')
    for row in cell['assets']:require(Path(row['path']).stat().st_size==row['bytes'],'ASSET_PRESENT')
    obs=Path(cell['observer_identity']['path'])
    import hashlib
    require(hashlib.sha256(obs.read_bytes()).hexdigest()==cell['observer_identity']['sha256'],'OBSERVER_IDENTITY_CONTENT')
    tokenizer=AutoTokenizer.from_pretrained(cell['model'],local_files_only=True)
    tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='right'
    require(tokenizer.bos_token_id is None,'QWEN_NO_BOS_EXPECTED')
    contexts=json.loads(Path(cell['contexts']).read_text())
    prefix_id=eot.check_tokenizer(tokenizer);ruled_prepare=eot.build()
    packs=[];affected=[]
    for i in range(BATCHES):
        chunk=records[i*100:(i+1)*100];requests=[r['requested_rewrite']|{'case_id':r['case_id']} for r in chunk]
        require([r['case_id'] for r in chunk]==cell['packs'][i]['ids'],'PARENT_PACK_ORDER')
        native=prompts.prepare(tokenizer,requests,contexts,'cpu')
        require(native['identity']==cell['packs'][i]['identity'],'NATIVE_PACK_IDENTITY')
        ruled=ruled_prepare(tokenizer,requests,contexts,'cpu')
        rows=compare(native,ruled,prefix_id)
        for row in rows:row['batch']=i+1
        affected+=rows
        packs.append(dict(batch=i+1,ids=[r['case_id'] for r in chunk],native_identity=native['identity'],
            identity=ruled['identity'],position0_rows=len(rows)))
    eot.install(tokenizer)
    bench=CounterFactAdapter(tokenizer,contexts)
    for i,pack in enumerate(packs):
        p=bench.prepare(records[i*100:(i+1)*100])
        require(p['identity']==pack['identity'] and p['entry_key_prefix_exact'],'ADAPTER_RULED_PACK')
    resolved=resolve('qwen25','qwen25-'+ARM)
    prof=plain(profile(resolved,runtime_profile(cell)))
    require((prof['beta_base'],prof['c'],prof['beta_max_scale'],prof['lr'],prof['lambda_N'])==(2.5,2.5,2.5,.1,.001)
            and prof['eligible_layers']==[4,5,6,7,8] and prof['price_m1_anchor_guard'] is False,'Q3_PROFILE')
    write(out/'resolved-profile.json',dict(resolved=plain(resolved),profile=prof),limit=1024**2)
    result=dict(task_id=TASK,instruction_id=NONCE,stage='CPU_INPUT_READY_GPU_NOT_SUBMITTED',
        parent_config=member(OLD/'config.json'),parent_cell=CELL,stream=member(cell['stream']),
        contexts=member(cell['contexts']),model=cell['model'],records=2000,
        ordered_ids_sha256=cell['ordered_ids_sha256'],slice_identity=digest(records),
        baseline_stream=member(BASELINE_STREAM),baseline_lock=member(BASELINE_LOCK),server3_baseline_stream=member(SERVER3_STREAM),baseline_sample_match=True,
        packs=packs,position0_rows=affected,position0_rule=dict(prefix=eot.PREFIX,prefix_id=prefix_id,
            prepare_source_sha256=eot.install(tokenizer)['prepare_source_sha256']),
        arm=ARM,resolved=member(out/'resolved-profile.json'),resolved_config_sha256=resolved['sha256'],
        observer_identity=member(obs),W0_policy='MEASURED_IN_RUN_ON_SERVER3_H200',W0_cold=cell['cold_W0_H0'],
        batches=BATCHES,all_seen=list(ALL_SEEN),reserve_bytes=RESERVE_BYTES,no_model_load=True,no_GPU=True)
    write(out/'preparation.json',result,limit=8*1024**2)
    print(json.dumps(dict(stage=result['stage'],packs=len(packs),position0_rows=len(affected),
        by_batch={p['batch']:p['position0_rows'] for p in packs if p['position0_rows']},
        kinds=sorted({(r['kind'],r['canonical']) for r in affected}))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);prepare(p.parse_args().out)
