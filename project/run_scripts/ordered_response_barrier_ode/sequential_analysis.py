"""Read-only, CPU-only audit and aggregation of the Server4 sequential run.

Reuses the accepted prompt-pair reducer; never imports the execution runtime.
The cumulative table is explicitly ONLINE_AT_EDIT_TIME, not a W_t retention test.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gzip
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import round0_analysis as prior
from .round0_analysis_contracts import (
    AnalysisBoundary, canonical_hash, member, sha256_file,
    verify_canonical_identity, write_json_once, STREAM_ROOT, ORDER_ROOT,
)

ARMS = ("O", "QCL", "NQFIX", "ORBFH", "JAC")
CELLS = ("LM", "LA", "QM", "QA")
ROOTS = ("tech-r1/results/task-0", "tech-r2/execution-2/results/task-1",
         "tech-r1/results/task-2", "tech-r2/execution-2/results/task-3")
JOBS = ("37151_0", "37214_1", "37151_2", "37214_3")
SOURCE = {
    "MEMIT": ("9b167455c1562be076aa2437fa63565187804aa7", "2af244e2dfe45d9fc8902d889b688b40231c6662"),
    "AlphaEdit": ("f47d035dc1ac35b1adcd8c036336ea69c5f32a4e", "ce6debc1f0b0d8b03b3c9355320251f169c7e09c"),
}
KINDS = {"rewrite_target_new":100, "rewrite_target_true":100,
         "rephrase_target_new":200, "rephrase_target_true":200,
         "locality_target_true":1000, "locality_target_new":1000}
NR = "NOT_RECORDED"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AnalysisBoundary(message)


def deep_identity(value: Any) -> int:
    count = 0
    if isinstance(value, dict):
        # References (journal/member/fixed-z bindings) contain another object's
        # identity; only schema-bearing payloads carry their own canonical root.
        if "identity_sha256" in value and "schema" in value:
            verify_canonical_identity(value)
            count += 1
        for item in value.values():
            count += deep_identity(item)
    elif isinstance(value, list):
        for item in value:
            count += deep_identity(item)
    elif isinstance(value, float):
        require(np.isfinite(value), "nonfinite raw scalar")
    return count


@contextmanager
def canonical_v2_reducer():
    """Bind v2 cardinalities only, restore legacy reducer globals exactly."""
    old = (prior.KIND_COUNTS, prior.KIND_ORDER, prior.EXPECTED_EVALUATION_ROWS)
    prior.KIND_COUNTS, prior.KIND_ORDER, prior.EXPECTED_EVALUATION_ROWS = KINDS, tuple(KINDS), 2600
    try:
        yield
    finally:
        prior.KIND_COUNTS, prior.KIND_ORDER, prior.EXPECTED_EVALUATION_ROWS = old


def stats(values) -> dict:
    a = np.asarray(list(values), dtype=np.float64)
    a = a[np.isfinite(a)]
    return {"n":len(a), **({k:float(v) for k,v in zip(
        ("mean","median","q25","q75","p90","max"),
        (a.mean(), *np.quantile(a, [.5,.25,.75,.9]), a.max()))} if len(a) else
        {k:None for k in ("mean","median","q25","q75","p90","max")})}


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    data = frame.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode()
    with path.open("xb") as out:
        if path.suffix == ".gz":
            with gzip.GzipFile(filename="", mode="wb", fileobj=out, mtime=0) as z:
                z.write(data)
        else:
            out.write(data)


def summarize_requests(frame: pd.DataFrame) -> dict:
    n = len(frame)
    row = {"request_n":n}
    specs = {"RS":("rewrite_success",1), "PS":("rephrase_prompt_success_count",2),
             "PS_strict":("rephrase_strict_success",1), "NS":("canonical_ns_numerator",10),
             "rewrite_acc":("rewrite_target_new_accuracy",1),
             "rephrase_acc":("rephrase_target_new_accuracy_count",2),
             "neighborhood_true_acc":("locality_target_true_accuracy_count",10)}
    for name,(column,mult) in specs.items():
        num = int(frame[column].sum())
        row.update({name+"_num":num, name+"_den":n*mult, name+"_rate":num/(n*mult)})
    if frame.locality_prediction_preservation_denominator.notna().all():
        num = int(frame.locality_prediction_preservation_numerator.sum())
        den = int(frame.locality_prediction_preservation_denominator.sum())
        row.update(PP_num=num, PP_den=den, PP_rate=num/den)
    for col in frame.columns:
        if "nll" in col or "margin" in col:
            row.update({col+"_"+k:v for k,v in stats(frame[col]).items()})
    return row


def evaluation(e: dict, order: list[str], meta: dict, endpoint: bool):
    rows, req, perf, summaries = prior._validate_evaluation(e, order, endpoint=endpoint)
    # Strict cardinality AND prompt order: the old reducer checks sets/pairs.
    for kind, n in KINDS.items():
        r = rows[rows.kind == kind]
        expected = [(s,p) for s in order for p in range(n//100)]
        require(list(zip(r.request_sha256, r.prompt_index)) == expected, "prompt/order mismatch")
    distributions = []
    pairs = []
    for category in ("rewrite", "rephrase", "locality"):
        new = rows[rows.kind == category+"_target_new"]
        true = rows[rows.kind == category+"_target_true"]
        keys = ["request_sha256", "case_id", "prompt_index"]
        p = new.merge(true, on=keys, suffixes=("_new","_true"), validate="one_to_one")
        p["margin_true_minus_new"] = p.nll_true-p.nll_new
        p["success"] = (p.nll_true<p.nll_new) if category=="locality" else (p.nll_new<p.nll_true)
        stored = e['locality']['canonical_ns'] if category=='locality' else e['preference'][category]
        require(canonical_hash(p.success.astype(bool).tolist()) == stored['bit_vector_sha256'], "success bit order/hash")
        q = p[[*keys,"nll_new","nll_true","margin_true_minus_new","success"]].copy()
        q["category"] = category
        pairs.append(q.assign(**meta))
        for target in ("new","true"):
            for unit,v in (("prompt",p["nll_"+target]),("request_cluster",p.groupby("request_sha256",sort=False)["nll_"+target].mean())):
                margin = p.margin_true_minus_new if unit=='prompt' else p.groupby('request_sha256',sort=False).margin_true_minus_new.mean()
                distributions.append({**meta,"category":category,"target":target,"unit":unit,
                    "prompt_den":len(p),"strict_num":int(p['all_tokens_correct_'+target].sum()),
                    **{"nll_"+k:v for k,v in stats(v).items()}, **{"margin_"+k:v for k,v in stats(margin).items()}})
    return req.assign(**meta), distributions, pd.concat(pairs,ignore_index=True)


def aggregate(base: Path, out: Path) -> dict:
    require(not out.exists(), "create-once analysis output already exists")
    out.mkdir(parents=True)
    inventories, provenance, requests, dist, prompt_pairs, batches, steps, layers, derived = [],[],[],[],[],[],[],[],[]
    cell_sources=[]; identity_count=0; reference_order=None; order_cases=None
    gate_rows=[]
    def load(p:Path, expected=None):
        nonlocal identity_count
        m=member(p,kind="sealed_raw")
        if expected is not None: require(m['sha256']==expected, f"file SHA mismatch: {p}")
        inventories.append(m)
        v=json.loads(p.read_text()); identity_count+=deep_identity(v)
        return v
    with canonical_v2_reducer():
        for cid,(cell,sub,job) in enumerate(zip(CELLS,ROOTS,JOBS)):
            root=base/sub
            term=load(root/'terminal-receipt.json')
            result=load(root/'result.json',term['result_sha256'])
            first=load(root/'first-valid-gate.json')
            require(result['identity_sha256']==term['result_identity_sha256'], "result internal binding")
            family=result['writer_family']; source=SOURCE[family]
            require((result['source']['head'],result['source']['tree'])==source, "source lineage")
            require(result['cell_id']==cid and result['status']=='SEQUENTIAL_CELL_TERMINAL_VALID', "cell terminal")
            require(result['arms']==list(ARMS) and result['primary_endpoint_count']==5000, "arm denominator")
            require(result['stream_root']==STREAM_ROOT and result['order_root']==ORDER_ROOT, "common stream")
            require(result['fixed_z_compute_count']==5000 and result['fixed_z_recompute_count']==0, "fixed z")
            require(result['full_fp32'] and result['terminal_w0_pointer_bytes_restore_pass'] and result['terminal_method_state_restore_pass'], "dtype/restore")
            for k in ('nonfinite_count','imputation_count','retry_count','technical_failure_count','scientific_failure_count','inter_arm_state_carry_count'):
                require(result[k]==0,f"{cell} {k}")
            dtype=result['dtype_receipt'];require(dtype['loaded_model_dtype']=='torch.float32' and not dtype['autocast_enabled'] and not dtype['quantized'] and dtype['bf16_fp16_cast_count']==0, 'FP32 receipt')
            cell_sources.append({"cell":cell,"root":str(root),"job":job,"result":{k:v for k,v in result.items() if k!='arm_receipts'},"terminal":term})
            provenance.append({"cell":cell,"model":result['model_alias'],"writer":family,"job":job,"result_root":str(root),
                "source_head":source[0],"source_tree":source[1],"result_sha256":term['result_sha256'],"terminal_identity":term['identity_sha256'],
                "easyedit_head":result['easyedit']['head'],"easyedit_tree":result['easyedit']['tree'],
                "cell_wall_seconds":result['wall_seconds'],"model_load_seconds":result['model_load_seconds'],
                "model_forward_count":result['model_forward_invocation_count'],
                "peak_allocated_GiB":result['memory']['peak_allocated_bytes']/2**30,
                "peak_reserved_GiB":result['memory']['peak_reserved_bytes']/2**30,"requests":5000,"batches":50,"full_fp32":True})
            common_entry=None;common_cache=None
            for ar in result['arm_receipts']:
                arm=ar['arm'];a=load(root/ar['relative_path'],ar['file_sha256'])
                require(a['identity_sha256']==ar['identity_sha256'] and a['journal_count']==10, "arm receipt")
                if common_entry is None:common_entry=a['entry_weight_sha256'];common_cache=a['entry_method_state_sha256']
                require(a['entry_weight_sha256']==common_entry and a['entry_method_state_sha256']==common_cache, "inter-arm entry isolation")
                last_w,last_cache=common_entry,common_cache
                order=[];cases=[]
                for idx,j in enumerate(a['journals']):
                    b=load(root/j['relative_path'],j['file_sha256'])
                    require(b['identity_sha256']==j['identity_sha256'] and b['batch_index']==idx and b['arm']==arm, "journal mapping")
                    require(b['request_count']==100 and len(set(b['request_sha256']))==100, "batch requests")
                    require(canonical_hash(b['request_sha256'])==b['request_order_sha256'], "batch order hash")
                    require(b['entry_weight_sha256']==last_w and b['entry_method_state_sha256']==last_cache,'W/cache chain')
                    last_w,last_cache=b['committed_weight_sha256'],b['committed_method_state_sha256']
                    require(b['fixed_z']['compute_count']==100 and b['fixed_z']['recompute_count']==0,'batch z count')
                    for k in ('controller_evaluator_influence_count','dynamic_z_recompute_count','retry_count','backtracking_count','fallback_count','nonfinite_count','imputation_count'):
                        require(b[k]==0,f"batch {k}")
                    require(b['full_fp32'] and b['status']=='SEQUENTIAL_BATCH_TERMINAL_VALID','batch terminal')
                    w=b['alpha_history_width']
                    if family=='AlphaEdit': require((w['entry'],w['exit'],w['append'])==(idx*100,(idx+1)*100,100),'Alpha cache width')
                    else:require(b['memit_covariance_static'] and w['append']==0 and last_cache==common_cache,'MEMIT static cov')
                    e=b['endpoint'];require(e['selected_weight_endpoint_sha256']==last_w,'endpoint commit binding')
                    require(e['fixed_z_identity_sha256']==b['fixed_z']['identity_sha256'],'endpoint fixed z')
                    order+=b['request_sha256'];cases+=b['case_ids']
                    meta={'cell':cell,'arm':arm,'batch':idx+1}
                    for stage,ev in [('ENTRY_CURRENT_COHORT',b['entry_evaluation']),('IMMEDIATE_POST',e['evaluation'])]:
                        rq,ds,ps=evaluation(ev,b['request_sha256'],{**meta,'stage':stage},stage=='IMMEDIATE_POST')
                        requests.append(rq);dist.extend(ds)
                        prompt_pairs.append(ps)
                    t=e['mechanism_telemetry']['telemetry'];f=e['endpoint_facts']['facts']
                    mech={k:v for k,v in t.items() if isinstance(v,(int,float,str,bool)) or v is None}
                    mech.update({prefix+'_'+k:v for prefix,d in [('adapter',e['adapter_counts']),('jvp',e['jvp_counts'])] for k,v in d.items()})
                    mech.update({**meta,'endpoint_status':e['status'],'wall_seconds':e['wall_seconds'],'model_forward_invocation_count':e['model_forward_invocation_count'],
                        'semantic_request_strict_count':t['terminal_semantic']['request_strict_count'],'semantic_tie_count':t['terminal_semantic']['strict_tie_count'],
                        'first_hit_prefix':(t.get('first_hit') or {}).get('prefix_length'),
                        'cache_entry_width':w['entry'],'cache_exit_width':w['exit'], 'raw_path':str(root/j['relative_path'])})
                    st=t.get('steps',[]);require(len(st)==(0 if arm=='O' else 20), 'node count')
                    if st:
                        mech.update({'q_terminal_'+k:v for k,v in stats(st[-1]['per_request_q_res']).items()})
                    batches.append(mech)
                    for s in st:
                        row={**meta,**{k:v for k,v in s.items() if not isinstance(v,(dict,list))}}
                        for key in prior.ARRAY_FIELDS:
                            require(len(s[key])==100, 'node request vector')
                            row.update({key+'_'+k:v for k,v in stats(x for x in s[key] if x is not None).items()})
                        row['potential_worsened_requests']=sum(s['per_request_actual_potential_worsened'])
                        steps.append(row)
                    energy=f['terminal_net_frobenius_squared_by_weight'];magnitudes={k:float(v)**.5 for k,v in energy.items()}
                    for name,v in magnitudes.items():
                        layers.append({**meta,'layer':int(name.split('.')[2]),'update_magnitude':v,'update_squared_norm':energy[name],
                            'update_magnitude_share':v/sum(magnitudes.values()),'update_squared_norm_share':energy[name]/sum(energy.values())})
                    de=b['derived_endpoint']
                    if de is not None:
                        dr={'cell':cell,'arm':'ORBHit','batch':idx+1,'status':de['status'],'state_version':de.get('state_version'),'separate_arm_denominator':0}
                        if de.get('evaluation') is not None:
                            rq,ds,ps=evaluation(de['evaluation'],b['request_sha256'],{**meta,'arm':'ORBHit','stage':'DERIVED_PREFIX'},True)
                            requests.append(rq);dist.extend(ds);prompt_pairs.append(ps);dr.update(summarize_requests(rq))
                        derived.append(dr)
                require(len(order)==1000 and len(set(order))==1000, 'arm unique requests')
                require(canonical_hash(order)==ORDER_ROOT,'all1000 order seal')
                require(last_w==a['terminal_weight_sha256'] and last_cache==a['terminal_method_state_sha256'],'terminal chain link')
                if reference_order is None:reference_order=order;order_cases=cases
                require(order==reference_order and cases==order_cases,'cross-cell/arm exact sample order')
                gate_rows.append({'cell':cell,'arm':arm,'batch_terminals':10,'requests':1000,'W_links':9,'cache_links':9,'fixed_z_compute':1000,
                    'recompute':0,'nonfinite':0,'restore':True,'entry_W':common_entry,'terminal_W':last_w,'entry_cache':common_cache,'terminal_cache':last_cache})
            print(f"FULL_REHASH_PASS {cell}: 5 arms / 50 batches / 5000 request endpoints",flush=True)
    req=pd.concat(requests,ignore_index=True);pp=pd.concat(prompt_pairs,ignore_index=True)
    bf=pd.DataFrame(batches);stf=pd.DataFrame(steps);lf=pd.DataFrame(layers)
    tables={'request-metrics.csv.gz':req,'batch-distributions.csv':pd.DataFrame(dist),'prompt-pairs.csv.gz':pp,
            'batch-mechanism-compute.csv':bf,'node-telemetry.csv.gz':stf,'layer-updates.csv':lf,
            'derived-prefix.csv':pd.DataFrame(derived),'cell-provenance-compute.csv':pd.DataFrame(provenance),'chain-integrity.csv':pd.DataFrame(gate_rows)}
    performance=[]
    for (c,a,b,s),g in req.groupby(['cell','arm','batch','stage'],sort=False):
        performance.append({'cell':c,'arm':a,'batch':b,'stage':s,**summarize_requests(g)})
    tables['batch-performance.csv']=pd.DataFrame(performance)
    cumulative=[];summary=[];distributions=[]
    for (c,a),g in req[req.stage=='IMMEDIATE_POST'].groupby(['cell','arm'],sort=False):
        summary.append({'cell':c,'arm':a,'scope':'ONLINE_AT_EDIT_TIME_10_BATCHES',**summarize_requests(g)})
        summary.append({'cell':c,'arm':a,'scope':'FINAL_W10_CURRENT_B10_ONLY',**summarize_requests(g[g.batch==10])})
        for end in range(1,11):
            cumulative.append({'cell':c,'arm':a,'through_batch':end,'evaluation_type':'ONLINE_AT_EDIT_TIME_PREFIX_NOT_CUMULATIVE_W_EVAL',**summarize_requests(g[g.batch<=end])})
    for (c,a,s,cat),g in pp.groupby(['cell','arm','stage','category'],sort=False):
        for scope,gg in [('ALL_BATCH_ENDPOINTS',g),('B10_ONLY',g[g.batch==10])]:
            for unit,ggg in [('prompt',gg),('request_cluster',gg.groupby('request_sha256',sort=False)[['nll_new','nll_true','margin_true_minus_new']].mean())]:
                distributions.append({'cell':c,'arm':a,'stage':s,'category':cat,'scope':scope,'unit':unit,
                    **{col+'_'+k:v for col in ['nll_new','nll_true','margin_true_minus_new'] for k,v in stats(ggg[col]).items()}})
    tables['performance-summary.csv']=pd.DataFrame(summary)
    tables['online-prefix-not-cumulative-W.csv']=pd.DataFrame(cumulative)
    tables['aggregate-distributions.csv']=pd.DataFrame(distributions)
    paired=[]
    fields=['rewrite_success','rephrase_prompt_success_count','rephrase_strict_success','canonical_ns_rate','rewrite_target_new_nll',
        'rewrite_target_true_nll','rephrase_target_new_nll_mean','rephrase_target_true_nll_mean','rewrite_margin_true_minus_new','rephrase_margin_true_minus_new_mean']
    for c in CELLS:
        g=req[(req.cell==c)&(req.stage=='IMMEDIATE_POST')]
        o=g[g.arm=='O']
        for a in ARMS[1:]:
            p=g[g.arm==a].merge(o,on=['batch','request_sha256'],suffixes=('_arm','_O'),validate='one_to_one')
            require(len(p)==1000,'paired denominator')
            for key in fields:
                d=p[key+'_arm']-p[key+'_O']
                paired.append({'cell':c,'arm':a,'reference':'O','metric':key,**stats(d),'positive':int((d>0).sum()),'zero':int((d==0).sum()),'negative':int((d<0).sum()),
                    'paired_request_order_sha256':canonical_hash(p.request_sha256.tolist())})
    tables['paired-versus-official.csv']=pd.DataFrame(paired)
    outliers=[]
    for (c,a),g in req[req.stage=='IMMEDIATE_POST'].groupby(['cell','arm'],sort=False):
        for metric in ('rewrite_target_new_nll','rephrase_target_new_nll_mean','locality_target_true_nll_mean'):
            for _,r in g.nlargest(5,metric).iterrows():
                outliers.append({'cell':c,'arm':a,'batch':int(r.batch),'request_sha256':r.request_sha256,'metric':metric,'value':r[metric]})
    tables['outliers.csv']=pd.DataFrame(outliers)
    for name,frame in tables.items():write_csv(out/name,frame)
    write_json_once(out/'raw-member-inventory.json',inventories)
    write_json_once(out/'source-and-runtime-provenance.json',cell_sources)
    # Save scalar metadata only, not weight snapshots / raw prompts.
    audit={'status':'FULL_REHASH_PASS','cells':4,'arms':20,'batches':200,'primary_request_endpoints':20000,
        'unique_common_requests':1000,'order_root':ORDER_ROOT,'stream_root':STREAM_ROOT,'deep_identity_count':identity_count,
        'raw_members':len(inventories),'raw_member_root':canonical_hash(inventories),'W_links':180,'cache_links':180,
        'fixed_z_compute':20000,'fixed_z_recompute':0,'terminal_restore':4,'nonfinite':0,'imputation':0,
        'final_W10_full1000_evaluation':'NOT_RECORDED_NO_WEIGHT_CHECKPOINT',
        'cumulative_Wt_seen_prefix_evaluation':'NOT_RECORDED_NO_WEIGHT_CHECKPOINT',
        'analysis_model_gpu_slurm_action':0,'scientific_promotion':False,
        'table_rows':{name:len(frame) for name,frame in tables.items()}}
    audit['identity_sha256']=canonical_hash(audit);write_json_once(out/'analysis-audit.json',audit)
    print(json.dumps(audit,ensure_ascii=False),flush=True)
    return audit


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();aggregate(args.raw_root,args.output)
