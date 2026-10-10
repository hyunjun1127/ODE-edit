"""Single W20 saved-weight evaluation; no edits, W0, generation or CP writes."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

from official.experiments.prepare import digest, file_sha, read, write_new
from official.runners.server2.zsre_reeval_restore import require, verify_member, load_and_restore

INSTRUCTION="USER-GH-ZSRE-SAVED-WEIGHTS-2K-REEVAL-20261009-R1"
EVALUATOR_SHA="d6a5b34eafd27660a2dee4632c638b6bf4c3614246071711cf5159a002415a45"
PARITY_SHA="9883f16036525278bbfcdb45f5f08cc8c799f638a26f91b4d9da2a0d70103803"
TRACKING_SHA="633344063046eba678595e436b4ff3d3162a062e9b31ff244bc48ab09a9237f1"
REPO=Path(__file__).resolve().parents[3]
AUDIT=REPO/"audits/servers/server2/zsre-2k-reeval-20261009/inventory.json"
METHODS=("FT","MEMIT","ALPHAEDIT","ALPHAEDIT_BLUE","MEMIT_FE","SPHERE")
COUNTS=dict(rewrite=5557,paraphrase=5557,neighborhood=9694)

def member(path):
    p=Path(path)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=file_sha(p))

def check_stat(row):
    s=Path(row["path"]).stat()
    for k,v in (("bytes",s.st_size),("device",s.st_dev),("inode",s.st_ino),("mtime_ns",s.st_mtime_ns)):
        require(k not in row or int(row[k])==v,"SEALED_INPUT_STAT_CHANGED_"+k)

def runtime():
    return {k:importlib.metadata.version(k) for k in ("torch","transformers","numpy","scipy")}

def tokenizer(snapshot):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,use_fast=True)
    tok.padding_side="right"
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    require(tok.pad_token_id==tok.eos_token_id==50256,"GPTJ_TOKENIZER_PAD_EOS")
    return tok

def cpu_prepare(out):
    from official.evaluation.zsre_query_parity import compare_queries
    for rel,sha in (("evaluation/zsre_paper.py",EVALUATOR_SHA),
                    ("evaluation/zsre_query_parity.py",PARITY_SHA),
                    ("tracking/schema.py",TRACKING_SHA)):
        require(file_sha(REPO/"official"/rel)==sha,"COMMON_INPUT_SHA")
    inventory=read(AUDIT)
    require(len(inventory["rows"])==6 and all(r["status"]=="FINAL_W20_FULLSHA_VERIFIED"
        for r in inventory["rows"]),"FINAL_CP_INVENTORY")
    manifest=read(verify_member(inventory["rows"][0]["original_manifest"]))
    bases=[]
    for item in manifest["model_assets"]:
        check_stat(item)
        if item["bytes"]<10*1024**2:verify_member(item)
        bases.append(dict(item,verification="CURRENT_FULL_SHA" if item["bytes"]<10*1024**2
            else "PRIOR_FULL_SHA_PLUS_CURRENT_UNCHANGED_STAT"))
    stream=manifest["streams"]["zsre"]["member"]
    records=read(verify_member(stream))
    require(len(records)==2000 and [r["occurrence_index"] for r in records]==list(range(1,2001)),
            "ORDERED_2K_STREAM")
    tok=tokenizer(manifest["model_snapshot"])
    proof=compare_queries(tok,records,model_family="gptj")
    require(proof["token_denominators"]==COUNTS and proof["requests"]==2000,"QUERY_COUNTS")
    rows=[]
    for old in inventory["rows"]:
        row=dict(old)
        check_stat(row["checkpoint"])
        require(read(verify_member(row["latest"]))["sha256"]==row["checkpoint"]["sha256"],
                "FINAL_POINTER_CHANGED")
        path=Path(row["original_manifest"]["path"]).parent/"source/official/hparams"/row["method"]/"gptj.json"
        row["hparams"]=member(path)
        require(row["identity"]==manifest["checkpoint_identities"]["zsre"][row["method"]],
                "ORIGINAL_CONFIG_IDENTITY")
        rows.append(row)
    inputs=dict(instruction_id=INSTRUCTION,rows=rows,stream=stream,model_assets=bases,
        model_snapshot=manifest["model_snapshot"],model_revision=manifest["model_revision"],
        tokenizer_sha256=manifest["tokenizer_sha256"],tokenizer_files_sha256=manifest["tokenizer_files_sha256"],
        runtime=runtime(),query_proof=proof,evaluator_sha256=EVALUATOR_SHA,
        common_publication="4533756ea5b731793595d09822b1e66dc4458395",
        original_manifest=inventory["rows"][0]["original_manifest"])
    write_new(Path(out)/"inputs.json",inputs)
    print(json.dumps(dict(status="CPU_PREPARED_NOT_SUBMITTED",proof=proof)))

def tracking_config(row,inputs,source,config_sha,attempt):
    return dict(server="server2",task_id="zsre-2k-reeval-20261009",arm=row["method"],attempt=attempt,
        source_sha=source,config_sha=config_sha,model="gptj",model_family="gptj",writer=row["method"],
        role="eval_only",metric_schema="official-baselines-scalar-v1",dataset="zsre",
        instruction_id=INSTRUCTION,evaluation_profile="zsre-public-query-W20-only-v1",
        checkpoint_sha256=row["checkpoint"]["sha256"],evaluator_sha256=EVALUATOR_SHA,
        stream_sha256=inputs["stream"]["sha256"],tokenizer_sha256=inputs["tokenizer_sha256"],
        source_run_id=str(row["original_job_id"]))

def verify_source(root):
    lock=read(root/"source-lock.json")
    for entry in lock["members"]:
        require(file_sha(root/"source"/entry["path"])==entry["sha256"],"FROZEN_SOURCE_CHANGED")
    for entry in read(root/"input-lock.json")["members"]:
        verify_member(entry)
    return lock

def validate_result(raw,inputs,*,model_family='gptj',counts=None):
    counts=COUNTS if counts is None else counts
    require(raw["schema"]=="official-zsre-public-query-eval-only-v1","EVAL_SCHEMA")
    require(raw["model_family"]==model_family and len(raw["cases"])==2000,"EVAL_FULL_2K")
    require(raw["query_sha256"]==inputs["query_proof"]["query_sha256"],"RUNTIME_QUERY_MISMATCH")
    require(raw["token_denominators"]==counts,"RUNTIME_DENOMINATORS")
    require(raw["work"]["queries"]==sum(counts.values()),"RUNTIME_QUERY_COVERAGE")
    require(raw["model_no_mutation"] is True and raw["RNG_restored"] is True,"EVAL_STATE_GUARD")
    cases=raw["cases"]
    require([c["occurrence_index"] for c in cases]==list(range(1,2001)),"RESULT_ORDER")
    summary={}
    for group,label in (("rewrite","Efficacy"),("paraphrase","Generalization"),("neighborhood","Specificity")):
        means=[];tokens=0
        for c in cases:
            obs=c[group+"_observations"]
            bits=[x["predicted_token_id"]==x["target_token_id"] for x in obs]
            require(bits and bits==c[group+"_prompts_correct"] and
                    bits==[x["correct"] for x in obs],"RAW_CORRECTNESS")
            means.append(math.fsum(bits)/len(bits));tokens+=len(bits)
        require(tokens==counts[group],"RAW_TOKEN_DENOMINATOR")
        summary[label]=100*math.fsum(means)/2000
        require(math.isfinite(summary[label]) and
                abs(summary[label]-raw["summary"][label])<1e-10,"RAW_REDUCER_MISMATCH")
    require(raw["summary"]["Specificity_loc_ans"]==raw["summary"]["Specificity"],"LOC_ALIAS")
    require(raw["summary"]["requests"]==2000 and "W0_prediction_agreement" not in raw["summary"],"NO_W0")
    return summary

def execute(root,method):
    import torch
    from transformers import AutoModelForCausalLM
    from official.evaluation.zsre_paper import evaluate
    from official.tracking import init,official_zsre_metrics
    root=Path(root).resolve();lock=verify_source(root);inputs=read(root/"inputs.json")
    row=next(r for r in inputs["rows"] if r["method"]==method)
    out=root/"outputs"/method;out.mkdir(parents=True,exist_ok=False)
    cfg=read(root/"configs"/(method+".json"));tracker=None;code=1
    try:
        require(runtime()==inputs["runtime"],"RUNTIME_VERSIONS_CHANGED")
        require(torch.cuda.is_available() and torch.cuda.device_count()==1,"ONE_ALLOCATED_GPU")
        for item in inputs["model_assets"]:check_stat(item)
        require(file_sha(Path(inputs["model_snapshot"])/"tokenizer.json")
            ==inputs["tokenizer_files_sha256"]["tokenizer.json"],"TOKENIZER_CHANGED")
        records=read(verify_member(inputs["stream"]))
        hparams=read(verify_member(row["hparams"]))
        tracker=init(env_file="/mnt/raid5/janghj/ODE-edit/servers/local/wandb.env",
                     spool=out/"tracking",config=cfg)
        require(tracker.log({"eval_progress/completed_queries":0,
                            "eval_progress/total_queries":sum(COUNTS.values())}),"TRACKING_REJECTED_START")
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        tok=tokenizer(inputs["model_snapshot"])
        model=AutoModelForCausalLM.from_pretrained(inputs["model_snapshot"],local_files_only=True,
            torch_dtype=torch.float32,attn_implementation="eager").to("cuda:0").eval()
        require(model.config.model_type=="gptj" and model.config.n_layer==28
            and model.config.n_embd==4096 and model.config.vocab_size==50400
            and all(p.dtype==torch.float32 for p in model.parameters()),"GPTJ_FP32_STRUCTURE")
        restored=load_and_restore(model,row,hparams)
        write_new(out/"restore.json",restored)
        last=[-1e30]
        def progress(value):
            now=time.monotonic()
            if now-last[0]>=15 or value["completed_queries"]==value["total_queries"]:
                require(tracker.log({"eval_progress/"+k:v for k,v in value.items()}),
                        "TRACKING_REJECTED_PROGRESS")
                last[0]=now
        identity=dict(checkpoint=row["checkpoint"],original_identity=row["identity"],
            original_job_id=row["original_job_id"],eval_source=lock["code_commit"],
            config_sha256=cfg["config_sha"],evaluator_sha256=EVALUATOR_SHA,
            stream_sha256=inputs["stream"]["sha256"],tokenizer_sha256=inputs["tokenizer_sha256"],
            actual_job_id=os.environ["SLURM_JOB_ID"])
        with torch.autocast("cuda",enabled=False):
            raw=evaluate(model,tok,records,model_family="gptj",batch_size=16,device="cuda:0",
                         identity=identity,progress=progress)
        summary=validate_result(raw,inputs)
        verify_member(row["checkpoint"])
        verify_member(row["latest"])
        write_new(out/"evaluation.json",raw)
        payload=official_zsre_metrics(raw["summary"],config_values=cfg,
            endpoint="all_seen/post",edits=2000,post_state_edits=2000)
        require(tracker.log(payload),"TRACKING_REJECTED_FINAL")
        write_new(out/"result.json",dict(status="EVAL_COMPLETE_W20_2000",method=method,
            model="gptj",dataset="zsre",job_id=os.environ["SLURM_JOB_ID"],source=lock["code_commit"],
            config_sha256=cfg["config_sha"],checkpoint=row["checkpoint"],summary=summary,
            token_denominators=COUNTS,raw=member(out/"evaluation.json"),identity=identity,
            no_edits=True,no_W0=True,no_generation=True,original_CP_unchanged=True,
            numerical_public_output_parity="NOT_SEPARATELY_MEASURED"))
        code=0
    except BaseException as error:
        write_new(out/"failure.json",dict(status="TECHNICAL_FAILURE",error_type=type(error).__name__,
            message=str(error)[:500],automatic_retry=False,original_CP_KEEP=True))
        raise
    finally:
        if tracker is not None:
            write_new(out/"transport-finish.json",tracker.finish(exit_code=code,timeout=45))

def collect(root):
    root=Path(root).resolve();lock=verify_source(root);inputs=read(root/"inputs.json")
    results=[];missing=[]
    for method in METHODS:
        try:
            out=root/"outputs"/method
            result=read(out/"result.json")
            raw=read(verify_member(result["raw"]))
            values=validate_result(raw,inputs)
            require(result["source"]==lock["code_commit"],"RESULT_SOURCE")
            results.append(dict(method=method,job_id=result["job_id"],summary=values,
                token_denominators=COUNTS,raw=result["raw"],
                transport=read(out/"transport-finish.json")))
        except Exception as error:
            missing.append(dict(method=method,error_type=type(error).__name__,message=str(error)[:240]))
    value=dict(status="EVAL_COMPLETE_6" if not missing else "PARTIAL_OR_FAILED",results=results,
        missing=missing,source=lock["code_commit"],original_CP_KEEP=True,raw_local_only=True)
    write_new(root/"collector/result.json",value)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=("prepare","run","collect"));p.add_argument("--root",type=Path,required=True)
    p.add_argument("--method",choices=METHODS);a=p.parse_args()
    if a.command=="prepare":cpu_prepare(a.root)
    elif a.command=="run":execute(a.root,a.method)
    else:collect(a.root)
