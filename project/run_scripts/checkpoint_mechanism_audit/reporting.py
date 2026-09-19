"""CPU-only raw-free publication builder for the checkpoint mechanism audit.

Reads terminal-bound analysis outputs; never polls jobs or loads model tensors.
Missing measurements are explicit status rows/figure text, never score zeroes.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .archival import CP, METRICS, read, require, save, sha

REPO=Path(__file__).resolve().parents[3]
CELLS=REPO/"plans/global/2026-09-20-server2-checkpoint-mechanism-audit-cells-v1.csv"
TERMINAL={"PASS","FAILED","BLOCKED","SKIPPED"}
HYPOTHESIS_STATUSES={"SUPPORTED","MIXED","NOT_SUPPORTED","UNRESOLVED"}
DEFAULT_CONTEXT={"scientific_promotion":False,"interpretation_exception":"User-authorized H1–H4 judgments for this instruction only; observations, explanations and unresolved dependencies remain separate.","checkpoint_saved":False,"new_edit_chains":0,"new_z_optimizations":0}


def validate_public_context(value):
    forbidden={"prompt","target_new","target_true","teacher_logits","access_token","password","credential","raw_stdout","generation_text","environment_variables"}
    if isinstance(value,dict):
        require(not(forbidden & set(value)),"raw or sensitive context field")
        for v in value.values():validate_public_context(v)
    elif isinstance(value,list):
        for v in value:validate_public_context(v)


def verify_geometry_csv(folder):
    """Independent scalar CSV ↔ original per-state JSON comparison, no tensors."""
    folder=Path(folder);frame=pd.read_csv(folder/"checkpoint_geometry.csv",float_precision="round_trip")
    require(frame.batch.tolist()==[0,*CP],"geometry state inventory")
    sources=[]
    for b in CP:
        path=folder/f"B{b:03d}-geometry.json";raw=read(path);r=frame[frame.batch==b].iloc[0]
        checks={"relative_w0_norm":raw["relative_w0_norm"],"weight_norm":raw["weight_norm"],"history_trace":raw["history"]["trace"],"interval_delta_norm":raw["interval_delta_norm"]}
        if b>1:
            checks.update({"stored_action_mean":raw["stored_action"]["mean"],"stored_action_mc_se":raw["stored_action"]["mc_se"],"interval_update_total_cross_term":raw["interval_update_total_cross_term"]})
        require(all(float(r[k])==float(v) for k,v in checks.items()),f"geometry CSV/JSON mismatch B{b}")
        require(raw["weight_sha256"]==r.weight_sha256,"geometry W identity mismatch")
        sources.append({"path":str(path),"sha256":sha(path)})
    sources.append({"path":str(folder/"checkpoint_geometry.csv"),"sha256":sha(folder/"checkpoint_geometry.csv")})
    return frame,sources


def _fmt(x):
    if x is None or (isinstance(x,float) and not np.isfinite(x)):return "NOT_MEASURED"
    if isinstance(x,(float,np.floating)):return f"{x:.6g}"
    return str(x).replace("|","\\|").replace("\n"," ")


def markdown(frame, columns=None):
    if frame.empty:return "NOT_MEASURED"
    frame=frame[columns] if columns else frame
    return "\n".join(["|"+"|".join(frame.columns)+"|","|"+"|".join("---" for _ in frame.columns)+"|"]+["|"+"|".join(_fmt(x) for x in row)+"|" for row in frame.itertuples(index=False,name=None)])


def verify_receipt(path):
    receipt=read(path)
    require(receipt.get("status")=="PASS",f"not successful completed receipt: {path}")
    for member in receipt.get("outputs",[]):
        p=Path(member["path"])
        require(p.is_file() and p.stat().st_size==member["bytes"] and sha(p)==member["sha256"],f"analysis member changed: {p}")
    return receipt


def cells_from_input(path, archival_status, geometry_status, final=False):
    rows=pd.read_csv(CELLS).fillna("")
    supplied={}
    if path:
        obj=read(path);obj=obj.get("cells",obj)
        if isinstance(obj,list):
            require(len({r["cell_id"] for r in obj})==len(obj),"duplicate cell statuses")
            supplied={r["cell_id"]:r for r in obj}
        else:supplied={key:({"status":value} if isinstance(value,str) else value) for key,value in obj.items()}
        require(not(set(supplied)-set(rows.cell_id)),"unknown cell status")
    if final:require(set(supplied)==set(rows.cell_id),"final report requires explicit complete31-cell statuses")
    result=[]
    for row in rows.to_dict("records"):
        c=row["cell_id"];given=supplied.get(c)
        default=archival_status if c=="A01" else geometry_status if c=="B00" else "NOT_MEASURED"
        status=given.get("status") if given else default
        if final:require(status in TERMINAL,f"nonterminal cell {c}: {status}")
        if c=="A01" and status=="PASS":require(archival_status=="PASS","A01 false PASS")
        if c=="B00" and status=="PASS":require(geometry_status=="PASS","B00 false PASS")
        # A report's own completion is not a scientific promotion of other cells.
        result.append({**row,**(given or {}),"cell_id":c,"status":status,"reason":(given or {}).get("reason", "terminal-bound existing CPU receipt" if status=="PASS" else "No completed evidence supplied to this report build")})
    return pd.DataFrame(result)


def functional_summary(long):
    rows=[]
    for b in CP:
        x=long[long.checkpoint_batch==b]
        for name,mask in (("all_seen",np.ones(len(x),bool)),("first100",x.arrival_batch.eq(1)),("first1000",x.arrival_batch.le(10))):
            if name=="first1000" and b<10:continue
            for tag,g in x[mask].groupby("metric_tag",sort=False):
                r={"checkpoint_batch":b,"edits":b*100,"cohort":name,"metric_tag":tag,"numerator":int(g.success.sum()),"denominator":len(g),"rate":float(g.success.mean()),"strict_numerator":int(g.target_strict.sum()),"strict_denominator":len(g),"token_correct":int(g.target_token_correct.sum()),"token_denominator":int(g.target_token_count.sum()),"requests":int(g.case_id.nunique())}
                for field in ("new_nll","true_nll","safety_margin"):
                    r[field+"_mean"]=float(g[field].mean())
                    for q,label in ((.01,"q01"),(.05,"q05"),(.5,"q50"),(.95,"q95"),(.99,"q99")):
                        r[field+"_"+label]=float(g[field].quantile(q))
                rows.append(r)
    return pd.DataFrame(rows)


def terminal_csvs(results,name):
    """Only consume CSVs bound by a terminal analysis receipt in their folder."""
    frames=[];sources=[]
    for path in sorted(results.rglob(name)):
        terminal=path.parent/"terminal.json"
        if not terminal.is_file():continue
        r=read(terminal)
        require(r.get("status") in TERMINAL|{"FAILED_TECHNICAL"},f"nonterminal result {terminal}")
        matches=[m for m in r.get("members",[]) if Path(m["path"]).name==name]
        require(len(matches)==1,f"unbound CSV {path}")
        member=matches[0]
        require(path.stat().st_size==member["bytes"] and sha(path)==member["sha256"],f"CSV hash mismatch {path}")
        try:frame=pd.read_csv(path)
        except pd.errors.EmptyDataError:continue
        if not frame.empty:
            frame["receipt_status"]=r["status"];frames.append(frame)
        sources.append({"path":str(path),"sha256":member["sha256"],"terminal":str(terminal),"terminal_sha256":sha(terminal)})
    return (pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()),sources


def prepare_optional(results,context):
    tables={};sources=[]
    for name in ("fixed_probe_history.csv","native_write_modes.csv","reconstruction.csv","counterfactuals.csv","activation_margin.csv"):
        frame,src=terminal_csvs(results,name);tables[name]=frame;sources+=src
    # Explicit small CSVs can come from a separately sealed completed reducer.
    # Parent must provide exact SHA; no search of live logs or inferred numbers.
    for name,member in context.get("completed_tables",{}).items():
        require(name in tables,"unsupported optional table")
        p=Path(member["path"]);require(sha(p)==member["sha256"],"completed table SHA mismatch")
        tables[name]=pd.read_csv(p);sources.append(member)
    return tables,sources


def summarize_optional(tables):
    out={}
    p=tables["fixed_probe_history.csv"]
    if not p.empty:
        value=next((c for c in ("normalized_score","nu","normalized_nu") if c in p),None)
        if value:
            rows=[]
            for history,g in p.groupby("history_batch"):
                r={"history_batch":int(history),"probe_count":len(g),"raw_score_mean":float(g.raw_score.mean()),"nu_mean":float(g[value].mean()),"nu_q05":float(g[value].quantile(.05)),"nu_q50":float(g[value].quantile(.5)),"nu_q95":float(g[value].quantile(.95)),"negative_raw_count":int(g.raw_score.lt(0).sum()),"nu_outside_unit_count":int((g[value].le(0)|g[value].gt(1)).sum()),"nu_finite_count":int(np.isfinite(g[value]).sum()),"projected_key_zero_count":int(g.pk_zero.sum()) if "pk_zero" in g else None,"status":"MEASURED_RAW_UNCLIPPED"}
                rows.append(r)
            out["fixed_probe_summary.csv"]=pd.DataFrame(rows)
        else:out["fixed_probe_summary.csv"]=pd.DataFrame([{"status":"SCHEMA_UNRESOLVED","reason":"normalized-score column unavailable"}])
    else:out["fixed_probe_summary.csv"]=pd.DataFrame([{"status":"NOT_MEASURED"}])
    modes=tables["native_write_modes.csv"]
    out["native_write_modes.csv"]=modes if not modes.empty else pd.DataFrame([{"status":"NOT_MEASURED"}])
    for name in ("reconstruction.csv","counterfactuals.csv","activation_margin.csv"):
        frame=tables[name]
        if name=="activation_margin.csv" and not frame.empty:
            forbidden={"prompt","target_new","target_true","generation","text","token_ids","logits"}
            require(not(forbidden & set(frame.columns)),"raw content in publication activation table")
            frame=frame.copy()
            for canonical,source in (("D_energy","DK_squared_norm"),("E_energy","EK_squared_norm"),("cross_term","EK_DK_signed_cross_term"),("remainder","nonlinear_remainder")):
                if canonical not in frame and source in frame:frame[canonical]=frame[source]
        out[name]=frame if not frame.empty else pd.DataFrame([{"status":"NOT_MEASURED"}])
    return out


def hypotheses(transitions,cells,context):
    result={}
    final=transitions[(transitions.cohort=="at_write")&(transitions.end_batch==100)]
    for h in ("H1","H2","H3","H4"):
        result[h]={"status":"UNRESOLVED","basis":"필요한 독립 model/operator/activation evidence 및 owner 판정이 아직 결속되지 않음.","provisional":True}
    directions=[]
    for view in ("original_all_case","interval_target_conflict_free"):
        x=final[(final.view==view)&final.metric_tag.isin(METRICS)].set_index("metric_tag")
        if set(x.index)==set(METRICS):
            rates=x.loss_numerator/x.loss_denominator.replace(0,np.nan)
            directions.append(bool(rates["NS"]>rates["RS"] and rates["NS"]>rates["PS"]))
    if len(directions)==2:
        result["H1"]={"status":"SUPPORTED" if all(directions) else "MIXED","basis":"원래 all-case 및 interval-target-conflict-free의 at-write→B100에서 NS 조건부 loss가 RS/PS보다 큰지 확인한 기술적 판정. 다른 stream에 대한 인과/보편 명제는 아님.","provisional":True}
    overrides=context.get("hypotheses",{})
    require(not(set(overrides)-set(result)),"unknown hypothesis")
    for h,x in overrides.items():
        require(x.get("status") in HYPOTHESIS_STATUSES and bool(x.get("basis")),"hypothesis evidence missing")
        result[h]={**x,"provisional":bool(x.get("provisional",False))}
    return result


def _blank(ax,text):
    ax.text(.5,.5,text,ha="center",va="center",transform=ax.transAxes,wrap=True)
    ax.set_xticks([]);ax.set_yticks([])


def figure(kind,tables):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.grid":True,"grid.alpha":.2,"savefig.dpi":160,"figure.dpi":100})
    if kind=="01-functional":
        fig,axes=plt.subplots(2,3,figsize=(13,7))
        fs=tables["functional_summary.csv"];tr=tables["paired_transitions.csv"]
        for j,tag in enumerate(METRICS):
            for cohort in ("all_seen","first100","first1000"):
                g=fs[(fs.metric_tag==tag)&(fs.cohort==cohort)]
                axes[0,j].plot(g.edits,100*g.rate,marker="o",label=cohort)
            axes[0,j].set(title=f"{tag}: stored endpoint preference",xlabel="committed edits",ylabel="success (%)")
            g=tr[(tr.metric_tag==tag)&(tr.cohort=="at_write")&(tr.view=="original_all_case")]
            axes[1,j].plot(g.end_batch*100,100*g.loss_numerator/g.loss_denominator.replace(0,np.nan),label="lost / at-write successes")
            axes[1,j].plot(g.end_batch*100,100*g.gain_numerator/g.gain_denominator.replace(0,np.nan),label="gained / at-write failures")
            axes[1,j].set(xlabel="committed edits",ylabel="conditional transition (%)")
            axes[0,j].legend(fontsize=7);axes[1,j].legend(fontsize=7)
        fig.suptitle("Stored archival observations; not new model parity / W0 retention")
    else:
        fig,axes=plt.subplots(2,2,figsize=(12,7));axes=axes.ravel()
        if kind=="02-geometry":
            g=tables["checkpoint_geometry.csv"];v=g[g.batch>1]
            axes[0].plot(g.batch*100,g.relative_w0_norm,marker="o");axes[0].set(title="Actual FP64-subtracted cumulative delta / W0",xlabel="committed edits")
            axes[1].plot(g.batch*100,g.history_trace,marker="o");axes[1].set(title="Stored post-write M trace (not capacity)",xlabel="committed edits")
            axes[2].errorbar(v.batch*100,v.stored_action_mean,yerr=v.stored_action_mc_halfwidth,marker="o");axes[2].set(title="Stored J estimate ± 1.96 MC SE; 256 shared probes",xlabel="interval endpoint edits")
            axes[3].axhline(0,color="gray");axes[3].plot(v.batch*100,v.interval_update_total_cross_term,marker="o");axes[3].set(title="Interval ||D||² - sum per-batch ||delta||²",xlabel="interval endpoint edits")
        elif kind=="03-operator":
            p=tables["fixed_probe_summary.csv"];m=tables["native_write_modes.csv"]
            if "nu_mean" in p:
                axes[0].plot(p.history_batch,p.nu_mean,marker="o");axes[0].fill_between(p.history_batch,p.nu_q05,p.nu_q95,alpha=.2);axes[0].set(title="Fixed probe raw nu: mean / 5–95% (not clipped)",xlabel="history batch")
            else:_blank(axes[0],"NOT_MEASURED\nFixed-probe history operator")
            if "write_energy" in m:
                for b,g in m.groupby("target_batch"):
                    axes[1].plot(g["mode"],g.gain,label=f"B{b}")
                    axes[2].plot(g["mode"],g.target_loading_squared,label=f"B{b}")
                    axes[3].plot(g["mode"],g.write_energy,label=f"B{b}")
                for ax,title in zip(axes[1:],("Actual B singular gain","Target loading squared","Gain squared × loading energy")):
                    ax.set(title=title,xlabel="thin-SVD mode");ax.legend(fontsize=6)
            else:
                for ax in axes[1:]:_blank(ax,"NOT_MEASURED\nValidated native gain / demand decomposition")
        elif kind=="04-activation":
            a=tables["activation_margin.csv"]
            needed={"actual_margin_change","predicted_margin_change","D_energy","cross_term"}
            if needed<=set(a):
                axes[0].scatter(a.predicted_margin_change,a.actual_margin_change,s=14);axes[0].set(xlabel="entry-gradient prediction",ylabel="actual margin change",title="Outcome-selected panel, not population estimate")
                axes[1].scatter(a.D_energy,a.actual_margin_change,s=14);axes[1].set(xlabel="||D K||² (all-token TF paths)",ylabel="actual margin change")
                axes[2].scatter(a.cross_term,a.actual_margin_change,s=14);axes[2].set(xlabel="2 <E K, D K>",ylabel="actual margin change")
                axes[3].scatter(a.predicted_margin_change,a.actual_margin_change-a.predicted_margin_change,s=14);axes[3].set(xlabel="linear prediction",ylabel="nonlinear remainder")
            else:
                for ax in axes:_blank(ax,"NOT_MEASURED\nAll-valid-token activation / margin evidence")
        else:raise ValueError(kind)
    fig.tight_layout();return fig


def save_figures(output,tables):
    rows=[]
    source_map={"01-functional":["functional_summary.csv","paired_transitions.csv"],"02-geometry":["checkpoint_geometry.csv"],"03-operator":["fixed_probe_summary.csv","native_write_modes.csv"],"04-activation":["activation_margin.csv"]}
    for kind,inputs in source_map.items():
        first={}
        for iteration in (0,1):
            fig=figure(kind,tables)
            for ext in ("png","pdf"):
                b=io.BytesIO();meta={"Software":"checkpoint_mechanism_audit.reporting"} if ext=="png" else {"Creator":"checkpoint_mechanism_audit.reporting","CreationDate":None,"ModDate":None}
                fig.savefig(b,format=ext,metadata=meta);content=b.getvalue();h=hashlib.sha256(content).hexdigest()
                if iteration==0:
                    first[ext]=h
                    with (output/f"{kind}.{ext}").open("xb") as f:f.write(content)
                else:require(h==first[ext],f"nondeterministic figure {kind}.{ext}")
            plt.close(fig)
        rows.append({"figure":kind,"source_csvs":inputs,"source_sha256":{x:sha(output/x) for x in inputs},"png_sha256":first["png"],"pdf_sha256":first["pdf"],"regenerated_twice_byte_exact":True,"image_generation_tools":False})
    return rows


def build(results,output,cell_statuses=None,context_path=None,final=False):
    start=time.monotonic();results=Path(results);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    context={**DEFAULT_CONTEXT,**(read(context_path) if context_path else {})}
    validate_public_context(context)
    archive=verify_receipt(results/"archival/archival-receipt.json")
    geometry=verify_receipt(results/"geometry/geometry-receipt.json")
    cells=cells_from_input(cell_statuses,archive["status"],geometry["status"],final)
    long=pd.read_parquet(results/"archival/functional_long.parquet")
    tables={"functional_summary.csv":functional_summary(long)}
    for name in ("paired_transitions.csv","paired_bootstrap.csv","at_write_outcomes.csv"):
        tables[name]=pd.read_csv(results/"archival"/name)
    tables["checkpoint_geometry.csv"],geometry_sources=verify_geometry_csv(results/"geometry")
    optional,optional_sources=prepare_optional(results,context);tables.update(summarize_optional(optional))
    tables["cell_status.csv"]=cells
    if context.get('source_bindings'):
        tables['source_bindings.csv']=pd.DataFrame(context['source_bindings'])
    if context.get('artifact_index'):
        tables['artifact-index.csv']=pd.DataFrame(context['artifact_index'])
    panel=pd.read_csv(results/"archival/mechanism_panel.csv")
    tables["mechanism_panel_summary.csv"]=panel.groupby(["interval_start","interval_end","selection_role"]).agg(rows=("identity","size"),distinct_requests=("case_id","nunique"),mean_entry_margin=("entry_margin","mean"),mean_endpoint_margin=("endpoint_margin","mean")).reset_index()
    judgements=hypotheses(tables["paired_transitions.csv"],cells,context)
    tables["hypotheses.csv"]=pd.DataFrame([{"hypothesis":h,**r} for h,r in judgements.items()])
    for name,table in tables.items():table.to_csv(output/name,index=False)
    plots=save_figures(output,tables)
    fs=tables["functional_summary.csv"];tr=tables["paired_transitions.csv"];ci=tables["paired_bootstrap.csv"]
    latest=fs[(fs.checkpoint_batch==100)&(fs.cohort=="all_seen")]
    main_transition=tr[(tr.cohort=="at_write")&(tr.end_batch==100)&(tr.view=="original_all_case")]
    overwrite_transition=tr[(tr.cohort=="at_write")&(tr.end_batch==100)&tr.metric_tag.isin(METRICS)]
    main_ci=ci[(ci.cohort=="at_write")&(ci.end_batch==100)&(ci.view=="original_all_case")]
    nll=latest[["metric_tag","new_nll_mean","new_nll_q05","new_nll_q50","new_nll_q95","true_nll_mean","safety_margin_q01","safety_margin_q05","safety_margin_q50"]]
    geom=tables["checkpoint_geometry.csv"];last=geom[geom.batch==100].iloc[0]
    command=f"python -B -m project.run_scripts.checkpoint_mechanism_audit.reporting --results {results} --output <NEW_EMPTY_OUTPUT>"+(f" --cell-statuses {cell_statuses}" if cell_statuses else "")+(f" --context {context_path}" if context_path else "")+(" --final" if final else "")
    cell_counts=cells.status.value_counts().to_dict()
    gate_section=context.get('gate_summary_ko','실제 gate 요약은 아직 제공되지 않았다.')
    cost_table=markdown(pd.DataFrame(context.get('cost_ledger',[])))
    state="최종 terminal 수집 보고" if final else "중간 CPU 근거 보고 — 전체 task 완료 아님"
    body=f"""# 회수 AlphaEdit BLUE L4-only checkpoint 기전 분석

작성 서버: Server2 / SH2. 상태: **{state}**. scientific_promotion=false.
사용자 instruction ODEEDIT-S06-S2-CHECKPOINT-MECHANISM-AUDIT-20260920-V1에 한정해 H1–H4 판정이 허용되었다. 관측, 가능한 설명, 미분리 요인을 구분한다. 아래 archive 수치는 새 GPU parity의 증거가 아니다.

## 1. 현재 완료 범위와 핵심 수치

저장 current100 및 seen12를 독립 CPU reducer로 검사했다. 중복 {archive['current_seen_duplicates']:,}행은 scalar 일치를 확인한 뒤 한 번만 셌다. 유일한 요청은 10,000개이며, dedup 관측 {archive['observations']:,}행과 at-write anchor {archive['at_write_anchors']:,}행을 독립 표본수로 오해하지 않는다. 같은 요청의 반복 관측이다.
Cell 상태: `{json.dumps(cell_counts,ensure_ascii=False)}`. Geometry는 actual W0+12개 W/M, 동일 random vector256개를 사용하는11개 구간을 검사했다. C00/C01 등 모델·writer·evaluator gate의 성공 여부는 cell 표의 실제 status만 따른다.

### Actual gate와 차단 경계

{gate_section}

### B100 전체 seen 요청의 저장 평가

{markdown(latest,['metric_tag','numerator','denominator','rate','strict_numerator','strict_denominator','token_correct','token_denominator'])}

### 요청별 실제 at-write → B100 전이

{markdown(main_transition,['metric_tag','rows','start_success','end_success','n11','n10','n01','n00','loss_denominator','gain_denominator','margin_delta_mean'])}

Loss 조건부 분모는 시작 성공, recovery 분모는 시작 실패다. `RP_JOINT`는 같은 request의 RS∧PS0∧PS1이며 strict도 세 prompt가 모두 strict일 때만 성공이다. Joint safety margin은 세 prompt safety margin의 최소값이다. At-write NS는 first-write-N이며 W0-correct-N 유지율이 아니다.

## 2. 원 실행·입력·수치 결속

원 arm은 AlphaEdit_L4_ONLY, 표시명 AlphaEdit_BLUE_L4_ONLY다. L4 down_proj FP32[4096,14336] 하나와 post-write cache_c[1,14336,14336]를 대상으로 한다. M은 static Wikipedia C0 또는 forward weight가 아니다. Base revision은 8afb486c1db24fe5011ec46dfbe5b5dccdb575c2이며 W0 저장 BF16 tensor를 FP32 cast한 identity를 사용한다. Source layers[4,5,6,7,8]의 P slot0=L4다.

원 실행 contract: Torch2.9.1+cu128, Transformers4.44.2, FP32/eager/autocast=false, TF32 matmul=false/cuDNN=true. Writer/evaluator BOS 차이, evaluator MB16·수동 left-padding·implicit position_ids를 유지한다. 이 문구는 실제 GPU 환경 PASS 선언이 아니며 원 요구사항이다. 실제 실행/환경 identity는 `context.json` 및 terminal receipt에 분리 기록한다. Migration39283_3/runtime40426 표기 차이는 숨기지 않고 W/M/model/source/order/context hash로 결속한다.

B001 archive 기대값은 RS100/100, PS190/200, NS867/1000이고 CPU 검산에서 일치했다. EN100/194/865를 사용하지 않았다. 첫 entry.context_hash는 JSON null의 digest였고 native lazy context 초기화 뒤 contexts.json/commit에는 봉인된 실제 context digest가 기록되었다. 이는 상태별 관측을 그대로 남긴 것이며 context를 다시 생성하지 않았다.

Source publication·실행 source·analysis source·report SHA는 서로 다른 identity다. Context 미제공 field는 NOT_RECORDED이며 최신 main SHA를 과거 실행 SHA로 대체하지 않는다. 새 W/M/checkpoint/복원 delta 저장은 0; 기존12CP는 읽기 전용 보존이다. 새 z fitting·editing chain·GSS·EN 실행은 없다.

## 3. 평가 정의와 종단 설계

Join은 (arm,metric_tag,identity), 관측 key에는 checkpoint batch를 추가했다. Identity는 원 evaluator의 [case_id,prompt_index,prompt,target_new,target_true] canonical JSON SHA다. Category가 digest에 없으므로 metric tag를 생략하지 않았다. 원 denominator는 request당 R1/P2/N10이다.

RS/PS safety margin=true_nll−new_nll, NS=new_nll−true_nll이며 양수만 성공이고 tie는 실패다. 각 NLL은 해당 target 자체의 teacher-forced sequence에서 target token 평균이다. Free generation이나 full-vocabulary classification accuracy와 동일하지 않다. RS/PS strict는 new target의 모든 token 정답, NS strict는 true target의 모든 token 정답이다.

각 request의 최초 anchor는 arrival batch current.json이다. 첫 sparse seen checkpoint나 entry.json을 at-write 또는 W0 평가로 바꾸지 않았다. First100은12경계, first1000은B10 이후, adjacent는 출발점에 이미 도착한 동일 rows만 비교한다. Age는 arrival batch를 고정한다. 처음 관측한 failure는 sparse interval-censored이며 관측 사이에 실패/회복이 없었다는 보장은 없다. W0 first100 새 평가가 결속되지 않은 범위는 W0→post loss로 표현하지 않는다.

## 4. Overwrite 및 전이 불확실성

원래 all-case를 유지한 census: `{json.dumps(archive['overwrite_counts'],ensure_ascii=False)}`.
같은 batch의 서로 다른 target은 BATCH_INTERNAL_CONFLICT이며 배열 마지막을 실제 최신 정답으로 확정하지 않았다. Active-version은 별도 진단이고 원 benchmark를 대체하지 않는다. 같은 latest batch에서 target이 동일한 중복 요청은 함께 남을 수 있다. 두 상충 target이 original true보다 모두 우세하더라도 두 사실의 완전 동시 저장으로 해석하지 않는다.

{markdown(overwrite_transition,['view','metric_tag','rows','start_success','end_success','loss_numerator','loss_denominator','gain_numerator','gain_denominator'])}

Request cluster bootstrap2000회, seed20260920. 같은 cohort membership에 같은 재표집을 사용하여 P/N rows와 observed times를 묶는다. Subject-relation sensitivity도 동일하게2000회다. 아래 구간은 pointwise conditional percentile95%이며 동시 band나 여러 edit 순서에 대한 보장은 아니다. Growing-cohort의 서로 다른 membership에는 각각 조건부 재표집이 적용된다. 분모0 replicate는 별도 finite count와 NA로 표시했다.

{markdown(main_ci,['metric_tag','cluster','clusters','replicates','net_rate_lo','net_rate_hi','loss_rate_lo','loss_rate_hi','gain_rate_lo','gain_rate_hi'])}

## 5. Strict, token 및 NLL tail

Strict/token numden은 첫 표와 `functional_summary.csv`에 보존한다. 다음은 B100 all-seen의 per-prompt paired-target NLL 요약이다. Prompt를 독립 request로 재계수하지 않는다. 낮은 RS 또는 safety margin만으로 전체 pretrained capability 저하를 판정하지 않는다.

{markdown(nll)}

전체12점, first100/first1000, margin q01/q05/q50/q95/q99 및 true/new NLL은 CSV에 있다. 관측하지 않은 checkpoint·PS/NS를 보간하지 않았다.

## 6. Actual W/M와 stored-history action

W 차이는 FP64 cast 뒤 subtraction했고 norm/reduction도 FP64다. B100 ||W100−W0||/||W0||={_fmt(last.relative_w0_norm)}, M trace={_fmt(last.history_trace)}. M trace 증가를 layer capacity 소진으로 대체하지 않는다.

{markdown(geom,['batch','relative_w0_norm','history_trace','interval_delta_norm','stored_action_mean','stored_action_mc_se','stored_action_relative_halfwidth','stored_action_alignment_dimensionless','interval_update_total_cross_term'])}

J=tr(D M_a Dᵀ)는 stored quadratic 작용이며 실제 forgetting 수치가 아니다. 11구간에 같은 Rademacher output vectors256개를 사용했다.64/128은 진단뿐이며 조기종료하지 않았다. J/n_a, J/tr(M_a), dJ/(||D||²tr(M_a)) 및 zero/negative flags는 원 CSV에 있다.1.96 MC SE는 근사적인 Monte Carlo 정밀도 지표이지 preservation threshold/엄밀한 유한표본 CI가 아니다. Stored M의 FP32 누적 오차는 actual historical key response와 차이를 만들 수 있다. 임의 방향 PSD checks는 PSD/rank 인증이 아니다.

구간 ||D||²−Σ||Δbatch||²는 내부 update의 총 교차항이다. 개별 pair의 상쇄나 single-batch 손상을 복원한 값이 아니다. Norm/angle만으로 output locality 손상을 확정하지 않는다.

## 7. 실제 모델·operator·demand 검증 경계

`fixed_probe_summary.csv`, `native_write_modes.csv`, `reconstruction.csv`, `counterfactuals.csv`가 없던 측정은 NOT_MEASURED status이며 0으로 채우지 않았다. Native700과 stream-ID-disjoint geometry512는 다른 panel이며 기존 reference512/G256을 교체하지 않는다. 동일 key를13history에 사용하고 각 native100은 자기 S/B를 유지한다.

일반 비대칭 H=λI+PM에는 LU/solve를 사용하며 명시 inverse/rawCG/Cholesky를 쓰지 않는다. Raw score를 ideal 범위에 clipping하지 않는다. R은 saved z와 bare h0+(Wentry−W0)k_bare로 결속하며 mean-context K를 residual에 대입하지 않는다. B1만 actual next-weight tensor와 비교 가능하고 나머지는 RECONSTRUCTED_NATIVE_WRITE다. B1/B91 dense-factor parity 실패 시 해당 attribution을 보류한다.

직접 thin SVD B의 gain²×||Rv||²는 factor write Frobenius energy를 분해한다. Actual/native 재현 잔차 및 cross term, near-degenerate band를 별도로 기록한다. 작은 K singular value가 필연적으로 증폭한다는 가정을 하지 않는다. 고정 K/R×history,20개 R permutation은 대수적 counterfactual이며 실제 editing 성과가 아니다.

## 8. Activation–margin 사후 panel

{markdown(tables['mechanism_panel_summary.csv'])}

두 구간 모두 first100 NS1000에서 lost16+matched-retained16을 선정했다. Target conflict를 제외하고 target 길이 일치, relation 일치, entry margin 차이, seed hash 순으로 중복 없이 match했다. 선택은 outcome-dependent 사후 기전 표본이며 대표 성능 추정이나 controller threshold selector가 아니다.

실제 분석은 true/new 각각 all-valid-token TF 경로, E K/D K 및 2〈EK,DK〉, s=0/.5/1의 실제 margin·strict, entry gradient 선형예측과 remainder를 구분한다. 해당 CSV가 NOT_MEASURED이면 archive panel 선택만 완료한 것이다. 세 점과 한 entry gradient는 선분 전체의 인증이 아니다. Parent R/P 변화는 별도 observer이고 사례별 token patch를 구현 가능한 공통 weight update라고 부르지 않는다.

## 9. H1–H4 판정 및 H5

{markdown(tables['hypotheses.csv'])}

관측 사실과 가능한 설명을 구분한다. H1의 차등 시간양상은 이 단일 stream의 기술적 관측이다. History growth가 그 손상의 원인인지는 고정-key operator 및 실제 activation/signed margin 연결 없이는 분리되지 않는다. Current request composition, target demand, physical rounding, overwrite 및 nonlinear suffix가 경쟁 설명이다. H5는 current edit response를 유지하며 손상을 줄일 자유도 조사 필요성이라는 후속 질문이며 새 optimizer/GSS/EN/layer arm은 NOT_RUN이다.

## 10. 비용·실행·제한과 재현

Archive CPU wall={archive['wall_seconds']:.6f}s. Geometry CPU wall={geometry.get('seconds','NOT_RECORDED')}s, original checkpoint full-hash 비용={geometry.get('hash_seconds','NOT_RECORDED')}s, peak RSS={geometry.get('peak_rss_kib','NOT_RECORDED')}KiB. 이 시간은 새 model/operator GPU 시간과 다르다. Worker 병렬 wall을 무조건 더해 end-to-end로 부르지 않는다. 실제 scheduler allocated GPU-sec/teacher-prefix/suffix F/B/solve/hash/I/O는 제공된 `context.json`의 cost ledger만 사용하며 미기록 값은 추정하지 않는다.

원 설계 RAM64GiB budget보다 Server2 admission ceiling60416MiB가 작다. 최신 task override는 프로젝트 cap2 안의1GPU×2lane, 각8CPU/exportNONE/Requeue0이다. 공통 actual gate가 선행하고 의존성이 있는 단계는 slot을 채우려 중복 실행하지 않는다. 본 CPU report builder는 scheduler 조회/model/GPU/eval을 하지 않는다.

{cost_table}

비용 ledger의 allocated GPU seconds는 scheduler allocation wall×GPU수이며 CUDA kernel busy time이 아니다. 포함된 load/hash/verification/evaluation 시간을 다시 더해 합계를 부풀리지 않는다. 미계측 세부 F/B·I/O는 NOT_RECORDED로 남기고 추정으로 분할하지 않는다. 실행한 C00/C01 capture/evaluation은 frozen parameter forward-only이며 backward0이다.

재현 명령:

```bash
{command}
```

4종 figure는 위 CSV와 본 Python code로 생성했다. PNG/PDF를 각각 두 번 생성하여 byte-exact 재현을 검사한다. Imagegen/수동수치수정은0. `figure-manifest.json`에 입력CSV·출력SHA를 기록한다. Raw prompts/weights/cache/tensor/prediction/log는 Git에 포함하지 않고 local 보존한다. NO_BROADCAST_NOT_REQUIRED. 이번 report 생성은 새 과학실험·무관 task recall·새 checkpoint 저장이 아니다.

## 11. Cell별 상태

{markdown(cells,['cell_id','stage','status','reason'])}

모든31cell의 성공을 가정하지 않는다. Final은 PASS/FAILED/BLOCKED/SKIPPED terminal 수집 후 생성하며 failed dependency만 관련 claim을 차단한다. Draft는 미실행 status를 terminal로 위조하지 않는다. 낮은 efficacy/큰slack/신호없음 때문에 관측을 제거하거나 threshold를 조정하지 않았다.
"""
    (output/"report-ko.md").write_text(body,encoding="utf-8")
    if 'numerical_parity' in context:save(output/'numerical-parity.json',context['numerical_parity'])
    if 'timing' in context:save(output/'timing.json',context['timing'])
    save(output/"context.json",context)
    save(output/"figure-manifest.json",{"figures":plots,"regeneration_command":command,"matplotlib":matplotlib.__version__})
    inputs=[{"path":str(results/"archival/archival-receipt.json"),"sha256":sha(results/"archival/archival-receipt.json")},{"path":str(results/"geometry/geometry-receipt.json"),"sha256":sha(results/"geometry/geometry-receipt.json")},{"path":str(CELLS),"sha256":sha(CELLS)}]+optional_sources+geometry_sources
    if context_path:inputs.append({"path":str(context_path),"sha256":sha(context_path)})
    if cell_statuses:inputs.append({"path":str(cell_statuses),"sha256":sha(cell_statuses)})
    members=[{"path":p.name,"bytes":p.stat().st_size,"sha256":sha(p)} for p in sorted(output.iterdir()) if p.is_file()]
    manifest={"status":"FINAL_TERMINAL_REPORT" if final else "PARTIAL_DRAFT_NOT_CAMPAIGN_COMPLETION","analysis_source":{"path":str(Path(__file__)),"sha256":sha(__file__)},"runtime":{"python":platform.python_version(),"pandas":pd.__version__,"numpy":np.__version__,"matplotlib":matplotlib.__version__},"inputs":inputs,"members":members,"report_sha256":sha(output/"report-ko.md"),"cell_counts":cell_counts,"seconds":time.monotonic()-start,"scientific_promotion":False,"new_gpu_or_evaluator_calls":0,"checkpoint_saved":False,"raw_broadcast":"NO_BROADCAST_NOT_REQUIRED"}
    save(output/"report-manifest.json",manifest)
    save(output/"rooted-receipt.json",{"manifest_sha256":sha(output/"report-manifest.json"),"report_sha256":manifest["report_sha256"],"member_root":hashlib.sha256(json.dumps(members,sort_keys=True,separators=(",",":")).encode()).hexdigest(),"status":manifest["status"],"checkpoint_saved":False})
    return manifest


def main():
    p=argparse.ArgumentParser();p.add_argument("--results",required=True);p.add_argument("--output",required=True);p.add_argument("--cell-statuses");p.add_argument("--context");p.add_argument("--final",action="store_true")
    a=p.parse_args();m=build(a.results,a.output,a.cell_statuses,a.context,a.final);print(json.dumps({k:m[k] for k in ("status","report_sha256","cell_counts","seconds")},ensure_ascii=False))


if __name__=="__main__":main()
