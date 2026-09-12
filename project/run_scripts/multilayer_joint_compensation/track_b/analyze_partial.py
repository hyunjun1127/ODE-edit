"""CPU-only terminal rehash, reconstruction and independent prompt reduction."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import torch
from ..contracts import sha,digest,save,tensor_sha,DATA
from .runtime import verify_bundle

RUN=Path('/mnt/raid5/janghj/ODE-edit/local/multilayer-joint-compensation/20260911-v1/track_b/Middle-B-OS-tech-r1')
COMMON=Path('/mnt/raid5/janghj/ODE-edit/local/multilayer-joint-compensation/20260911-v1/imports/common-r1-Middle-v1')
PRIOR=Path('/mnt/raid5/janghj/ODE-edit/local/blue-l4-progress-barrier/attempt-v1')
PRIOR_NATIVE=PRIOR/'imports/A/Middle/native-r1'
SOURCE='3c0fd4f18189dac634db76f58b0e5e3b07f1ff8c'

def verify_execution_source(execution,repository):
    """Verify immutable Git blobs, not a subsequently edited worktree."""
    head=execution['source_head']
    tree=subprocess.check_output(['git','rev-parse',head+'^{tree}'],cwd=repository,text=True).strip()
    if tree!=execution['source_tree']:raise ValueError('FROZEN_EXECUTION_TREE')
    root=Path(execution['worktree'])
    members=[]
    for member in execution['source_members']:
        relative=str(Path(member['path']).relative_to(root))
        data=subprocess.check_output(['git','show',head+':'+relative],cwd=repository)
        actual=hashlib.sha256(data).hexdigest()
        if actual!=member['sha256']:raise ValueError('FROZEN_EXECUTION_BLOB:'+relative)
        members.append(dict(path=relative,bytes=len(data),sha256=actual))
    return dict(head=head,tree=tree,count=len(members),members_root=digest(members),mechanism='IMMUTABLE_GIT_BLOBS')

def table(path,rows):
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def validate_rows(data,panel,records):
    expected=[]
    for name,indices in panel['panels'].items():
        for metric in ('RS','PS','NS'):
            for index in indices:
                r=records[index];rw=r['requested_rewrite']
                prompts=([rw['prompt'].format(rw['subject'])] if metric=='RS' else r['paraphrase_prompts'] if metric=='PS' else r['neighborhood_prompts'])
                expected += [(name,metric,int(r['case_id']),j,digest([int(r['case_id']),j,p,rw['target_new']['str'],rw['target_true']['str']])) for j,p in enumerate(prompts)]
    rows=data['rows'];actual=[(r['panel'],r['metric'],r['case_id'],r['prompt_index'],r['identity']) for r in rows]
    if actual!=expected or len(set(actual))!=len(actual) or len(rows)!=data['pairs'] or data['panel_identity']!=digest(panel):raise ValueError('EVALUATION_ORDER_DENOMINATOR_IDENTITY')
    for r in rows:
        a,b=r['new_nll'],r['true_nll']
        if not math.isfinite(a) or not math.isfinite(b):raise ValueError('NONFINITE_PAIR')
        if r['success']!=(b<a if r['metric']=='NS' else a<b) or r['margin']!=b-a:raise ValueError('PAIR_REDUCER_MISMATCH')
        for key in ('new','true'):
            if not 0<=r[key+'_token_correct']<=r[key+'_token_count'] or r[key+'_token_count']<=0:raise ValueError('TOKEN_DENOMINATOR')
    return rows

def reduce_rows(state,rows):
    groups=defaultdict(list)
    for r in rows:groups[r['panel'],r['metric']].append(r)
    result=[]
    for (panel,metric),rr in groups.items():
        num=sum(r['true_nll']<r['new_nll'] if metric=='NS' else r['new_nll']<r['true_nll'] for r in rr)
        target='true' if metric=='NS' else 'new'
        result.append(dict(state=state,panel=panel,metric=metric,numerator=num,denominator=len(rr),percent=100*num/len(rr),
            mean_new_nll=statistics.fmean(r['new_nll'] for r in rr),mean_true_nll=statistics.fmean(r['true_nll'] for r in rr),
            TF_strict_numerator=sum(r[target+'_strict'] for r in rr),TF_strict_denominator=len(rr),
            token_numerator=sum(r[target+'_token_correct'] for r in rr),token_denominator=sum(r[target+'_token_count'] for r in rr),
            ties=sum(r['new_nll']==r['true_nll'] for r in rr)))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output).absolute();out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(8)
    raw=RUN/'output';terminal=json.loads((raw/'terminal.json').read_text());node=json.loads((raw/'nodes/node-00.json').read_text())
    if terminal['source_head']!=SOURCE or terminal['status']!='B_ENDPOINT_FINITE_RECORDED' or not terminal['W0_selected_restored']:raise ValueError('TERMINAL_BINDING')
    execution=json.loads((RUN/'execution.lock.json').read_text())
    if execution['source_head']!=SOURCE:raise ValueError('FROZEN_SOURCE')
    repo=Path(__file__).resolve().parents[4]
    frozen=verify_execution_source(execution,repo)
    analysis_source=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=repo,text=True).strip(),
        analyzer_path=str(Path(__file__).relative_to(repo)),analyzer_sha256=sha(__file__),
        execution_source_unchanged=frozen,
        authority='ODEEDIT-GH-SH1-SH2-MULTILAYER-AB-USER-RECALL-20260912-R1')
    common,receiver=verify_bundle(COMMON)
    members=[]
    for path in sorted(raw.rglob('*')):
        if path.is_file():
            if path.is_symlink():raise ValueError('RAW_SYMLINK')
            before=path.stat();value=sha(path);after=path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('RAW_CHANGED')
            members.append(dict(path=str(path),bytes=before.st_size,sha256=value))
    entry=torch.load(common/'entry.pt',map_location='cpu',weights_only=True,mmap=True)
    endpoint=torch.load(raw/'endpoint.pt',map_location='cpu',weights_only=True,mmap=True)
    nodes=torch.load(raw/'nodes/node-00.pt',map_location='cpu',weights_only=True,mmap=True)
    names=entry['names'];weights=endpoint['weights']
    for name,w in zip(names,weights):
        if w.dtype!=torch.float32 or tuple(w.shape)!=(4096,14336) or not torch.isfinite(w).all() or tensor_sha(w)!=terminal['endpoint_sha'][name]:raise ValueError('ENDPOINT_WEIGHT')
    if not torch.equal(weights[0],entry['WN'][0]) or not torch.equal(weights[1],nodes['weights'][0]):raise ValueError('FIXED_L4_OR_NODE_ENDPOINT')
    if not torch.equal(weights[1]-entry['WN'][1],nodes['actual_delta'][0]):raise ValueError('ACTUAL_DELTA')
    hist=json.loads((raw/'history-finalization/complete.json').read_text())
    if hist['terminal_batch_finalizations']!=1 or hist['terminal_layer_appends']!=2 or hist['inner_history_appends']!=0:raise ValueError('APPEND_COUNTS')
    history=[]
    for member in hist['members']:
        layer=member['layer'];keys=torch.load(raw/'history-finalization'/member['key']['name'],map_location='cpu',weights_only=True)
        if tensor_sha(keys)!=member['key']['tensor_sha256']:raise ValueError('POST_KEY_TENSOR')
        reconstructed=entry[f'M{layer}']+keys@keys.T
        saved=endpoint['histories'][layer]
        if not torch.equal(reconstructed,saved) or tensor_sha(saved)!=member['final_history_sha256']:raise ValueError('HISTORY_RECONSTRUCTION')
        history.append(dict(layer=layer,post_key_shape=list(keys.shape),M_endpoint_sha=member['final_history_sha256'],replay='FP32_EXACT_ONE_APPEND'))
    from scripts.fixed_counterfact import load_prefix
    records=load_prefix(DATA,10000)
    panel=entry['panel'];evaluated=validate_rows(json.loads((raw/'endpoint-full.json').read_text()),panel,records)
    metrics=reduce_rows('B-OS',evaluated);all_rows={'B-OS':evaluated}
    prior_manifest=json.loads((PRIOR/'transfer-source-manifest.json').read_text())
    prior_members=prior_manifest['members']
    def prior_verify(path):
        rel=str(path.relative_to(PRIOR/'imports'));m=next(m for m in prior_members if m['relative']==rel)
        if sha(path)!=m['sha256'] or path.stat().st_size!=m['bytes']:raise ValueError('PRIOR_REFERENCE_SHA')
        members.append(dict(path=str(path),bytes=m['bytes'],sha256=m['sha256']))
    prior_verify(PRIOR_NATIVE/'prepared-receipt.json')
    prep=json.loads((PRIOR_NATIVE/'prepared-receipt.json').read_text())
    if prep['WN_sha']!=tensor_sha(entry['WN'][0]) or prep['We_sha']!=tensor_sha(entry['We'][0]):raise ValueError('N4_REFERENCE_STATE')
    for state,name in [('W0','W0-full.json'),('We','ENTRY-full.json'),('N4','N-full.json')]:
        path=PRIOR_NATIVE/name;prior_verify(path)
        rows=validate_rows(json.loads(path.read_text()),panel,records);all_rows[state]=rows;metrics+=reduce_rows(state,rows)
    table(out/'endpoint-metrics.csv',metrics)
    transitions=[]
    for baseline in ('N4','We'):
        groups=defaultdict(list)
        for old,new in zip(all_rows[baseline],evaluated,strict=True):
            if old['identity']!=new['identity']:raise ValueError('PAIR_JOIN_IDENTITY')
            groups[new['panel'],new['metric']].append((old,new))
        for (pn,mt),rr in groups.items():
            transitions.append(dict(reference=baseline,endpoint='B-OS',panel=pn,metric=mt,denominator=len(rr),
                reference_success=sum(x['success'] for x,y in rr),success_to_failure=sum(x['success'] and not y['success'] for x,y in rr),
                failure_to_success=sum(not x['success'] and y['success'] for x,y in rr),
                NLL_new_delta_mean=statistics.fmean(y['new_nll']-x['new_nll'] for x,y in rr)))
    table(out/'paired-transitions.csv',transitions)
    solver=[dict(rhs=k,**{f:v for f,v in row.items() if f!='recursive_residuals'}) for k,row in node['solver']['pcg'].items()]
    table(out/'pcg.csv',solver)
    risks=[dict(panel=role,quantity='risk' if j<2 else 'train_mean_NLL',before=node['before'][j]['value' if j<2 else 'mean_nll'],
        after=node['after'][j]['value' if j<2 else 'mean_nll']) for j,role in enumerate(('Base128','Past128','Current100_600contexts'))]
    table(out/'controller-risk.csv',risks)
    counts={k:sum(v[k] for v in node['counts'].values()) for k in next(iter(node['counts'].values()))}
    summary=dict(status='TERMINAL_FINITE_APPROXIMATE_PARTIAL',job=45029,source_head=SOURCE,
        source_tree=execution['source_tree'],analysis_source=analysis_source,common_ready_sha=receiver['source_ready_sha256'],common_root=receiver['source_members_root'],
        endpoint=terminal['endpoint_sha'],history=history,current_effective=len(entry['raw_effective_inventory']['current_effective']),
        exact_prompt_pairs=len(evaluated),input_panel_sha=digest(panel),metrics=metrics,risks=risks,solver=solver,
        equality_residual=node['solver']['equality_residual'],stationarity_norm=node['solver']['stationarity_norm'],
        predicted_current_change=node['predicted_current_change'],actual_current_change=node['actual_current_change'],
        actual_delta_norm=node['actual_delta_norm'],actual_native_action=node['actual_native_action'],
        cost=dict(scheduler_gpu_seconds=14284,prior_failed_gpu_seconds=98,total_allocated_gpu_seconds=14382,
            measured_runtime_wall=terminal['total_wall_seconds'],edit_core_wall=node['wall_seconds'],
            evaluator_wall=terminal['compute']['seconds']['full_endpoint_evaluation'],history_wall=terminal['compute']['seconds']['native_key_capture'],
            peak_GPU_bytes=terminal['peak_GPU_allocated'],max_RSS_KiB=32779000,controller_counts=counts,
            runtime_ledger=terminal['compute'],old_44991_cost_included=True),
        missing=['independent BaseAudit/PastAudit','full signed L4/L8 intervention attribution','B1/B7/B100 and microbatch1/2/4 engineering comparison','remaining7B endpoints','same-entry native AlphaEdit/MEMIT6','B-BF4shortchain'],
        cross_server_reference='exact WN/We selected bytes and prompt identity; reused S1 observations, full cross-hardware numeric equivalence NOT_CLAIMED',
        result_selection=0,new_GPU_or_evaluator=0,scientific_promotion=False)
    save(out/'partial-summary.json',summary)
    lines=['# B 손상 보정 — Middle B-OS 사용자 recall 부분 보고',
        '', '**부분 결과**: job45029 COMPLETED0:0. Finite endpoint/가중치·history 복원 검산 PASS. 두 PCG RHS 모두 20회 미수렴한 근사해이며 정확한 quadratic 최적해 또는 campaign 완료가 아니다.',
        '', '## 1. 같은 Middle entry의 canonical 평가', '', '|State|Panel|Metric|n/d|%|', '|---|---|---|---:|---:|']
    for state in ('W0','We','N4','B-OS'):
        for r in metrics:
            if r['state']==state:lines.append(f"|{state}|{r['panel']}|{r['metric']}|{r['numerator']}/{r['denominator']}|{r['percent']:.3f}|")
    lines += ['', 'RS/PS=new NLL<true NLL, NS=true NLL<new NLL, tie failure. TF strict/token과 두 target NLL은 CSV에 별도 기록했다. Current/Fixed/Past는 각100 requests의 100/200/1000 prompts이며 총3900 pairs다. Fixed100/Past100은 새 BaseAudit/PastAudit bank와 같지 않다.',
        '', 'W0/We/N4는 같은 panel·순서·선택 weight bytes를 확인한 기존 S1 관측 재사용이다. 새 GPU forward0. Full cross-hardware numerical parity는 주장하지 않는다.',
        '', '## 2. Controller 관측 및 solver', '']
    for r in risks:lines.append(f"- {r['panel']} {r['quantity']}: {r['before']:.12g} → {r['after']:.12g}.")
    for r in solver:lines.append(f"- PCG {r['rhs']}: {r['iterations']} iterations, actual relative residual {r['relative_residual']:.12g}, {r['status']}.")
    lines += [f"- 일차 Current equality residual {summary['equality_residual']:.6g}, 실제 train-NLL 변화 {summary['actual_current_change']:.6g}. 일차 equality가 finite-step Current 보호를 보장하지 않았다.",
        f"- Stationarity norm {summary['stationarity_norm']:.6g}; 실제 correction norm {summary['actual_delta_norm']:.6g}, normalized native action {summary['actual_native_action']:.6g}.",
        '', '관측: 이번 B-OS에서는 Base/Past 위험이 모두 증가했고 Current 평균 NLL도 증가했다. 가능한 설명에는 큰 finite correction과 PCG 근사 오차가 함께 포함된다. 어느 요인이 주원인인지 이번 한 endpoint만으로 분리하지 못한다. 평균 Past NLL 감소와 양의 손상-tail risk 증가는 서로 다른 지표라 동시에 가능하다. 이 결과를 hard-Alpha capacity 불가능성이나 lifelong 열세로 확대하지 않는다.',
        '', '## 3. 실제 비용과 잔여 범위',
        f"- Scheduler {14284/3600:.6f} GPUh, 실패44991 98GPU-sec 포함 총 {14382/3600:.6f} GPUh. Cached common 준비비용과 native-z cold end-to-end 비용은 별도이며 여기에 포함하지 않았다.",
        f"- Edit core {node['wall_seconds']:.3f}s; terminal evaluation {summary['cost']['evaluator_wall']:.3f}s; history {summary['cost']['history_wall']:.3f}s. Peak GPU {terminal['peak_GPU_allocated']}B; MaxRSS32779000KiB.",
        f"- Controller ledger {json.dumps(counts,sort_keys=True)}. Host transfer/runtime 전체 ledger는 partial-summary.json에 보존했다.",
        '- L4는 exact WN4 고정; L8 node/endpoint equality, post-key FP32 outer-product append 각1회, 전체selected W0 restore 검산. GPU continuation replay는 수행하지 않았다.',
        '- A0/A-OS 상대 canonical 결과 수신 대기. BLUE/native6/기타 B7/shortchain은 아직 이 보고서에 측정값 없음. 성능 때문에 제외한 것이 아니라 실행·관측 미완료다.',
        '- BaseAudit/PastAudit와 L4-only/L8-only/joint signed interventions는 첫 runner에 누락됐다. 이를 측정한 것처럼 대입하지 않으며 후속 승인 범위에서 누락 관측만 보충해야 한다.',
        '- 후속 Middle B-BF4의 보수적 wall 추정은 OS의 43 matvec 대비 최대85×4이므로 약30.8h + 관측/준비. 실측 보장이나 GPU-hour cap이 아니며 max20/relres1e-4/동일bank를 변경하지 않는다.',
        '', '## 4. Provenance와 재현',f'- Frozen execution `{SOURCE}` / tree `{execution["source_tree"]}`.',
        f'- Common READY `{receiver["source_ready_sha256"]}`.',f'- Local raw `{raw}` (Git 제외, 원본 보존).',
        f'- Analysis source `{analysis_source["head"]}`; analyzer SHA `{analysis_source["analyzer_sha256"]}`. Frozen execution과 analysis source를 구분한다.',
        '- CPU 재현: `python -m project.run_scripts.multilayer_joint_compensation.track_b.analyze_partial --output <새 create-once 경로>`; frozen execution의 Git blobs와 실행 manifest를 대조하므로 이후 전용 branch 변경으로 원 실행을 덮어쓰지 않는다.',
        '- INITIAL_VALID는 FD/GGN/적용 gate일 뿐 terminal 성능 증거가 아니다. 44991 기존 세 FD값과 실패는 보존했다. scientific_promotion=false.',
        '- 새 후속 job은 최소 initial gate 뒤 다시 MONITORING_PAUSED_AWAITING_USER; 자동 terminal cascade 없음.', '']
    (out/'diagnostic-report-ko.md').write_text('\n'.join(lines))
    save(out/'input-manifest.json',dict(members=members,root=digest(members),common_receiver_sha=sha(COMMON/'receiver-ready.json'),source_execution_lock_sha=sha(RUN/'execution.lock.json'),raw_broadcast='NO_BROADCAST_NOT_REQUIRED_LOCAL_PROJECT_RETENTION'))
    outputs=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()]
    save(out/'analysis-manifest.json',dict(members=outputs,root=digest(outputs)))
    save(out/'rooted-receipt.json',dict(status=summary['status'],manifest_sha=sha(out/'analysis-manifest.json'),root=digest(outputs),CPU_only=True,scientific_promotion=False))
    print(json.dumps(dict(report=str(out/'diagnostic-report-ko.md'),sha256=sha(out/'diagnostic-report-ko.md'),status=summary['status'])),flush=True)

if __name__=='__main__':main()
