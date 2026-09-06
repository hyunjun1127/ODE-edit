"""Raw-free exact extraction; no model import, replay, imputation or decisions."""
import argparse,csv,hashlib,json,os,subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    digest=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()
def write(path,value):
    with open(path,'x',encoding='utf-8') as f:f.write(value)
def jwrite(path,obj):write(path,json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')
def csvwrite(path,rows):
    fields=sorted({k for r in rows for k in r})
    with open(path,'x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})


def metrics(p,meta):
    out=dict(meta,request_count=p['request_count'])
    for key,x in [('RS',p['preference']['rewrite']),('PS',p['preference']['rephrase']),('NS',p['locality']['canonical_ns'])]:
        for short,long in [('num','prompt_success_count'),('den','prompt_denominator'),('rate','prompt_success_rate'),
            ('strict_num','strict_request_success_count'),('strict_den','strict_request_denominator')]:out[f'{key}_{short}']=x[long]
    for kind,x in p['kind_summaries'].items():
        for k in ('nll_mean','nll_median','nll_p90','nll_max','all_tokens_correct_count','row_count'):out[f'{kind}_{k}']=x[k]
        rows=[r for r in p['rows'] if r['kind']==kind];by_case={}
        for r in rows:by_case.setdefault(r['case_id'],[]).append(r['all_tokens_correct'])
        out[f'{kind}_strict_teacher_forced_correct_num']=sum(all(v) for v in by_case.values())
        out[f'{kind}_strict_teacher_forced_correct_den']=len(by_case)
        out[f'{kind}_correct_token_count']=sum(r['correct_token_count'] for r in rows)
        out[f'{kind}_target_token_denominator']=sum(r['target_token_count'] for r in rows)
    return out


def bits(public,prefix):
    by={(r['case_id'],r['prompt_index'],r['kind']):r for r in public['rows']}
    out={}
    for (case,i,kind),r in by.items():
        if kind!=f'{prefix}_target_new':continue
        t=by[(case,i,f'{prefix}_target_true')]
        out[(case,i)]=dict(success=(t['nll']<r['nll']) if prefix=='locality' else (r['nll']<t['nll']),
            margin=(r['nll']-t['nll']) if prefix=='locality' else (t['nll']-r['nll']),
            input_identity=(r['input_identity_sha256'],t['input_identity_sha256']))
    return out


def collect(root,output,label,with_plots=False):
    root=Path(root);output=Path(output);output.mkdir(parents=True,mode=0o700)
    sample=read(root/'sample.lock.json');record={r['case_id']:r for r in sample['records']}
    gate=read(root/'smoke-gates.lock.json');pre={}
    tables={k:[] for k in ('run_registry','sequential_commit_checks','current_batch_metrics','seen_prefix_metrics',
        'rewrite_retention_matrix','prompt_transition_metrics','layer_allocation_nodes','layer_action_decomposition',
        'history_cost_shadows','l8_single_layer_shadows','compute_accounting','failure_registry','final_metrics')}
    inputs=set();completed=[];completed_batch_paths=[]
    def load(p):inputs.add(Path(p));return read(p)
    for alias in ('llama3-8b-inst','qwen2.5-7b-inst'):
        p=Path(gate['root'])/f'smoke-{alias}'/'W0-full.json';pre[alias]=load(p)['evaluation']
        tables['final_metrics'].append(metrics(pre[alias],dict(alias=alias,arm='PRE_EDIT_ORIGINAL_W0',scope='same1000_W0')))
    for chain in sorted(root.glob('chain-*')):
        runtime=load(chain/'runtime.lock.json');alias=runtime['alias'];arm=runtime['arm'];last={};at_write={};previous_bits={}
        terminal=load(chain/'terminal-receipt.json') if (chain/'terminal-receipt.json').exists() else None
        registry=dict(alias=alias,arm=arm,path=str(chain),status=terminal['status'] if terminal else 'INCOMPLETE',
            requested_contract=1000,entered_requests=sum(len(load(p)['case_ids']) for p in chain.glob('batch-*/entry.json')),
            completed_batches=len(list(chain.glob('batch-*/complete.json'))),slurm_job_id=runtime['slurm_job'],
            model_load_and_context_setup_seconds=runtime['model_load_seconds'],
            process_total_seconds=terminal['total_seconds'] if terminal else None,
            dedicated_one_gpu_process_hours=terminal['total_seconds']/3600 if terminal else None,
            terminal_compute=terminal['compute'] if terminal else 'NOT_YET_TERMINAL',
            W0_restored=terminal['W0_restored'] if terminal else 'NOT_YET_TERMINAL')
        tables['run_registry'].append(registry)
        if terminal and terminal['completed_batches']==10 and terminal['requested']==1000:
            if terminal['status']!='TERMINAL_VALID' or not terminal['W0_restored'] or terminal['history_appends']!=10:
                raise RuntimeError('TERMINAL_COMPLETENESS_BOUNDARY')
            completed.append(int(chain.name.split('-')[1]))
        if (chain/'failure.json').exists():
            f=load(chain/'failure.json');tables['failure_registry'].append(dict(alias=alias,arm=arm,stage=f['stage'],
                status=f['status'],identity=sha(chain/'failure.json'),completed_batches=f['completed_batches'],
                missing_requests=1000-100*f['completed_batches'],imputation=0))
        for bd in sorted(chain.glob('batch-*')):
            if not (bd/'complete.json').exists():continue
            completed_batch_paths.append(bd)
            summary=load(bd/'complete.json');writer=load(bd/'writer.json');commit=load(bd/'commit.json')
            k=summary['batch_index'];meta=dict(alias=alias,arm=arm,batch=k,W_sha256=commit['committed_weight_sha256'])
            current=writer['endpoint']['evaluation'];tables['current_batch_metrics'].append(metrics(current,meta))
            tables['sequential_commit_checks'].append(dict(meta,**{key:commit[key] for key in (
                'W_entry','M_entry','committed_M_content_sha256','history_append_count','writer_recompute_count',
                'fixed_z_recompute_count','model_forward_count','evaluator_count','fixed_z_sha256')}))
            tables['compute_accounting'].append(dict(meta,compute_z=writer['compute_z'],
                write_including_endpoint=writer['write_including_endpoint'],seen=summary['seen_compute'],
                checkpoint=summary['checkpoint_compute'],full_batch=summary['full_batch_compute'],
                endpoint_evaluation_seconds=writer['endpoint_evaluation_seconds'],
                history=writer.get('history_finalization_compute','NOT_RECORDED'),
                source_endpoint_physical_write_count=writer['endpoint']['physical_write_count'],
                inner_virtual_euler_nodes=len(writer.get('nodes',[])),main_jvp_count=writer.get('main_jvp_count',0),
                persistent_commit_count=1,commit_compute=commit['compute'],
                peak_gpu_bytes=summary['peak_gpu_allocated'],peak_host_rss_kib=summary['peak_host_rss_kib']))
            # Official and JV use the same actual committed endpoint observer.
            # This is net endpoint action, not a sum of Euler velocity actions.
            for x in writer['actual_physical_action']['layers']:
                tables['layer_action_decomposition'].append(dict(meta,**x,kind='actual_endpoint_net_FP32'))
            if (bd/'seen-full.json').exists():
                full=load(bd/'seen-full.json');tables['seen_prefix_metrics'].append(metrics(full,dict(meta,scope='single_W_seen_prefix')))
                w0=bits(pre[alias],'locality');observed=bits(full,'locality')
                for key,b in observed.items():
                    a=w0[key]
                    if a['input_identity']!=b['input_identity']:raise RuntimeError('W0_NEIGHBORHOOD_PAIR_IDENTITY')
                    tables['prompt_transition_metrics'].append(dict(meta,case_id=key[0],prompt_index=key[1],
                        W0_success=a['success'],endpoint_success=b['success'],new_failure=a['success'] and not b['success'],
                        recovery=not a['success'] and b['success']))
                if k==10:tables['final_metrics'].append(metrics(full,dict(meta,scope='single_final_W10_full1000')))
            for r in load(bd/'rewrite-retention.json'):
                case=r['case_id'];cohort=record[case]['batch_index'];now=r['success']
                if cohort==k:at_write[case]=now
                overwrite=any(x['subject_relation_group']==record[case]['subject_relation_group'] and
                    cohort<x['batch_index']<=k and x['target_new_sha256']!=record[case]['target_new_sha256'] for x in sample['records'])
                tables['rewrite_retention_matrix'].append(dict(meta,**r,cohort=cohort,age=k-cohort,
                    at_write_success=at_write[case],at_write_success_now_failure=at_write[case] and not now,
                    previous_failure_now_recovery=case in previous_bits and not previous_bits[case] and now,
                    overwrite_candidate=overwrite))
                previous_bits[case]=now
            for n in writer.get('nodes',[]):
                m=dict(meta,node=n['node'],state_version=n['state_version'])
                tables['layer_allocation_nodes'].append(dict(m,**{key:n[key] for key in (
                    'g','full_H','G','c','raw_physical_coefficients','q_layers','qN_ref','V_before','V_after','V_ratio',
                    'model_error_normalized','model_error_raw_activation','finite_step_dissipation_defect',
                    'predicted_target_contribution','response_residual_cosine','KKT_stationarity')}))
                for x in n['layer_actions']:tables['layer_action_decomposition'].append(dict(m,**x,kind='velocity_action'))
                for x in n['actual_physical']:tables['layer_action_decomposition'].append(dict(m,**x,kind='actual_FP32_DeltaW'))
                if n['shadows']:
                    for x in n['shadows']['single_layer']:tables['l8_single_layer_shadows'].append(dict(m,**x))
                    if 'history_cost' in n['shadows']:tables['history_cost_shadows'].append(dict(m,**n['shadows']['history_cost']))
            last=summary
    for name,rows in tables.items():csvwrite(output/f'{name}.csv',rows)
    stamp=datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
    text=['# AlphaEdit JV sequential routing — factual report',f'\n작성: {stamp} / 단계: {label}',
        '\nRS/PS/NS는 pinned NLL preference이고 tie는 실패다. 자유 생성 accuracy가 아니다. 아래 final 값은 동일한 W10의 전체 1,000 requests이며 online-own-batch pooling과 다르다.',
        '\n| Model | Arm / state | RS | PS | strict PS | NS |', '|---|---|---:|---:|---:|---:|']
    for x in tables['final_metrics']:
        text.append(f"| {x['alias']} | {x['arm']} / {x['scope']} | {x['RS_num']}/{x['RS_den']} | {x['PS_num']}/{x['PS_den']} | {x['PS_strict_num']}/{x['PS_strict_den']} | {x['NS_num']}/{x['NS_den']} |")
    text+=['\n## Current-B100와 seen-prefix (완료 receipt만)',
        '\n| Model | Arm | Batch/state scope | RS | PS | strict PS | NS |',
        '|---|---|---|---:|---:|---:|---:|']
    for scope,values in [('current B100',tables['current_batch_metrics']),('same Wk seen-prefix',tables['seen_prefix_metrics'])]:
        for x in values:
            text.append(f"| {x['alias']} | {x['arm']} | B{x['batch']} {scope} | {x['RS_num']}/{x['RS_den']} | {x['PS_num']}/{x['PS_den']} | {x['PS_strict_num']}/{x['PS_strict_den']} | {x['NS_num']}/{x['NS_den']} |")
    text+=['\n## 완전성·해석 경계',f'\n완료 chain: {sorted(completed)} / 계약: 6 chains × 1,000 requests. 미완료는 위 final 분모로 채우지 않는다; imputation0. Scientific promotion0.',
        '\n첫 B100, B5 및 final seen-prefix는 각 표에 분리했다. 실패·overwrite 후보는 canonical denominator에 유지하고 조건부 forgetting을 별도 행으로 기록한다. 평가하지 않은 intermediate PS/NS는 NOT_RECORDED다.',
        '\n## 상태와 계측',
        '\nsequential_commit_checks는 실제 W/M commit→entry identity 및 append1/recompute0을 결속한다. checkpoint1/5/10은 실제 selected weight/dense M tensor를 저장하고 다시 읽어 hash를 검증했다. Low-rank journal replay parity는 NOT_TESTED이며 hash만으로 복원성을 주장하지 않는다.',
        '\nlayer_allocation_nodes와 layer_action_decomposition은 g/full H/G, raw/normalized work, 실제 FP32 DeltaW를 구분한다. L8 share 감소 자체는 redistribution 성공이 아니다. 다른 layer의 절대 write/기여와 RS/PS를 함께 보아야 한다. 초기 history-cost shadow는 actual basis/response/N0/qref를 유지하며 M0+L2 Gram을 실제 계산한 observer다.',
        '\n## 비용',
        '\ncompute_accounting의 writer 시간은 endpoint 평가 포함값과 endpoint 평가값을 함께 제공한다. Keys/solves/JVP/shadow/forward/materialization은 node와 writer receipt에 분리했다. History bracket은 post-key부터 snapshot/restore까지로, 순수 append-only 시간과 동일시하지 않는다. 첫 B100 비용의 10배는 예상치이지 실제 전체 시간은 아니다.',
        '\n## 독립 검토',
        '\nA/B/C 및 Cases A..H 판정은 current-B100, all-seen retention, 절대 layer action, same-state history-cost 및 실제 L8-only trajectory를 함께 비교한다. Main 네 chain 보고를 L8-only 완료까지 미루지 않는다. 1,000 edits 이후 generalization, global causal claim 또는 learned history preservation 보장은 이번 범위 밖이다.',
        f'\n원본 root: `{root}`',f"\nSource: `{read(root/'source.lock.json')['head']}`; sample root: `{sample['ordered_root']}`."]
    context_root=root.parent/'checkpoint-context-support-20260906-v1'
    context_receipts=[]
    if context_root.exists():
        for p in sorted(context_root.glob('chain-*/context-recovery-receipt.json')):
            r=load(p);raw_path=Path(r['raw_local_path'])
            if sha(raw_path)!=r['file_sha256'] or raw_path.stat().st_size!=r['bytes']:
                raise RuntimeError('PRIVATE_CONTEXT_CACHE_RECOVERY_IDENTITY')
            context_receipts.append(dict(chain=p.parent.name,**r))
    csvwrite(output/'checkpoint_context_support.csv',context_receipts)
    text+=['\n## Checkpoint 복원 보조 state',
        '\ncheckpoint_context_support.csv는 이미 생성된 native context cache를 해당 chain runtime hash 및 stdout의 정확한 byte 구간과 결속한다. 문자열은 별도 private local JSON에 보존하며 Git에 싣지 않는다. Selected weights/M checkpoint 자체는 수정하지 않았고 추가 generation/model replay는 0이다. 복원 시 pinned pretrained snapshot/source/hparams/P와 해당 selected-weight/M checkpoint 및 context-cache 보조 파일을 함께 사용한다. 추가 trajectory replay parity는 NOT_TESTED다.']
    from .synthesis import summarize
    text+=summarize(root,output,completed,completed_batch_paths=completed_batch_paths,load=load)
    if with_plots:
        from .plots import plot
        for p in plot(output):text.append(f'\n![trajectory]({Path(p).name})')
    write(output/'factual-report-ko.md','\n'.join(text)+'\n')
    for name in ('source.lock.json','science.lock.json','sample.lock.json','resource.lock.json','smoke-gates.lock.json'):
        inputs.add(root/name)
    code=Path(__file__).resolve().parent
    analysis_identity=dict(head=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip(),
        members=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(code.glob('*.py'))])
    manifest=dict(label=label,source=read(root/'source.lock.json'),analysis_implementation=analysis_identity,sample_root=sample['ordered_root'],
        inputs=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(inputs)],
        members=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(output.iterdir())],
        complete_chains=sorted(completed),model_replay_count=0,imputation_count=0)
    jwrite(output/'analysis-manifest.json',manifest)
    root_hash=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    jwrite(output/'rooted-analysis-receipt.json',dict(root_sha256=root_hash,manifest_sha256=sha(output/'analysis-manifest.json'),
        report_sha256=sha(output/'factual-report-ko.md'),completed_chains=sorted(completed),status='FACTUAL_EXTRACTION_COMPLETE'))
    return sorted(completed)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--label',required=True);p.add_argument('--plots',action='store_true')
    a=p.parse_args();print(collect(a.root,a.output,a.label,with_plots=a.plots))
