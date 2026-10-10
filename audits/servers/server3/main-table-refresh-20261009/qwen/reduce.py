"""CPU-only independent stored W20 reduction; no model imports or mutation."""
import json, pathlib, hashlib, math, collections, datetime, csv
ROOT=pathlib.Path('/data/janghj/ODE-edit/local/qwen-price-q3-eot-2k-20261009/execution-r1')
OUT=pathlib.Path('audits/servers/server3/main-table-refresh-20261009/qwen')
REPORT=pathlib.Path('experiment-reports/servers/server3/main-table-refresh-20261009/qwen')
inventory=[]
def read(p):
 p=pathlib.Path(p);b=p.read_bytes();s=p.stat();inventory.append(dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),mtime_ns=s.st_mtime_ns));return json.loads(b)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
lock=read(ROOT/'execution.lock.json');result=read(ROOT/'final/result.json');initial=read(ROOT/'final/initial.json');terminal=read(ROOT/'final/terminal.json');profile=read(ROOT/'final/resolved-profile.json')
prep=read(lock['preparation']['path']);assert sha(lock['preparation']['path'])==lock['preparation']['sha256']
stream=pathlib.Path('/data/janghj/ODE-edit/local/official-baselines-20261008/inputs/cf-stream.json');records=read(stream)
if isinstance(records,dict):records=records['records']
ids=[x['case_id'] for x in records];assert len(ids)==len(set(ids))==2000
identity=read(prep['observer_identity']['path'])['rows'];expected={r['identity']:r for r in identity if r['case_id'] in set(ids)}
rows=[]
for p in sorted((ROOT/'final/batch-20/post').glob('chunk-*.json')):
 d=read(p);assert d['state']==result['state'];rows.extend(d['rows'])
assert len(rows)==26000 and len({r['identity'] for r in rows})==26000
assert list(dict.fromkeys(r['case_id'] for r in rows))==ids
for r in rows:
 assert r['endpoint']=='W20'
 for k,v in expected[r['identity']].items():assert r[k]==v,(k,r['identity'])
 for target in ['true','new']:
  assert math.isfinite(r[target+'_nll']);n=r[target+'_token_count'];c=r[target+'_token_correct'];assert type(n)==type(c)==int and 0<=c<=n and n>0
  assert r[target+'_strict']==(c==n)
 assert math.isclose(r['margin_true_minus_new'],r['true_nll']-r['new_nll'],abs_tol=1e-12)
metrics={}
for k,count in [('R',1),('P',2),('N',10)]:
 group=[r for r in rows if r['kind']==k];by=collections.defaultdict(list);desired='true' if k=='N' else 'new'
 for r in group:by[r['case_id']].append(r)
 assert len(by)==2000 and all(len(v)==count for v in by.values())
 success=lambda r: r['true_nll']<r['new_nll'] if k=='N' else r['new_nll']<r['true_nll']
 successes=sum(map(success,group));n=len(group);tokens=sum(r[desired+'_token_count'] for r in group);correct=sum(r[desired+'_token_correct'] for r in group)
 m=dict(denominator=n,numerator=successes,rate=successes/n,true_nll_mean=sum(r['true_nll'] for r in group)/n,new_nll_mean=sum(r['new_nll'] for r in group)/n,desired_token_count=tokens,desired_token_correct=correct,token_micro=correct/tokens,prompt_macro=sum(r[desired+'_token_correct']/r[desired+'_token_count'] for r in group)/n,strict_numerator=sum(r[desired+'_strict'] for r in group),strict_denominator=n,new_strict_numerator=sum(r['new_strict'] for r in group))
 for field,v in m.items():assert math.isclose(v,result['final']['summary'][k][field],abs_tol=1e-12),field
 m.update(success_pct=successes/n*100,request_macro_pct=100*sum(sum(map(success,v))/len(v) for v in by.values())/2000,ties=sum(r['new_nll']==r['true_nll'] for r in group));metrics[k]=m
previous=initial['cold'];rng=initial['RNG'];appends=0
for i in range(1,21):
 c=read(ROOT/f'final/batch-{i:02d}/commit.json');assert c['batch']==i and c['before']==previous and c['RNG_before']==rng and c['RNG_after']==rng and c['source']==result['source'] and c['config_sha']==result['config_sha'];assert c['history_appends']==5
 assert c['post']['no_mutation'] and c['post']['state']==c['after'];previous=c['after'];appends+=5
assert previous==result['state'] and result['batches']==20 and result['status']==terminal['status']=='COMPLETE'
assert result['source']==initial['source']==lock['source_commit']==terminal['source']
assert result['config_sha']==initial['profile_sha'] and initial['job_id']=='61813'
harmonic=3/sum(1/metrics[k]['success_pct'] for k in ['R','P','N'])
row=dict(rerun_attempt='execution-r1',model='Qwen2.5-7B-Instruct',method='PRICE Q3-beta250 (selected settings, final eval2k)',dataset='CF',ordered_stream_sha256=sha(stream),cold_state_identity=initial['cold'],source_commit=result['source'],config_sha256=result['config_sha'],server='server3',job_id='61813',job_name='qwen-price-q3-eot-2k',observed_state='W20_FACTUAL_COMPLETE',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),main_table_eligible=False,eligibility_reason='Q3-selected configuration; no explicit README main promotion/exception found. final scientific eval2k is not the tier tuning run itself.',Score=harmonic,Eff=metrics['R']['success_pct'],Gen=metrics['P']['success_pct'],Loc=metrics['N']['success_pct'],Flu_status='DEFERRED',Con_status='DEFERRED',metrics=metrics,checks=dict(raw_rows=26000,requests=2000,commits=20,history_appends=appends,state_joins=19,raw_identity=True,stored_aggregate_match=True),program_seconds=terminal['seconds'],scheduler_elapsed_seconds=23281,report_path=str(REPORT/'report-ko.md'),limitations=['CPU owner audit; no independent reviewer','Historical jlz observer; arithmetic validated, no new official evaluator forward or tokenizer parity certification','W20 editable weights only; exact_resume NOT_AVAILABLE','No CP load/rehash/transfer/delete; no new GPU/evaluation'])
(OUT/'summary.json').write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n');(OUT/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
with (REPORT/'table.csv').open('w') as f:
 keys=['model','method','dataset','job_id','job_name','observed_state','main_table_eligible','Score','Eff','Gen','Loc','Flu_status','Con_status'];w=csv.DictWriter(f,keys);w.writeheader();w.writerow({k:row[k] for k in keys})
(REPORT/'report-ko.md').write_text(f'''# Qwen 61813 W20 CPU 회수 검산

Job `61813` / `qwen-price-q3-eot-2k`는 cold 2k final 실행이다. 실제 arm은 `Q3-beta250`, cohort_role=eval2k_final, tier=final이다. Q3 튜닝 자체와 구별하되 Q3 선택 설정의 본표 승격 예외는 확인되지 않아 **README 자동 반영 대상에서 제외**하고 GH에 수치를 전달한다. README는 편집하지 않았다.

| 범위 | Score | Efficacy | Generalization | Specificity | Flu | Con |
|---|---:|---:|---:|---:|---|---|
| CF W20 2000 | {harmonic:.6f} | {row['Eff']:.6f} | {row['Gen']:.6f} | {row['Loc']:.6f} | DEFERRED | DEFERRED |

단위는 %. NLL strict preference(tie 실패), 요청별 동일 R1/P2/N10이므로 prompt-pair 비율과 request macro가 일치한다. 분모 R2000/P4000/N20000. raw 26000행의 case 순서·row/token identity·유한성·token/strict 분모와 저장 집계를 독립 표준 Python reducer로 검산했다. 기존 jlz observer 사용이며 최신 official evaluator 실제 forward 동등성 검증은 이번에 수행하지 않았다. TF accuracy는 summary.json에 별도 저장했다.

20 commit, 19 W/H/RNG join, history100회 및 observer state receipt 정합. 원 source `{row['source_commit']}`, config `{row['config_sha256']}`, stream `{row['ordered_stream_sha256']}`. beta_base/c/beta_max_scale=2.5, M1=false/EOT pos0, 기본 설정으로 relabel하지 않았다. 관측시각 {row['observed_at']}.

프로그램 시간 {terminal['seconds']:.3f}s, 단발 accounting COMPLETED/0:0, allocation 06:28:01 (1GPU/8CPU/58GiB). 서로 중복 합산하지 않는다. Scheduler 완료만으로 판정하지 않고 W20 raw/terminal/result/commit을 확인했다. W20 snapshot은 editable L4–L8 weights만이며 exact resume 불가; 복원/전송/삭제 없이 보존했다. 생성평가 수치는 없으며 0으로 채우지 않았다.

재현: 준비 WT에서 `python3 {OUT}/reduce.py`. 원 raw는 local 유지, inventory.json에 읽은 파일 SHA/bytes/mtime 기록. owner CPU audit이며 별도 reviewer/GPU검산/새 job/기존 job 변경 없음. 현재 이 scope에서 검산한 새 본표 자동 적격 행은 0개이며 과거 타실험을 자동 승격하지 않았다.
''')
print(json.dumps({k:row[k] for k in ['Score','Eff','Gen','Loc','main_table_eligible']}))
