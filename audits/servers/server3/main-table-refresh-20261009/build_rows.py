import pathlib,json,hashlib,datetime,csv
A=pathlib.Path(__file__).parent;R=pathlib.Path('experiment-reports/servers/server3/main-table-refresh-20261009');now=datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(pathlib.Path(p).read_text())
def member(p):
 p=pathlib.Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
q=read(A/'qwen/summary.json');old=read('audits/servers/server3/official-baselines-20261008/qwen-61813-review/summary.json')
keys=['Score','Eff','Gen','Loc'];delta={k:q[k]-old[k] for k in keys};assert all(v==0 for v in delta.values())
q.update(main_table_action='KEEP_Q3_SEPARATE_REPORT_NO_PROMOTION',old_to_new=dict(old={k:old[k] for k in keys},new={k:q[k] for k in keys},delta=delta),raw_inventory=str(A/'qwen/inventory.json'))
lroot=pathlib.Path('/data/janghj/ODE-edit/local/llama-price-final-2k-20261009/execution-r1');initial=read(lroot/'final/initial.json');profile=read(lroot/'final/resolved-profile.json')
l=dict(model='Llama3-8B-Instruct',method='PRICE L1 selected settings',dataset='CF',job_id='61821',job_name='llama-price-L1-2k',observed_state='RUNNING',observed_at=now,source_commit=initial['source'],config_sha256=initial['profile_sha'],slice_identity=initial['slice'],cold_state_identity=initial.get('cold'),rerun_attempt='execution-r1',server='server3',main_table_eligible=False,main_table_action='STATUS_ONLY_PENDING_EXACT_MAIN_ELIGIBILITY; preserve existing 60103 exception',reason='Selected L1 final run; W20 result/terminal absent in bounded snapshot, 0 commits then. No completed metric.',metrics=None,report_path=str(R/'report-ko.md'))
inventory=[dict(task='Qwen Q3 final',job='61813',classification='COMPLETE_CF_Q3_SEPARATE',evidence=str(A/'qwen/summary.json')),dict(task='Llama L1 final',job='61821',classification='RUNNING_NO_W20',evidence=member(lroot/'final/initial.json')),dict(task='official baseline server3',job=None,classification='NOT_SUBMITTED_ON_SERVER3',evidence=member('/data/janghj/ODE-edit/local/official-baselines-20261008/submission-request-preflight-20261009.json')),dict(task='zsRE reeval server3',job=None,classification='NOT_APPLICABLE_NO_CP',evidence=member('audits/servers/server3/zsre-2k-reeval-20261009/inventory.json'))]
for task,path,reason in [('MEMIT-H54007','memit-history-fixed10k/20260928-v1/attempt-repair-r1/output/terminal.json','historical MEMIT-H variant; not stock MEMIT/main exception'),('MEMIT-HJ','memit-hj/20260930-v2/attempt-v1/output/terminal.json','historical experimental variants/technical incomplete'),('JLZ-v10A57699','jlz-realized-subject-v10/20261003-v1/attempt-v2/main-A/terminal.json','historical 500 horizon, not W20/2k'),('JLZ-v12','jlz-v12-shared-budget/20261004-v1/attempt-s3-r1/main-V12_MAIN/terminal.json','historical JLZ-v12 variant, no main exception'),('JLZ-v12Alpha pilot','jlz-v12-alphaedit-writer/20261004-v1/attempt-r1/pilot-V12_ALPHAEDIT/terminal.json','pilot excluded')]:
 p=pathlib.Path('/data/janghj/ODE-edit/local')/path;inventory.append(dict(task=task,classification='EXCLUDED_NOT_CURRENT_MAIN',reason=reason,evidence=member(p)))
obj=dict(nonce='USER-GH-ALL-SH-COMPLETED-MAIN-TABLE-REFRESH-20261009-R1-SERVER3',server='server3',observed_at=now,policy_source='99ecdba4',eligible_new_complete_main_rows=0,completed_separate_rows=1,rows=[q,l],inventory=inventory,scope='Known own local execution metadata bounded depth5; code/worktree/raw-chunk replicas excluded. Historical statuses not promoted. Exact accounting once for61813/61821; no repeated polling.',new_GPU_forward_jobs_mutations=0,reviewer='OWNER_CPU_AUDIT',broadcast='NO_BROADCAST_NOT_REQUIRED',reducer_sha256=member(A/'qwen/reduce.py')['sha256'])
(A/'table-rows.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
with (R/'table-rows.csv').open('w') as f:
 fields=['model','method','dataset','job_id','job_name','observed_state','main_table_eligible','Score','Eff','Gen','Loc','Flu_status','Con_status'];w=csv.DictWriter(f,fields);w.writeheader();w.writerows({k:r.get(k,'') for k in fields} for r in [q,l])
(R/'report-ko.md').write_text(f'''# Server3 완료 main-table 갱신 회수

최신 origin/main `99ecdba4` README 및 control/main-results-policy.json 기준, nonce USER-GH-ALL-SH-COMPLETED-MAIN-TABLE-REFRESH-20261009-R1-SERVER3. 관측 {now}. 본표 신규 완료 적격 행 **0개**, Q3 별도 완료 행 1개, L1 실행 중 1개다. README는 GH 단독 소유이므로 수정하지 않았다.

| 실행 | 상태 | Score | Eff | Gen | Loc | 본표 처리 |
|---|---|---:|---:|---:|---:|---|
| Qwen 61813 qwen-price-q3-eot-2k | W20/2000 COMPLETE | {q['Score']:.6f} | {q['Eff']:.6f} | {q['Gen']:.6f} | {q['Loc']:.6f} | 기존 Q3 별도 보고 유지 |
| Llama 61821 llama-price-L1-2k | RUNNING; W20 없음 | — | — | — | — | 상태만, 자동승격 없음 |

Qwen 원 raw26000행(R2000/P4000/N20000)을 CPU 독립 재집계하여 이전 보고와 네 지표 delta=0을 확인했다. case/row/token identity·finite·strict/count·20commit/19join/100H 검산 유지. full raw SHA/bytes는 qwen/inventory.json, source/config/cold/cohort는 table-rows.json. Flu/Con은 DEFERRED이며 W20 snapshot을 복원하거나 삭제하지 않았다. Q3-selected final은 tuning 자체는 아니지만 일반 갱신 지시로 기본 PRICE에 승격하지 않는다.

Llama는 W0/initial/profile만 확인했고 단발 snapshot에서 commit0, result/terminal 없음. 실제 accounting RUNNING을 성능 완료로 표시하지 않았다. 향후 종료를 기다리거나 polling하지 않는다. 선택 L1 source {l['source_commit']}, config {l['config_sha256']}. 기존 Llama60103 FREE100 본표 예외는 그대로 유지한다.

own 완료 zsRE checkpoint/공개-query 재평가 결과는 없다. 다른 서버 baseline migration·replica를 재집계하지 않았다. 과거 MEMIT-H/MEMIT-HJ/JLZ-v10/v12/pilot는 inventory에 제외 이유를 남겼으며 current official/PRICE로 재명명하지 않았다. 미제출 baseline과 W0/collector/qualification을 main으로 대체하지 않았다.

재현: `python3 {A}/qwen/reduce.py` 뒤 `python3 {A}/build_rows.py`. 원 raw와 모델/CP local KEEP. owner CPU audit, 별도 reviewer 없음. no GPU/model-forward/newjob/Slurm mutation/online history rewrite. compact Git만 공유(NO_BROADCAST_NOT_REQUIRED). 기존 실행은 그대로다.
''')
print(json.dumps({'eligible':0,'delta':delta,'rows':2}))
