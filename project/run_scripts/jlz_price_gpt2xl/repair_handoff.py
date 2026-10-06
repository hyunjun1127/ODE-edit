"""One-shot CPU handoff of the already observed repair failure-point gate.

Only immutable first candidate prefixes are reduced. No scheduler, later
progress, model, evaluator, monitoring loop, or scientific resume is invoked.
"""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from .common import *
from .repair import NEW
from .storage import write_bytes

def first_prefix(cell,config,lock,local):
    folder=NEW/cell/'batch-01'
    path=folder/'events.jsonl'
    with path.open('rb') as f:raw=f.readline()
    require(raw.endswith(b'\n'),'FIRST_COMPLETE_PREFIX')
    row=json.loads(raw);p=row['payload']
    require(row['task']==TASK and row['arm']==cell and row['batch']==1 and row['event']=='candidate'
        and p['candidate']==0 and p['ordinal']==1 and p['backward'] is True and not p['terminal']
        and p['gradient_status']=='MASKED_REQUEST_SUM_MEASURED','ACTUAL_OLD_FAILURE_POINT_RETURNED')
    require(len(p['F'])==100 and len(p['nll'])==100 and all(len(v)==6 for v in p['nll'])
        and all(math.isfinite(v) for v in p['F']),'INITIAL_FULL_B100_FINITE')
    active=p['active_mask'];updates=p['post_update_controller']['update_counts']
    require(len(active)==len(updates)==100 and all(type(a) is bool and u==int(a)
        for a,u in zip(active,updates)),'INITIAL_EXACT_FIRST_UPDATE_COUNTS')
    entry=json.loads((folder/'entry.json').read_text())
    require(entry['source']==lock['source_commit'] and entry['state']==config['models']['MEMIT']['cold_W0_H0']
        and entry['ids']==config['models']['MEMIT']['packs'][0]['ids'],'INITIAL_SOURCE_COLD_ORDER')
    reuse=json.loads((NEW/cell/'W0/cap-reuse.json').read_text())
    require(reuse['no_forward'] and reuse['no_raw_copy'] and reuse['no_checkpoint'],'INITIAL_W0_REFERENCE_ONLY')
    saved=local/(cell+'-candidate0-prefix.jsonl');write_bytes(saved,raw,limit=2*1024**2)
    return dict(cell=cell,status='ACTUAL_B1_C0_BACKWARD_AND_FIRST_UPDATE_RETURNED',candidate=0,
        batch=1,requests=100,active=sum(active),request_updates=sum(updates),backward=True,terminal=False,
        execution_source=lock['source_commit'],entry=member(folder/'entry.json'),
        source_stream=str(path),prefix_offset=0,prefix_bytes=len(raw),prefix_sha256=hashlib.sha256(raw).hexdigest(),
        sealed_prefix=member(saved),W0_reference_only=member(NEW/cell/'W0/cap-reuse.json'),
        no_claim_on_later_candidates_or_B1_commit_or_B2_or_W20=True)

def csv_file(path,rows,fields):
    out=io.StringIO(newline='');w=csv.DictWriter(out,fieldnames=fields);w.writeheader();w.writerows(rows)
    write_bytes(path,out.getvalue().encode(),limit=1024**2)

def handoff():
    c=json.loads((NEW/'config.json').read_text());lock=json.loads((NEW/'execution.lock.json').read_text())
    sub=json.loads((NEW/'submission.json').read_text())
    require(sha(NEW/'config.json')==lock['config_sha256'] and sub['source']==lock['source_commit'],
        'HANDOFF_SOURCE_CONFIG')
    rec=json.loads(verify(c['repair']['reconciliation']).read_text())
    cpu=json.loads(verify(c['cpu_preflight']).read_text())
    local=LOCAL/'checkpoint-repair-r1-handoff';local.mkdir(exist_ok=False)
    initial=[first_prefix(cell,c,lock,local) for cell in ('MEMIT_CAP075','ALPHAEDIT_CAP075')]
    value=dict(task_id=TASK,status='MONITORING_PAUSED_AWAITING_USER',execution_source=lock['source_commit'],
        execution_tree=lock['source_tree'],jobs=sub['jobs'],lock=member(NEW/'execution.lock.json'),
        source_archive=lock['archive'],config=member(NEW/'config.json'),submission=member(NEW/'submission.json'),
        held=member(NEW/'held-inspection.json'),repair_authority=c['repair']['reconciliation'],
        old_allocated_GPU_seconds=rec['allocated_GPU_seconds'],initial=initial,
        cpu_tests=dict(run=cpu['tests_run'],passed=cpu['tests_run']-cpu.get('skipped',2),
            skipped=cpu.get('skipped',2),model_GPU_validation=False),
        noCP=True,exact_resume='NOT_AVAILABLE',all_old_files_KEEP=True,
        new_allocation_cost='NOT_YET_TERMINAL; do not substitute initial elapsed',
        monitoring_active=False,automatic_retry=False,automatic_resume=False,
        B1_commit='NOT_OBSERVED',B2='NOT_OBSERVED',W20='NOT_OBSERVED',
        science_changed=False,broadcast='NO_BROADCAST_NOT_REQUIRED')
    write(local/'initial-receipt.json',value)
    report=ROOT/'experiment-reports/servers/server1/jlz-price-gpt2xl-2k/checkpoint-repair-r1'
    audit=ROOT/'audits/servers/server1/jlz-price-gpt2xl-2k/checkpoint-repair-r1'
    rows=[]
    for cell in CELLS:
        j=sub['mapping'][cell];old=rec['after'][rec['jobs'][cell]]
        rows.append(dict(cell=cell,old_job=rec['jobs'][cell],old_state=old['State'],
            old_GPU_seconds=int(old['ElapsedRaw']) if 'gres/gpu=1' in old['AllocTRES'] else 0,
            new_job=j['job'],dependency=j['dependency'] or 'NONE',source=lock['source_commit'],
            initial='B1_C0_BACKWARD_AND_UPDATE_OBSERVED' if cell.endswith('CAP075') else 'NOT_OBSERVED'))
    csv_file(report/'job-lineage.csv',rows,list(rows[0]))
    write(audit/'initial-receipt.json',value)
    manifest=dict(input_reuse=c['repair']['reuse'],execution_source=lock['source_commit'],
        CPU=member(verify(c['cpu_preflight'])),parent_lock=rec['parent_lock'],parent_submission=rec['parent_submission'],
        reconciliation=c['repair']['reconciliation'],new_lock=member(NEW/'execution.lock.json'),
        new_submission=member(NEW/'submission.json'),new_config=member(NEW/'config.json'),
        prefixes=[v['sealed_prefix'] for v in initial],raw_local_only=True,
        scientific_source_changed=['adapter.py parity-local pos -> prediction_positions only'],
        numerical_gates_precision_optimizer_budget_unchanged=True,
        independent_review=dict(pre='bounded source reviewer: no additional blocking issue found for 4b841d78',
            reproduction='independent CPU production-checkpoint fixture, 2/2 PASS',
            post='independent CPU first-prefix/source/cold/updates validation only; no later progress'))
    write(audit/'manifest.json',manifest)
    text='''# GPT2-XL PRICE checkpoint closure 수리 및 재제출

사용자 직접 repair recall로 같은 여섯 cell만 수리·재제출했다. 기존 source/raw/로그/실패 비용은 KEEP이다.

## RCA와 수정

기존 60094/60095/60096/60097/60098은 모두 B1 candidate 0 첫 backward의 `CheckpointError`로 실패했고 commit은 0회다. `Adapter.masked`의 native C0 parity 관측 루프가 checkpoint closure의 7개 lookup 인덱스 `pos`를 마지막 KL row의 1개 prediction 인덱스로 재할당했다. backward 재계산의 저장 metadata가 [7]→[1]로 바뀌었다. OOM·성능 gate·과학 수식 문제가 아니다.

parity-local 이름을 `prediction_positions`로 분리했다. Activation checkpoint/determinism 검사를 끄지 않았다. 목적/precision/threshold/계수/20평가·19update/분모는 원 계약대로 유지한다. 여기서 checkpoint는 autograd activation recompute이며 영속 model checkpoint 저장과 다르다.

동일 production masked/recompute 함수의 CPU fixture로 frozen 실패를 재현하고, 수정 후 checkpoint enabled/disabled 출력·keys·bases·5개 gradient의 정확 일치를 확인했다. 전체 CPU 55개 중 53 PASS/2 SKIP이며 모델 GPU parity/전체 실행 PASS가 아니다. 독립 source reviewer와 독립 first-prefix CPU reducer 범위도 구분한다.

## 취소와 재사용

원 collector 60100을 먼저, 미시작 ALPHAEDIT_FREE100 60099를 다음에 exact owner/name/launcher/source/elapsed0/할당없음 및 PENDING 조건으로 취소했다. 이미 FAILED인 5개는 다시 취소하지 않았다. old exact queue empty/terminal accounting으로 해제를 확인했다. 다른 실행 job은 취소하지 않았다.

원 5개 할당 비용은 872+45+42+49+49 = **1,057 GPU초**다. 원 W0 관측 815.182초와 native context 준비 11.857초는 이 lineage 내부이며 다시 합산하지 않는다. 아직 끝나지 않은 새 job의 전체 비용은 미측정이다.

검증된 native context READY(244B context, SHA `394232472ab1a4b72dcfa7662169fa5195a42c43350676df2e73683636120ed7`)와 동일 W0의 40 chunk/2,000 요청/26,000 prompt-pair를 참조 재사용한다. 원 실제 evaluator source/runtime/source/config/cold/row/token/order/summary를 CPU에서 검산했고 새 GPU startup의 runtime/cold hash를 재검사했다. raw 복제/context 재생성/W0 추가 forward 0이다. noCP이므로 편집 prefix resume가 아니며 여섯 run은 각각 fresh cold W0/H0다.

## 실제 제출과 관측 경계

| writer | CAP075 | CAP100 | FREE100 |
|---|---:|---:|---:|
| MEMIT | 60124 | 60125 afterany60124 | 60126 afterany60125 |
| AlphaEdit | 60127 | 60128 afterany60127 | 60129 afterany60128 |

collector 60130은 새 6개 모두 afterany다. READY가 이미 검증돼 두 CAP075 head 사이 dependency는 없다. 현재 own server1 resource/DAG를 검산하고 cap2, 각 GPU1/CPU8/64GiB/48h, collector GPU0/CPU8/24GiB/2h, devbox/exportNONE/Requeue0를 전체 held 검사한 뒤 release했다. 이 wall은 ETA/GPU-hour hard budget이 아니다. 취소 old ID는 새 dependency로 쓰지 않았다.

60124와 60127 각각 B1 candidate 0 전체 backward 완료, active100, request update100, finite F100 및 첫 update의 저장 기록을 확인했다. **기존 실패 지점 실제 통과**다. B1 terminal commit/H append/observer/B2/W20은 NOT_OBSERVED이며 남은 4개 초기 gate도 NOT_OBSERVED다. candidate0 통과를 2k 완료나 수치적 전체 동등성으로 확대하지 않는다.

사용자가 지정한 실패 지점 확인 후 `MONITORING_PAUSED_AWAITING_USER`. 등록 runner/collector는 자연 진행하며 agent polling/heartbeat/자동 retry/다른 task 조회는 중단한다. W&B는 기존 scientific schema/job-ID/고유 attempt 정책을 새 source에 유지하되 이번 CPU/초기 handoff를 전체 remote delivery 인증으로 쓰지 않는다.

## 재현·증거

```bash
/mnt/raid5/janghj/EasyEdit/.venv/bin/python -m unittest project.run_scripts.jlz_price_gpt2xl.test_checkpoint
```

실행 source: `SOURCE_SHA`. Lock SHA: `LOCK_SHA`.
Local attempt: `/mnt/raid5/janghj/ODE-edit/local/jlz-price-gpt2xl-2k/attempt-checkpoint-repair-r1`.
보고 표: [job-lineage.csv](job-lineage.csv). Rooted evidence는 repo의 `audits/servers/server1/jlz-price-gpt2xl-2k/checkpoint-repair-r1/`에 있다. first-prefix는 local-only로 봉인했고 full stdout/prompt/tensor/model/raw는 Git에 넣지 않았다. NO_BROADCAST_NOT_REQUIRED.
'''
    text=text.replace('SOURCE_SHA',lock['source_commit']).replace('LOCK_SHA',sha(NEW/'execution.lock.json'))
    write_bytes(report/'factual-report-ko.md',text.encode(),limit=1024**2)
    write(audit/'rooted-receipt.json',dict(report=member(report/'factual-report-ko.md'),
        tables=[member(report/'job-lineage.csv')],manifest=member(audit/'manifest.json'),
        initial=member(audit/'initial-receipt.json'),source=lock['source_commit']))
    for rel in ('audits/servers/server1/jlz-price-gpt2xl-2k/checkpoint-repair-r1/ack.json',
                'audits/servers/server1/jlz-price-gpt2xl-2k/checkpoint-repair-r1/owner-handoff.json',
                'runs/jlz-price-gpt2xl-2k/checkpoint-repair-r1/receipt.json'):
        # Existing-task-only names; old submission/status provenance retained.
        write(ROOT/rel,dict(task_id=TASK,status=value['status'],jobs=sub['jobs'],source=lock['source_commit'],
            lock_sha256=sha(NEW/'execution.lock.json'),report=member(report/'factual-report-ko.md'),
            monitoring_active=False,automatic_retry=False,initial='BOTH_WRITERS_OLD_FAILURE_POINT_PASSED',
            W20='NOT_OBSERVED'))
    print(json.dumps(dict(status=value['status'],jobs=sub['jobs'],source=lock['source_commit'],
        report=member(report/'factual-report-ko.md'),lock_sha256=sha(NEW/'execution.lock.json'))))

if __name__=='__main__':handoff()
