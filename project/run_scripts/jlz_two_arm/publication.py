"""Raw-free owner preparation/submission receipts, no scheduler or model calls."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
from .common import ROOT,LOCAL,INSTRUCTION,write,member,sha

TAG='jlz-twoarm-bs100x20-20261002-v1'
OVERRIDE='ODEEDIT-GH-SH4-JLZ-TWOARM-USE-BOTH-GPUS-20261002-R1'

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);p.add_argument('--cpu',type=Path);a=p.parse_args()
    prep=LOCAL/'preparation-v1';config=json.loads((prep/'configuration.json').read_text())
    audit=ROOT/'audits/servers/server4'/TAG;report=ROOT/'experiment-reports/servers/server4'/TAG
    status=dict(instruction_id=INSTRUCTION,override_nonce=OVERRIDE,owner='SH4',host='server4',
          session='01a04939-b5c7-7a03-ba2d-ef3343d62cfd',cwd=str(ROOT),
          recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          state='IMPLEMENTING_NOT_SUBMITTED',job_ids=[],actual_gpu_validation='NOT_RUN',
          checkpoint_saved=False,exact_resume='NOT_AVAILABLE',project_cap=2,task_cap=2,
          scheduling='prep shared once -> independent pilot A/B -> independent main A/B -> afterany collector',
          scientific_scope='each arm BS4x2 cap32 pilot then fresh W0 BS100x20 cap120 main; eta0/1',
          baselines='REUSE exact W20 first2000 quality; only same8 BS4x2 native functional pilot newly run',
          numerical_fidelity_policy='RECORD_ONLY_USER_DIRECTED',numerical_certification='NOT_ESTABLISHED',
          full_read=member(prep/'full-read.json'),broadcast='NO_BROADCAST_NOT_REQUIRED',
          other_jobs_mutated=False,source_archive='NOT_FROZEN_YET')
    if a.cpu:
        tests=json.loads(a.cpu.read_text());write(audit/'cpu-tests.json',tests);status['cpu_tests']=tests
    if a.run:
        lock=json.loads((a.run/'execution.lock.json').read_text());submission=json.loads((a.run/'submission.json').read_text())
        status.update(state='REGISTERED_RELEASED_INITIAL_NOT_OBSERVED',job_ids=submission['jobs'],
                      execution_source=lock['source_commit'],source_archive=lock['source_archive'],
                      lock=member(a.run/'execution.lock.json'),config=member(a.run/'config.json'))
        if (a.run/'handoff.json').exists():status.update(json.loads((a.run/'handoff.json').read_text()))
        write(audit/'submission.json',submission)
        inspection=json.loads((a.run/'held-inspection.json').read_text())
        write(audit/'held-inspection.json',inspection)
    write(audit/'full-read.json',json.loads((prep/'full-read.json').read_text()))
    write(audit/'source-review.json',dict(independent_reviewer='red_source_audit',scope='bounded CPU/source only; no GPU/no scheduler',
           fixes=['trial overflow -> nonmutating FloatingPointError rejection; initial/final still fatal',
                  'baseline native raw state schema/commit binding and identity convention bridge',
                  'reference list values reduced with exact membership',
                  'shared config+execution-lock matching stage join',
                  'cap helper exit4 distinguished; expanded array IDs; exact held dependency/argv',
                  'cross diagnostic token/wall counters; bounded same-run incumbent trace'],
           limitations=['runtime actual checks remain pending','nonselected pointer/version guard, not whole-model byte scan',
                        'MEMIT-H raw lacks state field; immutable B020 source+commit bridge only',
                        'cross-runtime baseline quality, no matched speed claim']))
    write(audit/'inputs-summary.json',dict(inputs=config['inputs'],reference_pool=config['reference_pool'],
           baseline_reuse=config['baselines'],schedules_counts={k:len(v) for k,v in config['schedules'].items()},
           read_only_assets=True,remote_payload='exact9 MEMIT-H completed W20 receipt/raw files only',
           receiver=member(LOCAL/'inputs/memit-h-baseline/receiver-receipt.json'),resource_plan=config['resources']))
    write(ROOT/'messages/acks/server4/2026-10-02-jlz-twoarm-bs100x20.json',status)
    write(ROOT/'messages/server-heads/server4/2026-10-02-jlz-twoarm-bs100x20.json',status)
    write(ROOT/'tasks/status'/TAG/'server4.json',status)
    text='''# JLZ two-arm: 구현·제출 인계

본 문서는 새 A/B 과학 결과 보고가 아니라 승인된 실행의 준비·등록 사실이다.
η=0/1 외의 목적함수·reference·route·예산은 공통이며 각 arm은 fresh W0/H0에서 독립 진행한다.

## 범위와 상태

'''+f"- 상태: `{status['state']}`\n- 실제 job mapping: `{status['job_ids']}`\n- Actual GPU 검증: `{status['actual_gpu_validation']}`\n"+'''
공통 W0 teacher/기술 준비 뒤 BS4×2 A/B pilot과 BS100×20 A/B main을 각각 두 독립 1GPU lane으로 구성한다.
각 GPU job은 8CPU/60416MiB, project/task cap2, exportNONE/Requeue0이다. 과거 모든 task·job은 불변이다.
source freeze/held 검사/release와 actual scientific validity는 별개다.

## 정독·입력·구현

정본7개 및 실행/2GPU overlay를 전체 읽고 exact SHA/size 검산했다. 같은 S4 모델 revision/C0/context/data를
CPU fullSHA/shape로 결속했다. Reference pool은 11919→11544→11134→10715문항/34relation;
2pilot+20main의 문항·reference order는 outcome 전에 고정했다. 새 prompt/teacher/tensor는 Git에 넣지 않는다.

구현은 inactive current NLL/KL을 포함하며 reference token→prompt 평균에 B를 곱한다.
모든 token의 실제 다층 FP32 weight 경로와 dX를 보존한다. M=PᵀAP의 small/direct 검산과 ηΩ가 독립 CPU 검사를 갖는다.
SPG/BB/prox/linesearch는 원 수학을 유지하며 final original oracle 1회를 동일 cap32/120 안에 예약한다.
유한 수치차이는 원 PASS/FAIL을 보존한 기록 전용이다. Identity/IO/state/clamp/초기·최종 비유한 값은 여전히 차단한다.

## 재사용 baseline 첫 독립 CPU 표

| W20 baseline | R (2000) | P (4000) | N (20000) | 재사용 경계 |
|---|---:|---:|---:|---|
| BASE_ALPHAEDIT | 1986 | 3729 | 13718 | local paired raw + B020 endpoint exact |
| BASE_MEMIT | 1295 | 2468 | 10365 | local paired raw + B020 endpoint exact |
| MEMIT-H | 1988 | 3640 | 15809 | S3 completed B020 source/commit bridge; raw state field 미저장 |

각 raw26000행의 case/prompt/target/order/finite/strict tie-failure/TF 분모를 독립 CPU reducer로 확인했다.
이 표는 새 JLZ 결과가 아니다. 역사 TF4.44.2와 이번4.57.1, H200/Blackwell 및 TF32 차이는 그대로 기록한다.
따라서 matched speedup은 주장하지 않는다. 비교용 새2k baseline fitting은0, 같은 first8 BS4×2 기능 확인만 별도 비용이다.

## 저장·자원·미측정

신규 W/H/R/delta/optimizer checkpoint0, exact_resume=NOT_AVAILABLE. RAM transaction만 사용한다.
저장 teacher는 immutable W0 reference input이며 edited-state checkpoint가 아니다.
GPU peak 계획80GiB/host48GiB는 **예상**이며 actual은 Slurm program receipt로 별도 남긴다.
host 요청은59GiB 이하, disk reserve20GiB, finite wall7days는 ETA나 과학 callbudget이 아니다.
최초 actual fixed-candidate comparison 뒤 공통 route를 성능결과 아닌 시간/메모리로 고정한다.
기술≤6 B100 calls, pilot≤128, main≤4800 및 teacher/native/observer 비용은 분리한다.

대표 main B1 finalcommit+5history+observer restore→B2 ownentry가 실제 확인되거나,
전체등록/release 뒤 실제 resource부족으로 시작하지 못하면 agent monitoring을 중단한다.
등록된 runner/collector만 자연 진행하며 이후 상세 리뷰는 사용자 recall에서 수행한다.

NO_BROADCAST_NOT_REQUIRED: 동일 S4 실행이며 승인된 작은 completed-baseline 파일 이외 대형 전송0.
자기검산과 독립 red CPU/source 감사를 사용했으며 actual GPU 감사는 아직 주장하지 않는다.

## 재현

`python -m unittest discover -s project/run_scripts/jlz_two_arm -t . -p 'test_*.py'`

실행은 전용 immutable local attempt의 config/lock/archive/launcher와 제출 receipt를 따른다.
원 실행/평가 source는 수정하지 않고 새 task-local namespace만 구현했다.
'''
    report.mkdir(parents=True,exist_ok=True)
    (report/'report-ko.md').write_text(text)
    plan=ROOT/'plans/updates/server4'/TAG;plan.mkdir(parents=True,exist_ok=True)
    write(plan/'execution-plan.json',status)
    print(json.dumps({'state':status['state'],'jobs':status['job_ids'],'report':str(report/'report-ko.md')}))

if __name__=='__main__':main()
