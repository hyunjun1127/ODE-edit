# 사용자 중지 및 CPU 코드 파이프라인 점검

최신 사용자: “리소스가 없으니 코드 파이프라인 점검만 하고 모니터링은 중단하자”.
본 기록은 이전 T4까지 능동 관찰 지시를 대체한다. 해당 사용자 지시 이후 scheduler/log/result를 조회하지 않았다. 기존 GPU53176/53177 및 CPU collector53179는 취소·수정·재제출하지 않았다. 현재 상태를 새로 관측한 기록이 아니며 실험 완료 보고도 아니다.

## 결과: 기존 CPU16 PASS, 파이프라인 전체 PASS는 아님

2026-09-24 UTC, CUDA_VISIBLE_DEVICES=''와 bytecode 저장 금지로 기존 CPU16 회귀검사를 실행하여 16/16 PASS(2.777초)를 확인했다. 전체 U 다섯 weight, 고정 설계173states/383tasks, 검열·분모·strict tie·산술·cache key·atomic 저장·토큰 불일치 거부·합성5그림·collector 최종 완료 기록 순서를 검사했다. 합성자료이며 모델 실행/실제 수치 검증은 아니다. 원 CP24 재해시·평가·모델 load·GPU0.

정적 소스 검토와 별도 최소 CPU 반례에서 **실제 제출 source의 T1 routing 오류**를 재현했다. 따라서 CPU16 PASS를 실행 파이프라인 전체 정상으로 확대하지 않는다.

| 항목 | 확인 사실 |
|---|---|
| 원 실행 source | `6ef71ed2a7f7639c6619c34e89314c1930288c0a` |
| 실행 lock SHA | `bc8ae75e983c68a1eb4489e9c3be98e0f77f8a567cf32e9eb88d6b9e1c994d9f` |
| backend.py SHA | `953bebe487bcefd5099be2eedb5db9e6c180733b31a8dffb0283720fe6b6fc9c` |
| 발견 위치 | `Backend.state`, backend.py:52–54; 호출 `Run.t1`, worker.py:113–115 |
| 입력 | ACTUAL recipe + force_removal=(1,[0]) |
| 재현 | `KeyError: construction_endpoint`, endpoint 호출0, 모델/GPU0 |
| 제출 source 적용 | WT backend와 frozen attempt-v1 backend의 bytes/SHA 동일 |
| 영향 | 해당 T1 대각선 검사에 도달하면 override 적용 전에 예외. 두 family가 같은 경로를 사용한다. 이후 단계 PASS 불가 |
| 실제 job 실패 여부 | 이번에는 조회하지 않음. 소스 반례이지 실제 scheduler 실패 관측이 아님 |
| 수리/Slurm 변경 | 수행하지 않음. 최신 지시는 점검만 승인 |

`Backend.state`는 force_removal이 있을 때 ACTUAL recipe에 없는 construction_endpoint와 removed_cohort_indices를 먼저 읽고, 그 다음에 force_removal을 적용한다. 최소 수리 제안은 명시 override를 먼저 분기하여 t/remove를 정한 뒤 나머지 recipe를 해석하는 것이다. 이번에는 제안만 기록하고 원/frozen source·threshold·job은 변경하지 않았다. 기존16개 테스트는 이 실제 routing 호출을 직접 실행하지 않아 결함을 포착하지 못했다.

## DAG·보존 경계

두 family persistent worker의 T1/T2P/T2F/T3B matching atomic PASS join, CPU53179의 afterany:53176:53177 및 실패 시 science를 통과시키지 않는 collector 경로를 소스에서 확인했다. 이는 실제 gate PASS나 현재 queue 상태 확인이 아니다. 두 worker의 단계별 상호 대기 때문에 한 lane만 확보되면 먼저 시작한 worker가 peer를 기다릴 수 있다. 실제 대기시간/할당비용은 이번에 측정하지 않았다.

이전 사용자 지시 전에 교체한 CPU53178→53179와 collector 완료 기록 순서 수리는 역사 그대로 유지한다. 이번에는 수리나 신규 제출을 하지 않는다. 기존 제출 프로그램의 내부 gate/오류 기록은 그대로 자연 진행하며, agent의 scheduler/log/result polling·heartbeat·자동 재개는 중단한다. `monitoring_active=false`, `automatic_resume=false`. 이후 수리·실제 상태 확인은 사용자 recall 후 판단한다.

모델 GPU T1/실제 peak/최종 과학 결과는 NOT_OBSERVED이며, 이번 점검으로 numerical validation PASS를 주장하지 않는다. Owner 직접 검토 및 CPU 반례이며 별도 독립 red agent는 사용하지 않았다. NO_BROADCAST_NOT_REQUIRED.

## 재현

저장소 root에서 CPU 전용으로 다음을 실행한다. 실험 output/log는 읽지 않는다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /data/janghj/EasyEdit/.venv/bin/python -m unittest project.run_scripts.historical_update_timeaxis.test_core project.run_scripts.historical_update_timeaxis.test_reducer -v
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python3 audits/servers/server4/historical-update-timeaxis-20260924-v1/user-pause-r1/reproduce_force_removal.py
```
