# SH4 official fresh main 재실험/표 보고

OWNER_ACK nonce=GH-SH4-MAIN-TABLE-FRESH-RERUN-20261009-R1 server=server4 accepted=true.
정본 main1bedeaf067cfc82af40e769eebace7703f0df558의 envelope와
control/main-results-policy.json 전체를 읽고 기존 clean own branch에 병합했다.
dirty root 및 다른 task 변경을 보존했다.

## 범위와 현재 상태

| 모델 | 방법 | CF | zsRE |
| --- | --- | --- | --- |
| Llama3 | AlphaEdit | NOT_SUBMITTED | NOT_SUBMITTED |
| Llama3 | AlphaEdit-BLUE | NOT_SUBMITTED | NOT_SUBMITTED |
| Llama3 | AlphaEdit+SPHERE | NOT_SUBMITTED | NOT_SUBMITTED |

sample/order가 다른 과거 CP 재사용을 이유로 AlphaEdit CF를 제외하지 않는다.
기존 runner는 이미 세 방법×두 dataset을 모두 생성하며 이번에 fresh 정책을
명시 결속했다. 과거 CP/source/raw 삭제·전송·덮어쓰기·기존 job hotpatch 없음.
현재 eligible 새 main job ID/name은 없다. README의 해당 칸은 빈칸이다.

한정 scheduler 조회의61418 PRICE OURS와61618 historical replay는 본 신규
official main의 job이 아니다. 두 job 및 별도 Qwen hparam task를 변경하지 않았다.
과거 cancelled60917–60923도 재활성화하지 않는다.

## 실제 변경·검산

- own main_results.py: rerun_attempt/model/method/dataset/ordered sample SHA/
  cold identity/source/config/server/actual job ID+name/state/time/report 경로.
- 실제 main 등록 후 한 번의 scheduler snapshot, 실행의 cold-origin 및 W20
  factual/generation 실측 event를 연결. README 직접 쓰기/반복 poll 없음.
- 같은 fresh chain origin 없는 main checkpoint resume를 차단. 기존 승인
  B2→B3 qualification resume는 main 표에서 제외한다.
- RUNNING은 ING:name, PENDING은 PENDING:name, 미제출은 빈칸.
  실패·취소·B19 이하·cold identity 없는 값을 W20 성적으로 기록하지 않는다.
- SH4에는 FLU/CON 연기 지시가 없어 기존 W0/W20 generation 계약을 유지한다.
  CF6칸과 zsRE3칸은 분리한다.

CPU own29 PASS(새 main-report 5 포함), source157 SHA/Python243/import0 PASS.
CPU mock job 번호는 실험/제출 증거가 아니며 W&B 업로드하지 않았다.
실제 GPU/native/resume/online/W20 PASS는 없다.

## 미제출 사유

1. main의 portable W0 reader repair/actual producer READY가 미결속이고 local
   assets.w0.cf 및 assets.w0.zsre는 null이다. 두 번째 fullW0를 생성하지 않는다.
2. 공통 NumPy AlphaEdit display reducer와 logger Python-round 검산 불일치를
   현재 source에서 다시 재현했다: OFFICIAL_DISPLAY_SCORE_MISMATCH.
   공통 helper/scorer를 own scope 밖에서 수정하거나 값을 변조하지 않았다.
3. 실제 native parity/연속B3 vs B2-resume와 reviewed main freeze가 미완료다.

이는 추가 USER 승인 대기가 아니라 source/input/technical gate다. 허위 READY나
의미 없는 held main 제출로 이를 우회하지 않았다. 새 source/report는 전용 branch에
게시하며 GH가 검토 후 README/main을 통합한다.

상세 6행 identity는 audit/main-table-fresh-rerun-ledger.json에 있으며
source_candidate와 executed source를 구분한다. 원 CP 보존 및 미래 job archive
정책은 별개로 유지하며 이번 작업에서 실제 CP 전송/삭제0이다.
