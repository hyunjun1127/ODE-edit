# Native weak B010 — 초기 실행 인계

ACK `ODEEDIT-GH-SH2-NATIVE-WEAK-B010-NLL1-20260930-R1`. Job **55413**는 held owner/resource/source 검사 후 release했고, 실제 첫 요청 target fit 1회·native 5solve·history append5와 두 번째 entry W/M/context/RNG/anchor 연결을 확인했다. `MONITORING_PAUSED_AWAITING_USER`이며 이 보고는 최종100요청 완료가 아니다.

## 고정 범위와 구현

B010 원본 부모의 다섯 FP32 W/M/context/RNG에서 동일 metadata500 prefix100, BS1×100 한 새 `NATIVE_WEAK_NLL1` 경로다. 원 native compute_z의 `loss < .05`만 `nll_loss <= 1.0`으로 바꾼 task-local 사본을 사용한다. AST 나머지 동일, 원 objective/Adam/clamp/blue=false/L2=10/5층 writer는 변경하지 않았다. 최초 crossing을 backward 이전에 반환하고 초기 충족0Adam·마지막forward crossing·25forward24Adam exhaustion을 구별한다. 실제 강도 동등성을 사전 주장하지 않는다.

CPU stop/AST/nonfinite/trace/noCP 10tests와 기존 reducer6tests PASS. 원 catalog1628rows 및 final2600candidate row identity exact; 패널144pair. CPU 준비 중 authority manifest의 원WT/실행archive 경로 혼재를 수정했고 model/GPU 실행 전이었다. 원파일 변경0.

## 실제 최초 기술 관측

- 첫 요청ID11336, fit 초기 NLL 10.353504181, 마지막 latent NLL 0.793096900.
- 종료 `NLL_THRESHOLD_REACHED`, loss evaluations 3, Adam 2. Undershoot는 결과이며 보정하지 않았다.
- 실제 five-weight write 후 여섯 rewriting context 평균 NLL 1.050809383. Latent fitting 수치와 구분한다.
- 첫 native method 23.541s. 전체100step wall/할당/GPU peak/최종성능은 아직 terminal 수집하지 않았다.
- first commit과 second entry가 정확히 연결된다. 이는 초기 correctness이며 모든 요청의 성능·완료 인증이 아니다.

## 저장·평가·자원

새 edited weight/M/resume bundle 저장0, `save_checkpoints=false`, `exact_resume=NOT_AVAILABLE`. 등록프로그램은 최종 RAM의 R100/P200/N1000을 종료 전에 평가하고 기존 greedy 및 CPU paired comparison을 수행한다. 기존3arm 최종 raw는 읽기 전용 재사용하며 재평가하지 않는다. NLL/TF/token/statehash/scalar ledger만 local에 보존한다.

Project cap3(최신 명시 authority)/task1, 1GPU/8CPU/60416MiB/exportNONE/Requeue0, wall8h. 같은host 원native12778GPU-sec·CUDA34.744GiB·host32.928GiB 및 추가600context/2600finalcalls 기반의 wall 여유이며 새측정총비용은 아니다. 기존job 변경0/새arm추가0. Source `30ef8b02327746edef82a889c0bacd9f961f8f3c`, tree `0a2c2296a01ed33a25655aca2a087e9d3f7c1799`; lock SHA `c682fa528498fba9475a15be168b6b3efd6bb843d4a197c0d46919211bb56491`. Output `/mnt/raid5/janghj/ODE-edit/local/native-weak-b010/20260930-v1/output`.

## 재현 및 다음 경계

`project/run_scripts/native_weak_b010/README.md`의 prepare/tokens/freeze/held submit 명령과 local source/lock을 참조한다. 기존 frozen source/archive/config는 수정하지 않는다. 입력·실행·publication source를 구분하며 publication은 이 보고를 포함하는 commit이다.

실제 initial-link 확인 직후 agent polling/terminal 대기/후속 제출은 중지한다. 등록프로그램 자체는 계속하며 사용자 recall 뒤만 상세 완료 리뷰한다. 새 comparison 결과는 NOT_YET_REVIEWED. Raw/CP/model/prompt/fullstdout Git0, `NO_BROADCAST_NOT_REQUIRED`, `scientific_promotion=false`.
