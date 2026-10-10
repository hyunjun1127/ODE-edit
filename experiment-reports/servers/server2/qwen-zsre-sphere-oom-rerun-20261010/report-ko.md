# Qwen zsRE SPHERE OOM 최소 수리 및 cold 재등록

지시 `USER-SH2-QWEN-ZSRE-SPHERE-OOM-RERUN-20261010-R1`, accepted turn `01a122f4-e240-7532-91fd-5295daaa902a`.

실제 새 job **62534 / s2-qwen25-zsre-sphere-oom-r1**을 held 등록·owner/fullargv/source/input/resource/dependency 검사 후 release했다. 2026-10-10 08:33:20 KST 초기 상태 PENDING, `afterany:62532`. 앞선 FE author CF62531→zsRE62532 뒤 직렬이므로 기존 cap4/DAG를 늘리지 않는다. GPU1/CPU6/59392MiB/48h, 기존 job 변경·취소0, 추가 GPU qualification0.

실행 source `9b70ecca6f445728542d7a55ab0014b01252ecc1`, official content tree `862667e41a2702f912aa17178aa9a249a0a84588582e0b074f3690987b7c0c2e`. 원 SPHERE config SHA `dcc7018f54b1f63c8c295ebe5d82b32fb933856e0d0b94b8659c9238ef41a814`를 변경하지 않았다. FE author의 clamp1/steps35를 SPHERE에 적용하지 않았다. 첫2000/BS100×20/seed0/FP32/eager/TF32off, 원 C0/P/모델/native history/투영 계수와 순서를 유지한다.

## 최초 입증 오류와 수정

62087은 source `7b5097aa`의 Qwen zsRE SPHERE이며 Slurm FAILED1:0. child `logs/qwen25-zsre-sphere-62087.out`에서 B10 L4..8 delta 계산 후 두 번째 `sparse_projection`의 `torch.linalg.eigh(C)` CUDA OOM이 확인됐다. 요청5.36GiB/free4.64GiB/process42.76GiB. 평가 지표 오류나 timeout이 아니다.

이전 layer의 `P_soft`와 `U`가 다음 함수 RHS 실행까지 살아 있었고, U view가 전체 eigenvector storage를 유지한다. 18944² FP32 한 행렬은1,435,500,544B이며 P_soft+eigvecs는2,871,001,088B다. 투영 결과를 실제 weight에 더한 직후 `del upd_matrix_proj, P_soft, U`만 추가했다. `sparse_projection` 수식/FP32 GPU eigh/eta/alpha/rank/연산 순서는 변경하지 않았다. target fit은 여전히 gradient-enabled이고 함수 전체 no_grad를 추가하지 않았다.

수리 commit `c54779f2`, SPHERE_main SHA `29ad88fe08c909242236405a66f011527112ec1710c5045775586c5bcb4f6692`. 원 upstream SHA도 SOURCES에 보존했다. CPU 실제 apply loop의 이전 buffer weakref 소멸, old negative control의 생존, 결과 bitwise 일치, 투영 비활성 branch, 원 투영식 비교 및 caller hparams/public-W0 검사를 통과했다(4 tests). FE 관련 검사를 합친 focused17/source166 SHA PASS. 실제 GPU OOM 해결 여부는 **NOT_YET_OBSERVED**.

## 보존 및 평가

원 B1..B9/partial raw/source/cost/W&B는 KEEP. B9 checkpoint `batch-09-7e44f382ba1f3bef.pt`, 8,535,442,009B, SHA `7e44f382ba1f3befcb9f4429f42fd328d55422306f9cef868a920c8bf7ec17b8`를 실제 fullSHA 확인했으며 load/resume/삭제하지 않았다. CF SPHERE62079는 정상 결과를 유지한다.

새 경로 `/mnt/raid5/janghj/ODE-edit/local/qwen-zsre-sphere-oom-rerun-20261010/registration-r1/runs/qwen25-zsre-sphere/`. 신규 독립 W0에서 시작하며 저장된 B9는 복원하지 않는다. 검증된 native context `5c01bc1a...`의 producer/model/tokenizer/source/runtime/seed와 실제 token을 공통 reader로 확인했다. W0 및 본실험 모두 최신 `official.evaluation.zsre_paper` public-query E/G/loc_ans request-macro를 사용한다. W0 receipt는 공개-query 관측으로 명시하며 옛 prediction agreement reference로 위장하지 않는다. Flu/Con 없음.

W0부터 committed batch마다 latest checkpoint, W20 보관. 최종 파일은 `checkpoint/latest.json` batch20/final_W20=true 및 immutable content SHA로 확인한다. 원 B9 외 추가 payload 예산은 FE 두 final + SPHERE latest/atomic temporary 총4개로 산정했다. 기존 성장48GiB/관측8GiB/여유32GiB 포함 요구128,899,350,528B, 준비 시 available154,249,408,512B. 임의 삭제나 256GiB gate 없음. 최종 consumer/보존 검증 전 KEEP, 자동 archive/delete 없음.

W&B authenticated project read 확인은 완료됐지만 새 run은 dependency 대기로 startup/online metric/W20는 아직 미관측이다. 과거 online history를 수정하지 않는다. 별도 GPU smoke/모델 forward 검산은 하지 않았고 sealed main만 자동 진행한다.

정확 receipt: `local/qwen-zsre-sphere-oom-rerun-20261010/submission-complete.json`; compact `audits/servers/server2/qwen-zsre-sphere-oom-rerun-20261010/{receipt,table-row}.json`. GH가 README 통합하며 SH2는 직접 편집하지 않았다. NO_BROADCAST_NOT_REQUIRED, raw/CP/model local KEEP. 장기 monitor/자동 retry 없음.
