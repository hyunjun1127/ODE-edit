# FE-MEMIT sequential10k — 제출 인계

사용자 최신 지시 “10k까지 진행하는 걸로 하자.”를 제출 전에 반영했다. 기존 `fe-sequential-2k` 승인 branch/local/report namespace는 유지하며 실제 profile/job명은 **fe-sequential-10k**다. 원 2k contract/CPU7/입력 준비는 보존했다.

## 실제 등록 상태

| 구분 | Job ID | 자원 | release 직후 단일 관측 |
|---|---:|---|---|
| persistent GPU runner | 59878 | 1GPU/6CPU/59392M/48h | PENDING, reason None |
| afterany CPU collector | 59879 | 0GPU/6CPU/24576M/4h | PENDING, reason None |

collector dependency는 `afterany:59878`. 두 job의 owner/argv/source/script/메모리/시간/dependency를 held 상태에서 확인하고 collector→runner 순서로 release했다. Admission시 현재 owner의 기존 Server2 GPU jobs는0, project cap2/taskcap1이었다. 다른 job변경0. release 이후 추가 scheduler/result polling을 하지 않았다. **실제 Llama startup·기술 PASS·완료/최종 metric은 NOT_OBSERVED**다.

## 범위와 결속

- cold W0/H0, fixed10k 전부, BS100×100, L4–L8. W0 target10,000fit + absolute-z4 canonical replay10,000회 후 own W/H chain.
- 35loss/34Adam 상한·원 FE objective/15000C0·no divisor·prewrite FP64Gram→CPUFP32H·FP64 W+delta→FP32 destination 그대로. CPU adapter는 이전 source와 scalar callback 두 줄 외 byte-exact.
- 100commit/500solves/500history/99joins. B101/추가 baseline/sweep 없음. max350000loss/340000Adam은 계획이다.
- W0 전10k, current pre/post, W5/W10/…/W100 all-seen. W100 R10000/P20000/N100000. first100/500, birth/active/superseded cohort는 같은 원 row에서 CPU 계산한다.
- 계획 저장1,729,000rows/신규관측1,727,700rows(B1 pre1300 W0재사용), unique requests10000. 반복관측을 독립 samples로 세지 않는다.
- 새 edited weight/H/target/RNG/resume tensor 영속저장0. Target table819200000B CPU RAM. exact_resume=NOT_AVAILABLE.

## W&B와 검사

[CPU online smoke](https://wandb.ai/wkdguswns2256/layer%20allocation/runs/d2b83c62f25547fd): **READY_ONLINE_VERIFIED**, remote3points/dropped0. 과학 run이 아닌 setup run이다. FE scientific run ID/URL은 실행시 task-local tracking receipt에 기록하며 지금은 미관측.

SH1 공통 logger init/log/finish를 재사용하고 FE는 기존 scalar만 whitelist 매핑한다. 키/raw/code/console/artifact 업로드0, scienceenv pin변경0. CPU7 재사용 + 좁은 horizon/100batch 순서/telemetry/resource/noCP 검사4/4 PASS. 독립 reviewer0, owner audit; 실제 GPU PASS와 구분한다.

## 실행 재현·provenance

- 실행 source `3645b40832b3da08d9c3ee3264b3d500fbbd6434` (이후 보고 publication과 구분).
- upstream FE `478134dfb24b43f4e18b47e8500893ce3f9cc50f`; read-only local source. 원 논문 bulk2000의 직접 재현이 아니라 동일 FE의 sequential10k profile이다.
- lock `/mnt/raid5/janghj/ODE-edit/local/fe-sequential-2k/attempt-online-10k-r1/execution.lock.json`, SHA256 `91d6113fca6ae76c92bc4b03abb8465937ff0bb58b0c164661e4d37ce3a3e937`.
- config/input `local/fe-sequential-2k/preparation-online-10k-r1/`; token10000/observer identity130000, 전체순서 loader 검산. 기존 17asset fullSHA receipt+현재stat 재사용.
- 결과/root `/mnt/raid5/janghj/ODE-edit/local/fe-sequential-2k/attempt-online-10k-r1/`; `main/`, `collector/`, `tracking/`는 등록된 프로그램이 기록한다.
- 등록 명령: `python -m project.run_scripts.fe_baseline.submit --config /mnt/raid5/janghj/ODE-edit/local/fe-sequential-2k/preparation-online-10k-r1/configuration.json --attempt-name attempt-online-10k-r1` (create-once; 재등록 지시가 아님).
- tracked audit `audits/servers/server2/fe-sequential-2k/submission-online-10k-r1.json`; 원 자세한 held/source/input 기록은 local 보존.

## 자원·한계

준비시 실제 node8×RTX A6000/49140MiB를 조회했다. 1GPU 48h는 요청 상한이며 **10k 완료 ETA가 아니다**. 원 2k 산정에서 이어진 config `host_plan_GiB.steady_C0_H_target=7.813` 요약은 이전 추정값이며, 정확 bytes 기준 H+C0+새 target의 합은 약8.42GiB다. 정확 target819200000B와 59392M request는 별도 결속되어 있다. 실행 lock은 소급 수정하지 않는다. 모델/행렬 peak와 첫 fit속도/전체시간은 runner 실측 전이다. timeout/실패 시 원 자료와 비용을 보존하나 noCP exact-resume/자동재시도는 없다.

공개 BF16/TF4.51.3/generated-context와 달리 이 profile은 FP32/eager/TF32off/기존 canonical context/실제 pinned runtime를 쓴다. TF strict와 논문 자유생성 Accuracy를 혼동하지 않는다. 실제 성능·보존·비용/완료 주장은 아직 없다.

Source/compact report만 Git 공유. raw/model/teacher/전체stdout/credential Git0, NO_BROADCAST_NOT_REQUIRED. sealed runner/collector는 자연진행하며 agent 장기모니터링 없이 사용자 recall을 기다린다.
