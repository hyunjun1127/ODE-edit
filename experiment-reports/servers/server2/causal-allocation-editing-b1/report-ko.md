# Causal Allocation Editing B1 — Slurm 제출 거부

권한/nonce: `USER-GH-SH2-CAUSAL-ALLOCATION-EDITING-B1`.
정본은 `messages/head/causal-allocation-editing-b1.json`과
`plans/global/causal-allocation-editing/server2-b1.json`이다.

## SUBMISSION_HANDOFF — RESOURCE_POLICY_CONFLICT

실행 source `4321f20d`를 전용 branch에 push하고 immutable `attempt/` source/archive/config/lock을 만든 뒤 정식 `sbatch --hold`를 호출했다. Scheduler가 다음 사유로 거부했다.

```text
sbatch: error: CPU/RAM policy: requested 8 CPUs exceeds 6 CPUs on server2 for 1 GPU(s)
sbatch: error: Batch job submission failed: Unspecified error
```

GPU job ID 없음, CPU collector ID 없음, release 없음, 새 GPU allocation/qualification/main/commit/H append 모두 0이다.
거부 직후 해당 exact job-name의 Server2 own queue가 비어 있음을 확인했다.
CPU8은 이번 명시 resource 계약이므로 6으로 임의 변경하거나 2GPU로 우회하지 않았다.
진행에는 **1GPU/6CPU로 변경하는 명시적 지시 또는 scheduler의 합법적 8CPU 허용**이 필요하다.
이번 source/input/실패 증거는 보존하며 자동 재시도·monitoring·다른 task 변경은 없다.

요청은 GPU1/CPU8/59392M/8h/exportNONE/no-requeue/server2/lab_gpu_s2였다.
첫 job 등록이 실패해 의존 CPU collector는 등록하지 않았다. 제출 실패와 실험 기술·과학 결과를 혼동하지 않는다.

- SH2/server2/session `01a0493a-074c-7f91-9a13-769116326fef`에서 전용 clean branch로 준비했다. 기존 root의 변경 1,390건은 보존했다.
- 기존 production `a1332fd70f0d4898b74399a349e41a225f0baf04`의 engine/solver/calibration/geometry/qualification/native reference/entry는 byte-exact 재사용한다. 변경은 명시적 horizon profile, Server2 준비/제출, collector 분모에 한정한다. 기본 production2k의 20batch 계약은 유지한다.
- 최초 CPU 회귀 실패는 `number<=20` 문자열 검사였다. 해당 검사를 profile horizon/default20 검사로 수정했고 원 실패 receipt/log를 보존했다. 최종 CPU 68개 통과(기존63+신규B1 5개)는 실제 GPU PASS가 아니다.
- 첫100, cold W0/H0, BS100×1, 1commit/5H/0 interbatch join, noB2/noCP. W0/W1만 R100/P200/N1000. 정확한 로컬 W0 bridge가 없어 first100 새 평가를 사용하며 full2000 W0를 실행하지 않는다.
- 모든 native context/논리B/층/solver budget/정밀도/수식은 그대로다. Positive finite 가격 정의 실패는 typed failure로 끝내고 임의 가격·추가 fit을 만들지 않는다.
- 실제 모델·C0 등 17개 로컬 asset을 원 production SHA/size와 대조했다. 원 context/native/import/토큰 identity 등 선택 79개 소형 파일은 exact allowlist로 수신했다. 대형 모델/CP/C0 전송·재생성은 없다. 원본 KEEP.
- runtime: Torch2.9.1+cu128/Transformers4.57.1, FP32/eager, geometryFP64, TF32 두 flag false. 원 production runtime source와 SHA를 대조했다. native 실제 CPU import/hparams 확인 완료; CUDA/model qualification 미실행.
- 첫100 native pack: 700 rows, valid10651/owner-padded12859 tokens, 최대 owner width32. 모델 load host 추정40GiB, 실행 host 추정37.728GiB. GPU 실제 관측 RTX A6000 49140MiB; memory peak는 아직 미측정이다. 저장 reserve12GiB, 새 checkpoint 없음/exact resume 불가.
- GPU runner 하나에서 qualification2→READY→같은 cold state의 B1을 순서대로 수행한다. CPU afterany collector를 함께 등록한다. GPU1/CPU8/59392M(상한60416M)/8h, collectorGPU0/CPU8/24576M/4h, exportNONE/Requeue0. 8h는 ETA가 아닌 요청 상한이다. 현재 project cap2/taskcap1을 적용한다.
- 다른 task STOP 및 SH4 59163/59164/59165는 조회·변경하지 않았다. 제출 후 GPU 완료를 기다리지 않으며 새 polling/heartbeat/retry/resume을 만들지 않는다.

## 경로와 재현

Local root: `/mnt/raid5/janghj/ODE-edit/local/causal-allocation-editing-b1/`.
`preparation/cpu.json`은 최초 실패, `preparation/cpu-ready.json`은 최종 회귀,
`preparation/input-runtime-binding.json`과 `inputs/transfer-receipt.json`은 입력 결속이다.
등록 후 `attempt/execution.lock.json`, `held-inspection.json`, `submission.json`이 실제 source/job을 결속한다.
실행 결과는 `attempt/main/`, 실제 qualification은 `attempt/qualification/`,
CPU 사실 보고는 `attempt/collector/report-ko.md`에 생성한다.

```bash
python -B -m project.run_scripts.causal_allocation_editing.b1_prepare --cpu-preflight <새 CPU receipt>
python -B -m project.run_scripts.causal_allocation_editing.b1_submit --config <configuration.json>
```

Create-once 경로와 중복 제출 검사가 있으므로 동일 실행을 재제출하는 명령이 아니다.
원 raw/teacher/model/tensor/stdout은 local-only. `NO_BROADCAST_NOT_REQUIRED`.
현재 문서는 제출 거부 인계이며 W1 완료/양수가격/GPU qualification 성공을 주장하지 않는다.
