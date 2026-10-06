# GPT2-XL PRICE 6cell 제출 인계

Nonce: `USER-GH-PRICE-MODEL-RUNS-TRACKING-20261007-SERVER1`.
상태: **GPU 6개 및 CPU collector를 held 검사 후 release**. 실험 완료가 아니다.

| 모델 | Writer | Arm | Job | Dependency |
|---|---|---|---:|---|
| GPT2-XL | MEMIT | CAP075 | 60094 | 없음 |
| GPT2-XL | MEMIT | CAP100 | 60095 | afterany:60094 |
| GPT2-XL | MEMIT | FREE100 | 60096 | afterany:60095 |
| GPT2-XL | AlphaEdit | CAP075 | 60097 | afterany:60094 |
| GPT2-XL | AlphaEdit | CAP100 | 60098 | afterany:60097 |
| GPT2-XL | AlphaEdit | FREE100 | 60099 | afterany:60098 |
| — | CPU collector | 6cell | 60100 | afterany:60094:60095:60096:60097:60098:60099 |

각 cell은 독립 cold W0/H0, 같은 first2000, BS100×20이다. 모두 ours이고 새 stock baseline 0이다.
server4 Llama/GPTJ는 이번 SH1 처리 범위가 아니며 전체 18cell 완료를 주장하지 않는다.
동일 task의 기존 제출은 없었다. old pending 취소 0, RUNNING/타 task 변경 0.

## Source·자산·권한

- 실행 source `537c42ad26753ba6498f9401904d94af1effdf91`, tree `a712f82c0e5086c07c00ea49ef9c8825dc3fd494`.
- config SHA `98a2112a7fc46eb05c54d079ba5cd390cb36c39d7e60cbb01eeb2dfb614af587`.
- lock SHA `c3e46222af01e0a9ba5d5881d5ad7ca072fa55917ca5f04931fde47cbf7bfcfe`.
- archive SHA `868b4ddd936b55a1c1a5f4cf51032c0b08e16698707bf12e5764c1f6e1991a39`, 3,194,880 bytes, source 297 members.
- Local receipt/output root: `/mnt/raid5/janghj/ODE-edit/local/jlz-price-gpt2xl-2k/attempt-execution-r1/`.
- 전달 SHA `01b5446c…`는 authority8082da5의 실제 bytes와 일치했다. 최신 main의 envelope는 명시 server4 job60001 KEEP 보강으로 `f978c411…`이며 이 bytes도 별도 결속했다. SH1 GPT2 범위는 동일하다.
- 역사적 method-only source85a09238과 receipt는 보존했다. 새 명시 recall만 해당 실행 차단을 대체했다.
- app root `/mnt/raid5/janghj/ODE-edit` 및 실제 SH1 session/host/origin을 확인했다. root 경계 helper의 과거 mismatch는 공유 설정을 바꾸지 않았고 전용 WT ignored boundary에서 PASS했다. root dirty 보존.
- Existing GPT2-XL revision15ea56dee5df4983c59b2538573817e1667135e2의 safetensors만 사용한다. model/C0/P/READY SHA·size 결속; 재계산/복제/기존 cache 덮어쓰기 0.
- 공통 context는 freeze 때 아직 없다. 최초 실제 MEMIT_CAP075에서만 native 생성·RNG restore·atomic READY 후 나머지 job이 검증한다. Alpha 첫 job의 afterany60094는 이를 위한 입력 경계다. file polling 없음.

## CPU·독립 source 검토

50 tests: 48 통과, 2 skip, 실패 0. 실제 모델 forward/backward 및 online logging 검증은 아니다.
두 bounded 독립 source reviewer가 runtime과 submit/collector를 분리 검토했다.

구체적 기술 보완은 세 가지다. 원 safetensors의 base `h.*` key를 정확히 결속했고,
GPT2Block의 positional attention mask/cache position을 보존했으며,
input READY 이전 실패의 collector counter를 미기록(null)로 보고하도록 보완했다.
수식·threshold·계수·입력·표본 변경은 없다.

GPT2 Conv1D `[6400,1600]` native addmm/bias, serial MLP, L13–17/anchor17/readout47,
final LN 1회, FP32 model/activation와 FP64 geometry를 유지한다.
실제 native YAML parser로 lr.5, 평가20/갱신19/terminal19, MEMIT20000,
Alpha L2=10/threshold.02를 결속했다. Alpha YAML20000은 미사용이며 C0 hybrid 0이다.
실제 B1의 native/staged parity·normal equation/LOO·terminal copy·Honce·B2 join은
봉인 runner 안에서 수행하며 현재 PASS로 보고하지 않는다.

재현 CPU 명령:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /mnt/raid5/janghj/EasyEdit/.venv/bin/python -m unittest \
  project.run_scripts.jlz_price_gpt2xl.test_contract \
  project.run_scripts.experiment_tracking.test_method \
  project.run_scripts.experiment_tracking.test_job_identity \
  project.run_scripts.experiment_tracking.test_tracking
```

## 자원·Tracking·관측 경계

project cap2, 외부 same-owner devbox GPU resource DAG 점유0을 제출 전·release 전에 확인했다.
MEMIT와 Alpha 두 lane의 최대 동시 GPU는2다. 각 GPU1/CPU8/64GiB/48h,
devbox/gpu/lab_gpu_s1/exportNONE/Requeue0. CPU collector GPU0/CPU8/24GiB/2h.
전체 owner/fullargv/source/script bytes/resources/dependency를 held에서 검사했다.
48h는 요청 wall이며 ETA·GPU-hour hard budget이 아니다.

계획 host peak26.88GiB/GPU peak21.12GiB는 추정값이다. 실측치가 아니다.
scalar/row/source/spool 전체 reserve15,160,049,664 bytes를 결속했고 디스크·inode 여유를 확인했다.
공유 파일시스템의 미래 여유를 보장하지 않으며 각 batch의 fresh guard를 유지한다.

W&B helper930e4653 계열의 엄격 scalar/identity/axes를 재사용한다. 중앙 기능표식 이관을 기다리는 제출 gate는 없다.
current/pre·current/post는 항상 현재100, all_seen/post는 실제 W5/10/15/20만,
W0_first2000 및 w0/current/N·w0/all_seen/N을 구분한다. R/P/N 9fields, pct/nats,
N desired=true, harmonic/missing semantics, edits/pre/post state와 monotonic20-slot fit axis를 유지한다.
실제 job ID가 run.name/config에 결속되고 immutable UUID/URL receipt와 mutable transport를 분리한다.
startup/finish bounded readback은 각 실제 job에 등록됐다. **run ID/URL·remote delivery는 NOT_OBSERVED**이며 CPU/fake SDK PASS로 대체하지 않는다.

제출 직후 한 번의 snapshot은 7개 모두 PENDING/reasonNone이었다. 이는 resource 부족 확정 근거가 아니다.
B1·W20·실측 allocation cost·실제 online readback은 INITIAL_NOT_OBSERVED/NOT_OBSERVED.
최신 지시대로 제출 snapshot 후 agent monitoring을 중지한다. runner/collector는 자연 진행한다.
recurring poll/heartbeat/자동 retry/후속 scientific submit 0.

NoCP, exact_resume=NOT_AVAILABLE. model/H/P/K/activation durable dump 0.
기존 raw/source/log는 local KEEP, 새 raw 복제 없이 NO_BROADCAST_NOT_REQUIRED.
Git에는 source·compact CSV/receipt만 게시하며 prompt/tensor/credential/fullstdout은 제외한다.

근거: [cell ledger](cell-ledger.csv), [submission receipt](../../../../audits/servers/server1/price-model-runs-tracking/submission-receipt.json), [pre-submit review](../../../../audits/servers/server1/price-model-runs-tracking/pre-submit-review.json).
