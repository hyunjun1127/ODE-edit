# Server2 baseline-only STOP 및 CAKE/BLUE 실패 진단

Nonce: `USER-GH-SH2-BASELINES-STOP-FAILURE-REVIEW-20261007`

상태: **STOPPED_USER**. baseline 실행 권한 중지, 수리·재제출 없음.

관측: 2026-10-07T11:24:04.362292+00:00 (UTC; scheduler 시작/종료 표기는 서버2 KST).

## 중단 결과

collector **60773 → PRUNE60771 → RECT60772** 순서로 exact `scancel`을 실행했고 모두 exit0, CANCELLED를 확인했다. collector는 PENDING/Dependency, 두 후속 arm은 이미 RUNNING이었다. CAKE/BLUE의 FAILED가 `afterany` 후속을 해제했으므로 부모 실패만으로 후속이 중단되지 않았다.

취소 전 janghj(1025)/server2/Command/task/source/lock 및 launcher SHA를 대조했다. 후보10개의 post queue는 비어 있고 active baseline 대상은 **0개**다. 한정 owner metadata inventory에서도 server2 또는 node 불명 job은 0개였다. 이는 서버 전체·다른 사용자 GPU 할당이 0이라는 주장이 아니다.

| Job | 등록 task/arm | 실제 최종 scheduler 상태 | 이번 처리 | allocated GPU-sec |
|---|---|---|---|---:|
| 59878 | fe-sequential-10k | CANCELLED | 기존 terminal, 취소 호출 없음 | 66722 |
| 59879 | fe-sequential-10k | CANCELLED | 기존 terminal, 취소 호출 없음 | 0 |
| 60656 | BASE_MEMIT | COMPLETED | 기존 terminal, 취소 호출 없음 | 10412 |
| 60657 | BASE_ALPHAEDIT | COMPLETED | 기존 terminal, 취소 호출 없음 | 12318 |
| 60658 | collector | COMPLETED | 기존 terminal, 취소 호출 없음 | 0 |
| 60769 | CAKE | FAILED | 기존 terminal, 취소 호출 없음 | 61 |
| 60770 | ALPHAEDIT_BLUE | FAILED | 기존 terminal, 취소 호출 없음 | 61 |
| 60771 | PRUNE | CANCELLED | 이번 exact 취소 | 293 |
| 60772 | RECT | CANCELLED | 이번 exact 취소 | 293 |
| 60773 | collector | CANCELLED | 이번 exact 취소 | 0 |

stock MEMIT/AlphaEdit 및 collector의 scheduler COMPLETED는 확인했으나 과학 결과는 이번 진단에서 NOT_REVIEWED다. FE 두 job은 16:10:10 KST에 이미 CANCELLED였으므로 다시 취소하지 않았다. W060652/60653/60147/60148, ours, 타 서버·타 사용자·불명 task는 변경하지 않았다. user 전체취소·wildcard는 사용하지 않았다.

## 최초 오류와 source 근거

CAKE60769와 ALPHAEDIT_BLUE60770의 첫 traceback은 동일하다.

```text
RuntimeError: NATIVE_GPTJ_LINEAR_FC_OUT_LAYOUT
run.py:103 -> native_cake_blue.py:704 -> require:56
```

실패 stage는 LOAD_W0(모델 load 뒤 native adapter binding 포함)다. 봉인 adapter는 `fc_out`이 Linear/FP32/[4096,16384]이며 **bias=None**이라고 요구한다. 그러나 해당 pinned Transformers4.57.1 GPTJMLP constructor line415는 `nn.Linear(intermediate_size, self.embed_dim)`를 만들고 Torch2.9.1 Linear는 기본 `bias=True`(line100, Parameter 생성 line112)다. 따라서 **stock GPT-J의 bias-present 구조와 task-owned guard의 biasless 가정이 충돌**한다. native fit/solve 품질이나 OOM의 증거는 아니며 public native apply 이전 오류다.

실제 GPU 객체의 type/bias/shape/dtype 개별 dump는 없다. 관측된 exception은 결합 guard가 실패했다는 직접 증거이고, bias 조건의 비호환성은 봉인 runtime/source로 설명한 RCA다. `runtime.json` 두 개는 NOT_RECORDED다.

두 arm 모두 완료 batch/commit0, commit receipt 없음이다. terminal의 native count0은 engine 생성 전 fallback이므로 성공한 native 작업 계측으로 쓰지 않는다. `rollback_verified=false`는 transaction이 아직 없어 rollback 미실행인 경로이며 복원 실패로 해석하지 않는다. PRUNE/RECT는 실행 중 사용자 중단된 partial로 보존하며 과학 완료 여부를 새로 추정하지 않는다.

## Identity 및 보존

- 실행 source: `3a4a107b7ae4c66f2f7a7f0cb441d26c5a639f68`; tree `60f2be99325236e8f197f51fdd57c3d0e8760579`.
- lock SHA256: `e08af8a5febe136b727eb8e8c0e01a0dfb6afecad9c23aaccaaf00a959e0d87d`.
- config file SHA256: `4cddaf03c88a7364e2622d1f8e3c7e0dd0d796cc545e73177f6f9951dd933790`.
- terminal config canonical digest: `172d7f204de97ed64ab75bde9e2338bf9532fc55164cd966eb08ef9b8017a3e1`. 파일 bytes SHA와 구분한다.
- adapter SHA256: `31246adca7e0b5510237f6da763c565da9d2a0fd68a6b4e382923ca6255f61f8`.
- modeling_gptj.py SHA256: `2fea54b39fba964e2c7516bbd66e20eff3296c06fe540405d0020dd8e5b8264b`.
- torch Linear source SHA256: `63fb3cae2a1561fec5e183e45c171d177080113a3996f1046eea35530cd33247`.

원 root `/mnt/raid5/janghj/ODE-edit/local/gptj-cake-blue-prune-rect-2k/attempt-r1/`의 failure/terminal/.err/.out/source/archive/config 및 모든 기존 CP/raw는 LOCAL_KEEP다. exact member path/size/SHA는 audit manifest에 기록했다. fullstdout/prompt/tensor/credential은 Git에 넣지 않았다. 과학 source/threshold/math/환경/다른 task 수정0, rescue checkpoint0, model/tensor load0, 새 GPU/Slurm0이다.

## 비용 및 tracking 경계

신규4arm의 parent job만 계산하면 CAKE/BLUE 각61초, PRUNE/RECT 각293초, 합 **708 allocated GPU-sec = 0.196667 GPUh**다. 실패 부모122초와 사용자중단 후속586초를 분리했다. .batch/.extern 중복 계상0; CPU collector GPU0이다. FE66722초와 stock MEMIT10412/AlphaEdit12318초는 과거 실행 비용이며 새 진단 비용에 재가산하지 않는다.

| 실패 arm | program seconds | peak host RSS bytes | peak GPU allocated bytes | peak GPU reserved bytes |
|---|---:|---:|---:|---:|
| CAKE | 54.395220 | 48570257408 | 24204579840 | 24205328384 |
| ALPHAEDIT_BLUE | 54.190668 | 48572526592 | 24204579840 | 24205328384 |

저장된 W&B receipt는 CAKE [2fbeae86b9804af8](https://wandb.ai/wkdguswns2256/layer%20allocation/runs/2fbeae86b9804af8), BLUE [bff481b2e9574927](https://wandb.ai/wkdguswns2256/layer%20allocation/runs/bff481b2e9574927)의 startup READY_ONLINE/finish FINISHED_SDK_FLUSHED다. 이번 새 remote readback/로그인 smoke/백필은 하지 않았으며 SDK flush를 remote delivery 또는 과학완료로 인증하지 않는다.

## 검산 및 종료

owner cancellation audit + bounded read-only source explorer를 사용했다. 독립 red/GPU 검증은 수행하지 않았다. 최종 CPU 검산은 표 산술·10개 ID/manifest/source/count 및 raw-free 범위만 확인한다. 상세 receipt는 `/mnt/raid5/janghj/ODE-edit/local/server2-baselines-stop-failure-review/receipt.json`(SHA256 `5a6f84a26824fea35040c42c1a74e9230c4fd395b2e6a11b09739cc2b8b5c240`)이다.

현재 범위는 cancellation/읽기전용 RCA 완료다. 수리·재실행·자동재개·반복 monitoring은 **NOT_RUN/권한 없음**이며 새 사용자 지시 전 baseline STOP을 유지한다. `NO_BROADCAST_NOT_REQUIRED`.
