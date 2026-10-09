# MEMIT_FE_HISTORY: GPT-J / Llama / Qwen CF 2K

USER `USER-GH-SH1-MEMIT-FE-HISTORY-THREE-MODEL-2K-20261009-R1`의 별도 variant다.
stock MEMIT_FE 결과로 relabel하지 않는다. 공통 Loc 변경 main3252c776/implementation3a400ae5를 포함한
준비 base main71aebbc5를 채택했다. 기존 frozen job/CP/raw는 변경하지 않았다.

## 실제 등록/release

execution source `eaf78c33` (implementation `115293d3`), 최소 CPU48 tests PASS/source157 SHA PASS.
GPT-J **61927**, Llama **61928**, Qwen **61929**: 전량 held owner/script/fullargv/resource/dependency
검산 후 release. 초기 세 job은 PENDING, 의존성 없음. 기존 FE61773 KEEP 및 합산 DAG 폭4/cap4.
W20/actual GPU success/온라인 remote readback은 아직 미관측이며 별도 qualification은 USER_DISABLED.
정확 source/config/lock/jobname/resources는 `audits/servers/server1/memit-fe-history-three-model-2k/submission.json`.

## 구현

- `official/baselines/memit_fe_history.py`: native FE apply/compute_z/compute_ks/context를 재사용.
- stock FE main의 optional `history_entry=None` seam은 기본 분기 수학을 유지한다.
  새 variant만 FP64 `solve(lambda_C*C0 + H_entry + K*K.T, K)`를 사용한다.
- 각 model의 기존 FE JSON/fit/loss/layerwise z/undivided residual/FP64 solver→FP32 write 유지.
  registry는 새 variant만 기존 FE hparams에 명시 결속한다. 가격/Alpha projected solve/replay는 없다.
- 마지막 층까지 native write 성공 후 own final model의 native mean keys를 각 층 한 번 관측하고,
  CPU FP32 H에 `K_final @ K_final.T`를 한 번 더한다. 실패 시 H 미진행, apply 오류 시 selected W 복원.
- B1 H0에서 실제 solve system이 기존 native system과 exact-equal인지 본실험 내부에서 확인한다.
  추가 solve/fit/forward qualification 없음. 전체 native GPU parity PASS라는 뜻은 아니다.
- 새 main runner는 각 모델 cold W0/H0, 기존 official ordered first2000/seed0/BS100×20.
  본실험 W0 및 W5/10/15/20 factual 평가 유지. CF generation DEFERRED, 중간 generation 없음.
- W0 및 매 batch factual 완료 후 최신 checkpoint 1개를 저장하고 W20 보존한다.
  H/context/cursor/RNG 복원을 구현했으며 별도 GPU resume 반복 검증은 USER_DISABLED다.
- 새 W&B method/writer/arm은 MEMIT_FE_HISTORY/memit_fe_history. 공통 official transport 사용.
  본실험 factual/commit 실시간 scalar만, 과거 history 덮어쓰기 없음.

## 자산/자원/검산 단계

세 모델의 기존 HF revision cache와 EasyEdit 각 층 C0를 read-only fullSHA/shape/count로 결속했다.
GPT-J는 기존 pytorch_model.bin, Llama/Qwen은 기존 safetensors를 그대로 읽는다. 다운로드/자산복제0.
H0 native solve 일치, additive H, 층별 final-key append-once, 실패 rollback, checkpoint roundtrip,
legacy history 거절, 새 method schema와 cap DAG를 작은 CPU fixture로 검산한다.
정확 CPU 결과와 실제 등록 ID는 후속 submission receipt로 구분하며, 현재 실제 GPU 결과는 미관측이다.

예정 1GPU/8CPU/98304MiB/48h(job ceiling, ETA 아님), server1 최신 직접 USER cap4와 stricter local/QoS 준수.
RUNNING/admitted pending 전체 DAG에 새3job을 포함하고 불필요 barrier 없이 빈 lane 사용.
held에서 실제 owner/source/full argv/resources/dependency/배치 script를 검산 후 release한다.
source와 config는 새 immutable attempt로 봉인하고 이전 job은 KEEP한다.

최종 checkpoint는 server1 local에 이미 있으므로 등록/보호 KEEP; 후속 FLUCON consumer pending.
대형 모델/raw/CP/Git/W&B 전송0. `NO_BROADCAST_NOT_REQUIRED`: same-host assets, compact Git source/report만 공유.
