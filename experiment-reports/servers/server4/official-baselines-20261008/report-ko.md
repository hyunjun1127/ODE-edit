# server4 official baseline 구현·취소 기록

## 최신 갱신: SH1 factual API 결속 (2026-10-09)

nonce `SH1-ALL-OFFICIAL-FACTUAL-PUBLISHED-20261009-R1` 및
`USER-GH-SH4-OFFICIAL-BASELINES-CONTINUE-20261009-R1` 입력을 채택했다.
main `55afa07d2555718c4ad8b2e483db8a260aa94e35`, factual source
`596896ff82a0c0aab8f64e4920f6f09d73072c58`, official tree
`e79c3939447d2fbbd536cb0449c9ed2da361fa2b`를 전용 WT에 병합했다.
아래의 최초 API 미게시/BLUE/Llama generation blocker 기록은 **역사적 상태**이며
공통 게시본 수신으로 source 수준에서 해소됐다. 실제 GPU parity 해소 주장은 아니다.

factual.py SHA `2bc41883b9261d084a3b4b99e0920911789659401a6d9b2b0f0d801a446ab47b`,
test SHA `6eababac9556f758ee4f31f8342500674081aa102515d464871458cdbed0c999` 일치.
게시 evaluate 및 build_zsre_w0_reference의 exact signature에 own caller를 연결했다.
공유 evaluator/tracking/algorithm bytes는 수정하지 않았다.

### 실제로 수행한 검산

- CPU 52 PASS: own caller9/wiring3, 공통 factual16/fakeSDK24. 최초 테스트 실행의
  own test harness가 잘못된 `metrics(..., official=True)`를 사용해 1건 실패했으며,
  게시 API의 `config_values`로 수정 후 52개 전부 재검산했다. 공통 구현/허용오차 변경0.
- source148 SHA 검증, Python209 AST, external-task imports0, diffcheck PASS.
- 실제 Llama tokenizer로 CF/zsRE 각 first2000 계획 검산 PASS, 모델 forward0.
  CF target candidate-sequence 52,000(두 target 합산), zsRE sequence6,000이다.
  이는 입력계획 count이며 W0 성적 또는 GPU 관측 건수가 아니다.
- current100은 milestone에서도100, all-seen은 실제 누적 endpoint만이다.
  official E/G/S/Score를 PRICE R/P/N pair count로 relabel하지 않는다.
  zsRE W0 `evaluation` 재사용/row 순서/외부 identity, loc_ans와 W0 agreement를 구분한다.
- 런처의 attempt/qualification 전달을 연결했다. 동일 shared logger가 실제 Slurm
  identity와 immutable run receipt를 맡는다. 이번 CPU 작업에서 online run 생성0.

새 자산 manifest와 6개 config:
`/data/janghj/ODE-edit/local/official-baselines-20261008/preparation-v2/`.
v1/원본 자산/이전 source와 raw를 보존했다. runtime/SDK 경로만 결속하며 환경 pin 변경0.

### 미완료 및 GH 공유 입력 문의

CF 모델 공통 W0 factual+generation 및 zsRE W0 prediction receipt/담당이 아직 미결속이다.
사용자 요청으로 GH 직접 문의를 시도했으나 앱 메시지 도구 미지원, SSH alias 해석 및
기존 인증 제약으로 **전달되지 않았다**. 호스트 키 검증을 끄거나 credential을 복제하지 않았다.
문의 원문과 실패 receipt는 ignored local에 보존한다. zsRE 기존 자산이 있다고 추정하지 않는다.

native CF parity, 실제 연속B3 대 B2-resume, zsRE GPU smoke, production READY/freeze와
6개 신규 제출은 **NOT_RUN / NOT_SUBMITTED**. 기존 Qwen/OURS 및 scheduler job 변경0.
이번 작업의 신규 job IDs는 `[]`이다. main 배포용 READY가 아닌 검토용 own branch 게시다.

자원은 현재 canonical server4=2와 local3 중 더 엄격한2를 admission 상한으로 판단한다.
이번 turn에는 scheduler 재조회/변경을 하지 않았다. 과거 running snapshot을 현재 상태로
주장하지 않는다. disk available 관측60,700,098,560bytes는 예약/메모리 PASS가 아니다.

검토 수준: SH4 owner CPU/source audit, 별도 independent reviewer/GPU/online PASS 없음.
NO_BROADCAST_NOT_REQUIRED: 같은 host의 기존 자산과 Git 소형 source/receipt만 사용.

---

아래는 최초 구현/취소 시점의 보존 기록이다.

권한: `USER-OFFICIAL-BASELINES-20261008-R1` 및 후속 사용자
“baseline run들은 일단 전부 취소하고 task 이어서 진행해”.

## 기존 baseline 취소

server4 소유 native baseline MEMIT/PRUNE/RECT/AlphaEdit/BLUE/CAKE
**60917–60922 및 CPU collector60923 모두 CANCELLED by1025**를 확인했다.
취소 전 각 job은 PENDING/JobHeldUser, runtime0, allocation없음이었다.
원 submission의 owner/Command/ReqNodeList/source와 결속했으며 collector와 후속부터 취소했다.
기존 source/lock/raw 삭제0, 다른 서버 job 변경0, 신규 baseline 제출0.

## Qwen 보존

Qwen **61598 RUNNING**이며 원 frozen source `9ecdf8342c52ee7e6a85a6ed741fc4f5468bc1ef`를
변경하지 않았다. [W&B run](https://wandb.ai/wkdguswns2256/layer%20allocation/runs/02b818498de04be2)
startup companion receipt는 `REMOTE_IDENTITY_VERIFIED`다. 아직 batch/edits 확인은 별도다.
Llama OURS61418과 PRICE collector61421도 변경하지 않았다.

## 새 official 구현

기준 main `537509729345c26efa641ac9f026993e5f5c036e`, official tree
`c750f75d14b722233810cc790cdc128d0edb965c`.
후속 main `16d999ae1de3dac846a562199127a0db791293d9`의 dispatch receipt도 fast-forward했다.
전용 branch `codex/server4-official-baselines-20261008`, dirty root 보존.
구현 commit `bc4ba7ca7c69dc42cd1022e5cc1ff8fa52202bfd` (GPU 실행 source가 아닌 검토용).

`official/runners/server4/`에 실제 native registry/state/runner, asset binder,
held-submit 및 resume 비교 연결을 작성했다. 담당은 Llama3 AE/BLUE/SPHERE × CF/zsRE6행.
방법 수치 공통 source는 수정하지 않았다. 현 runner는 아래 blockers가 해결되기 전 실행을 거부한다.

자산 manifest:
`/data/janghj/ODE-edit/local/official-baselines-20261008/preparation-v1/assets.json`.
28개 member, CF·zsRE 각각 first2000 normalized stream SHA가 official lock과 일치,
6개 config 생성. 기존 model/C0/P 자산만 사용했다.
L4–8 C0는 각 FP32 14336×14336, native mom2 count66,019,200이다.
token count와 Wikipedia 표본수를 혼동하지 않는다. P는 prior검증 `[5,14336,14336]`,
물리층 `[4,5,6,7,8]`, cutoff.02, SHA
`6d356468c6408dca694c1907ffde31cddb99f7e67fa2e74502910d5afbede5ec`를 현재 stat과 결속했다.
BLUE는 물리L4/L8 선택이며 index0/1로 잘못 사용하는 경로가 아니다.

## 검증과 blocker

- CPU wiring3개 PASS: checkpoint restore/cache/context, resource argv 한계, tensor exact hash.
- official source verifier134개 frozen member PASS, external task import0.
- `git apply --check --unidiff-zero` 검토용 BLUE patch PASS. shared file 적용0.
- S1 source `adb244e6...`의 native_generator와 official 파일 SHA가
  `ef3d3daf20c174cfa3627a3852e2d593eb506718ac8ddde62a909a26c87bd2c3`로 동일.
- native GPU smoke, CF parity, 실제 B2→B3 resume **NOT_RUN**.
- factual.py/API 부재: SH1 게시 후 결속 필요. 직접 메시지 tool은 사용 불가 응답으로
  전달되지 않았으며 사용자에게 API 인계를 요청했다.
- native_generator.py는 gpt2/gptj만 허용하여 **Llama generation은 현재 구조적으로 차단**.
- BLUE compute_z는 tensor 출력에도 `[0]`을 사용한다. 검토용 patch를 audit에 남겼고
  실제 모델 parity를 주장하지 않는다.
- 새 모델 공통 W0/ZSRE W0 prediction identity, official-only W&B adapter 미결속.
- checkpoint 저장은 이번 task에서만 승인된 예외다. 기존 최신1개/atomic 교체/W20보존
  API에 연결했지만 동시실행 디스크 reserve와 main-source READY는 아직 봉인하지 않았다.

자원 관측: local projectcap3, 당시 기존GPU2개에서 Qwen이 시작되었다.
filesystem free 관측61,104,771,072bytes는 예약이나 미래 보장이 아니다.
GPU/Slurm 신규실험0, 새 W0평가0, 환경변경0, 자동retry/agent monitoring0.
NO_BROADCAST_NOT_REQUIRED: local 자산 재사용이며 tracked 소형 source/report만 공유한다.

현재 단계: **IMPLEMENTATION_PARTIAL_SHARED_INTEGRATION_BLOCKED / BASELINES_CANCELLED**.
SH4 owner 자체 점검이며 별도 reviewer/GPU PASS를 주장하지 않는다.
