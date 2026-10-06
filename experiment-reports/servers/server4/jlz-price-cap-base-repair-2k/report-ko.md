# PRICE cap/base repair: Llama·Qwen 6-cell 2k

현재 구현·정적/입력 결속 완료, **미제출**. 실제 B1/GPU 검산과 W20 결과는 NOT_OBSERVED다.

| Model | Arm | Base | Local cap | 실제 상태 |
|---|---|---:|---|---|
| LLAMA | CAP075 | .75 | .75a | NOT_RUN |
| LLAMA | CAP100 | 1 | .75a | NOT_RUN |
| LLAMA | FREE100 | 1 | 없음(null) | NOT_RUN |
| QWEN | CAP075 | .75 | .75a | NOT_RUN |
| QWEN | CAP100 | 1 | .75a | NOT_RUN |
| QWEN | FREE100 | 1 | 없음(null) | NOT_RUN |

각 cell은 cold W0/H0, BS100×20, 같은 순서의 2000요청이다. 총 12000 applications이며 서로 다른 12000요청이라는 뜻은 아니다. FREE075/FLAT/REVERSE/추가 baseline은 제외한다.

## 구현 및 검산 범위

- 새 `cap_*.py`만 추가했다. 원 PRICE 0415 실행 Python closure와 기존 source/raw/jobs는 변경하지 않았다. Qwen은 task-local adapter에서 모델 종류·차원·readout27을 명시 결속했다. 기존 Llama31 기본 경로는 그대로다.
- ZERO/CAP 소유자 태그를 보존하고 선택된 endpoint만 정확 0/cap으로 저장한다. 양의 interior, moment, gradient reentry는 유지한다. tolerance/계수/목적 변경과 postcast rescue는 없다.
- beta maximum은 `max(base,.75*pi_max)`. FREE100은 cap=null이며 숨은 .75 clip을 사용하지 않는다. 가격·stage·own update·spend/slack·support/reentry와 실현 ratio의 유효 분모/충분통계를 producer/collector에 연결했다.
- 검토는 SH4 owner source/static/import/CLI/config audit다. 별도 reviewer나 toy/synthetic/GPU PASS를 주장하지 않는다. 실제 B1 cached geometry/proposal의 기존 LOO/KKT/FP32 feasibility, terminal payload/H 검사는 봉인 runner에서 수행한다.
- Llama W0는 원 scalar raw/token/cold/evaluator closure를 확인하여 참조 재사용하며 startup에서 runtime/device를 다시 비교한다. Qwen은 동일 조건의 W0가 없으므로 첫 cell에서 관측하고 뒤 cell은 exact identity일 때 재사용한다. 편집 상태/checkpoint 재개가 아니다.

## 자원·기록

- 최신 사용자 server4 **총 GPU cap2**를 두 모델과 같은 사용자 기존 allocation/admitted 전체에 적용한다. 이전 tracked/local3을 승계하지 않는다. 각 cell GPU1/CPU8/59392MiB, 요청 wall48h 상한. wall은 ETA가 아니다.
- 사전 추정 host peak: Llama37.92GiB, Qwen47.24GiB. GPU peak: Llama66.82GiB, Qwen73.62GiB. W&B sidecar를 포함한 계획치이며 실측 peak가 아니다.
- 여섯 실행의 scalar/evaluation/spool/atomic/error reserve 상한 약14.85GiB, 준비 시 free 약78.19GiB. 공유 filesystem의 미래 여유나 quota 보장이라는 뜻은 아니다. 매 batch fresh space/inode guard를 유지한다.
- SH1 `experiment_tracking` helper readonly 사용. 각 신규 job에서 online startup을 확인한 뒤 모델을 읽는다. entity `wkdguswns2256`, project `layer allocation`; scalar만 기록, raw/prompt/tensor/code/console/secret 업로드 금지.
- noCP, exact resume NOT_AVAILABLE. Source/compact report/manifest만 Git, raw는 local KEEP. NO_BROADCAST_NOT_REQUIRED.

결과와 비용은 collector가 실제 row/state/부모 accounting을 재집계한 뒤 기록한다. 미측정 RS/PS/NS/Harmonic/W20 값을 0이나 이전 결과로 대체하지 않는다.
