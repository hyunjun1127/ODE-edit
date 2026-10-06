# PRICE cap/base repair: Llama·Qwen 6-cell 2k

## 최신: 2026-10-07 사용자 recall — 로깅 오류 수리·6개 교체 제출

LLAMA_CAP075 **59931 FAILED(1:0), 200 allocated GPU-sec, commit0** 확인. `cap_tracking.py`가 native `nll[100][6]`에 `sum(payload['nll'])/B`를 적용하여 첫 candidate 기록 시 `TypeError: int + list`가 발생했다. 원 candidate는 authoritative JSONL에 기록됐고 RAM rollback receipt는 verified=true다. 성능/수식/solver 실패가 아니라 W&B caller scalar 변환 결함이다.

모든 MEMIT6 및 대기 Alpha6의 frozen caller에 같은 결함을 확인했다. 다른 arm에서 실제 TypeError가 발생했다고 주장하지 않는다. Qwen59934는 실행 중 사용자 승인으로 취소(333 allocated GPU-sec); 나머지 MEMIT 후속/collector와 Alpha6/collector는 pending 상태에서 교체 취소했다. Qwen은 강제 취소이므로 별도 rollback PASS를 주장하지 않는다. 기존 source/config/raw/실패 자료는 보존했다. 최초 attempt GPU 할당 총계는 **533 GPU-sec**이며 새 attempt 비용과 분리한다.

수리: 요청별 문맥 평균→요청 평균으로 `fit/nll` 변환, post-start logger 변환/전송 오류의 scientific exception 전파 차단, bounded caller-degraded receipt. Startup online 인증 gate와 authoritative raw IO 오류 gate는 그대로다. SH1 helper, 학습 수식/계수/gradient/예산/projection/모델/evaluator 변경0. 기존 실제 candidate/endpoint replay 및 caller 오류 격리 CPU 검사 **5개 PASS**; 새 GPU 결과 PASS가 아니다.

실행 source **2440e548be39e55a99747d7847d21a88df419b93**, config SHA **68137b5203505d24dc38abc8321e545dcf9ae1fedff92128438a4075a26451bc**. 새 독립 cold attempt는 `/data/janghj/ODE-edit/local/jlz-price-cap-base-repair-2k/logging-repair-20261007`; checkpoint resume가 아니다. 기존 Llama exact W0 reuse 정책 유지, Qwen의 완전한 exact W0가 없으면 필요한 최초 관측은 수행한다.

| Model/arm | 새 job | afterany |
|---|---:|---|
| LLAMA_CAP075 | 60001 | 없음 |
| LLAMA_CAP100 | 60002 | 60001 |
| LLAMA_FREE100 | 60003 | 60002 |
| QWEN_CAP075 | 60004 | 없음 |
| QWEN_CAP100 | 60005 | 60004 |
| QWEN_FREE100 | 60006 | 60005 |
| CPU collector | 60007 | 60001:60002:60003:60004:60005:60006 |

Alpha 후속까지 **14개 전체 held owner/fullargv/source/resources/dependency/script bytes 검사 후 release**했다. 합산 GPU cap2, 각GPU1/CPU8/59392MiB/48h/exportNONE/Requeue0, CPU collector0GPU/8CPU/24576MiB/4h 유지. 두 task 합산 serializer reserve 31,892,963,328 bytes와 fresh 공간/노드/memory/QoS admission 확인. Test-only ID는 실 job에서 제외했다. 등록 직후 단1회 snapshot은 전부 PENDING(Reason None)이며 실제 repaired B1/W&B startup/W20는 **NOT_OBSERVED**다. 새 반복 polling/heartbeat/autoretry 없이 봉인 runner/collector만 진행한다.

[새 제출 receipt](../../../../runs/jlz-price-cap-base-repair-2k/logging-repair-20261007/submission.json), [수리 owner audit](../../../../audits/servers/server4/jlz-price-cap-base-repair-2k/logging-repair-20261007/owner-review.json). 소형 source/receipt/report만 Git, 원 raw local KEEP; NO_BROADCAST_NOT_REQUIRED. 별도 독립 reviewer 미사용.

## 아래는 교체 전 최초 등록 원문 — 현재 job 상태가 아님

현재 **6개 GPU job + CPU collector 전량 held 검사·release 완료**. 등록 직후 단일 snapshot은 전부 PENDING이다. 실제 B1/GPU 검산·W&B online startup·W20 결과는 NOT_OBSERVED다.

실행 source: `87a5a736c455d5082f5f666ae562f9edc4e5d3b2`; config SHA256: `3e43efa32409423a51ef83495b4f5393243522ff0513679a30c8d2e726096bf8`. 게시/분석 commit과 실행 source는 구분한다.

| Model | Arm | Base | Local cap | Job | Afterany | 최초 상태 |
|---|---|---:|---|---:|---|---|
| LLAMA | CAP075 | .75 | .75a | 59931 | 없음 | PENDING |
| LLAMA | CAP100 | 1 | .75a | 59932 | 59931 | PENDING |
| LLAMA | FREE100 | 1 | 없음(null) | 59933 | 59932 | PENDING |
| QWEN | CAP075 | .75 | .75a | 59934 | 없음 | PENDING |
| QWEN | CAP100 | 1 | .75a | 59935 | 59934 | PENDING |
| QWEN | FREE100 | 1 | 없음(null) | 59936 | 59935 | PENDING |

CPU collector `59937`: afterany 정확6개 GPU 부모, GPU0/CPU8/24576MiB/4h. 두 model lane은 독립이며 predecessor 성능 PASS는 조건이 아니다. 모든 owner/source/fullargv/node/GPU/CPU/memory/wall/exportNONE/Requeue0/dependency 및 제출 script bytes를 held 상태에서 검사했다. Test-only resource check는 실제 job ID로 보고하지 않았다.

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

## 인계 경계

원본 attempt: `/data/janghj/ODE-edit/local/jlz-price-cap-base-repair-2k/attempt`. [제출 영수증](../../../../runs/jlz-price-cap-base-repair-2k/submission.json), [held/admission 감사](../../../../audits/servers/server4/jlz-price-cap-base-repair-2k/held-submission.json).

Release 뒤 단1회 scheduler snapshot만 취득했다. 반복 polling/heartbeat/자동 재시도는 없고 `monitoring_active=false`, `automatic_resume=false`다. 기존 job 변경·취소0. 봉인 runner는 B20 및 collector까지 진행하되 실제 기술 오류는 원 증거를 보존하고 차단한다. 상세 완료 리뷰는 사용자 recall 시 수행한다.
