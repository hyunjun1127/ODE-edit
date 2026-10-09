# Qwen baseline server4 → server2 이전 인계

Nonce: `USER-GH-S4-S2-QWEN-BASELINES-MIGRATION-RESULTS-20261009-R1`.
상태: **SOURCE_CEASED / HANDOFF_READY**, server4 새 제출0. server2 제출 완료를 뜻하지 않는다.

## 실제 원인과 종료

fresh accounting에서 CF FT61783은 FAILED1:0, 2026-10-09 12:31:28–12:31:29 KST, elapsed1초였다. stderr는 frozen launcher의 `free>=32GiB` 검사에서 `RESOURCE_BLOCKED_STORAGE_KEEP_SOURCE`를 기록했다. 모델/편집 pipeline 이전에 실패했으며 해당 attempt의 runs/shared-W0 관측 파일은 없다. 공간 소모 주체는 이 점검으로 확정하지 않았다.

원 submission/source/Command/WorkDir/owner/server와 현재 scontrol/accounting을 결속했다. 원 CF61783은 terminal 실패 KEEP(취소0). 미시작 CF GPU5+CF archive6 및 zsRE GPU6+archive6, 총23개를 후속부터 hold한 뒤 역순 취소했다. 각 취소 직전 다시 PENDING/StartUnknown/elapsed0/미할당을 검산했다.

| Method | CF source job | zsRE source job | disposition |
|---|---|---|---|
| FT | 61783 | 61755 | CF FAILED_KEEP; zsRE 이전 취소 |
| MEMIT | 61785 | 61757 | 미시작 이전 취소 |
| AlphaEdit | 61787 | 61759 | 미시작 이전 취소 |
| AlphaEdit-BLUE | 61789 | 61761 | 미시작 이전 취소 |
| MEMIT_FE | 61791 | 61763 | 미시작 이전 취소 |
| SPHERE | 61793 | 61765 | 미시작 이전 취소 |

archive61784/86/88/90/92/94 및61756/58/60/62/64/66도 함께 취소했다. PRICE/OURS/tuning/heldout/다른 서버는 변경0. old Llama60917..60923은 이미 CANCELLED/elapsed0임을 확인했으며 부활/추가취소하지 않았다. source/raw/CP/로그 삭제0, 대형전송0, GPU계산0.

## SH2 source/host 인계

`audits/servers/server4/qwen-migration-20261009/handoff-manifest.json`의20개 metadata파일(84,488B)을 Git으로 전달한다. source closure와 input lock의 기존 모든 member SHA를 CPU로 다시 대조했다. 모델/C0/P 대형 SHA는 기존 provenance이며 현 server2 자산 일치 확인을 대체하지 않는다.

- CF execution `dc80ec529c940019d1bee27a67a4e908eb37cc64`, zsRE 원 execution `d614add5e4c650821ed8d2503c071a1e02605ca8` 보존. 공통 native display companion 수리가 포함된 CF source를 기준으로 destination wrapper를 만들되 zsRE 과학 설정은 불변.
- FT/MEMIT/AlphaEdit/BLUE/MEMIT_FE/SPHERE × CF/zsRE 정확12. BLUE L2=1 유지, grid추가0. editseed0, BS100×20, cold2K, 원 native hparams/FP32/TF32off 유지.
- CF 원 일정은 `W0_AND_W20_FIRST2000` generation이며 DEFERRED로 임의 변경하지 않는다. zsRE에 CF생성 없음. 이 attempt에 reusable W0/context/CP는 생성되지 않았다. 원 필수 W0만 수행하거나 별도로 증명된 동일identity raw를 재사용한다.
- snapshot/tokenizer/source/ordered-stream/C0/P/reference/runtime fingerprint와 각host 경로를 구분했다. server2는 own node/QoS/cap/CPU/RAM/GPU, 로컬 자산/SDK/인증, source/job identity를 새 결속해야 한다. SH4의 credential/env 파일을 복제하지 않는다.
- SH4 submitter의 hardcoded server4/oldIDs를 그대로 실행하지 않는다. shared algorithm/logger는 재사용하고 SH2 자체 launcher/collector/host binding 사용. 별도 GPU qualification은 NOT_RUN_USER_DISABLED.
- checkpoint 최신1개/W20보존 유지. archive의 server4 cutover/trust/oldjob proof를 server2로 relabel하지 않는다. SH2 own source-owner adoption과 실제 새job proof, server1의 per-payload admission 및 독립검증/consumer종료가 필요하다. 원본 삭제는 검증조건 만족 후 exactpayload만이며 이번 인계 자체는 전송/삭제 PASS가 아니다.

## README 결과 eligibility

현재 이전 대상 Qwen12에는 완료 W20 main0, commit0, 실제평가0이다. 수치 칸은 비워두고 실패/이전 취소 상태를 구분한다. CF Flu/Con도 NOT_MEASURED이며 0 또는 DEFERRED로 바꾸지 않는다. 새 jobID는 SH2 실제 등록 후만 기입한다.

CPU matrix: `audits/servers/server4/qwen-migration-20261009/result-eligibility.json` 및 `.csv`. source/config/ordered sample SHA와 observed status를 분리했다. 이 scope 및 과거 취소 Llama baseline에서 신규 eligible numeric update0이다. 다른 task의 tuning/tier/heldout/PRICE를 자동 승격하지 않았고, README의 명시 historical 예외는 유지한다. README는 GH 단독 통합이다.

전체 scheduler 전후증거는 `/data/janghj/ODE-edit/local/qwen-migration-20261009/`에 보존. 추가승인/모델로드/복원/forward/fit/recurringmonitor/자동retry 없음. 직접 SH2/GH 전달 상태는 별도 delivery receipt로 구분한다.
