# Generation endpoint 저장 충돌 및 GPT2 W0 reader 교정

사용자 “전부 다 RERUN올리자”에 따라 기존 세 cell만 새 cold attempt로 등록한다. M3 저장 K가 없는 추가 두 cell은 본 교체 범위에 포함하지 않는다. 원 source/raw는 KEEP, NoCP이므로 이전 100 edits를 재개하는 것이 아니다.

61356 Llama는 B1 커밋 후 B2 generation subset에서 `IMMUTABLE_GENERATION_IDENTITY_CONFLICT`로 실패했다. `subset()`의 digest 변수 `key`를 member 복사 loop가 덮어써 고정 `compatibility_member.json`을 쓰는 결함을 실제 W1/W2 저장행으로 재현했다. task-private subclass는 loop 변수명만 분리하며 원 immutable/row/state 검사를 유지한다. SH1 공유 소스는 변경하지 않았다.

교체 준비 중 61358 GPT2도 startup 후 `OBSERVER_CARDINALITY`로 실패했다. W0 reference reader를 runner에만 연결하고 tracking module에는 연결하지 않은 결함이다. 동일 reader를 새 task process의 tracking에도 연결했다. 원 2,000 요청/26,000행·summary 검산이 통과했고 신규 W0 forward는 없다.

61359 collector를 먼저 취소했다. 61357 GPTJ는 실제 RUNNING으로 바뀐 상태였으며 이번 명시 전부 재실행 승인 범위에서 취소했다. 이미 FAILED인 61356/61358은 terminal/raw를 보존했다. baseline60917–60923 hold는 변경하지 않았다.

기존 실제 행 기반 회귀2개와 기존 좁은 회귀8개가 통과했다. 새 source GPU PASS를 주장하지 않는다. 기존 Llama B1 KV/MB4 실제 검산은 통과했지만 endpoint 저장 실패와 분리한다. 평가 일정은 current/post + W5/10/15/20 누적 그대로다.

새 source/config/attempt/W&B ID를 사용하며 합산 cap3 내 독립 GPU1 세 lane과 GPU0 afterany collector로 등록한다. CPU8/59392MiB/hard60416MiB/48h 상한이며 실측 ETA가 아니다. 원 source·교체 이력·미측정 값은 보존하고 반복 모니터/자동 retry는 만들지 않는다.

실제 source `10fa1f6a77243312e8d6e0770262f7016931a36c`로 아래 네 job의 held검사/release를 완료했다. 세 GPU는 독립 lane이며 초기 snapshot은 모두 PENDING이다.

| Cell | Job | 의존성 |
| --- | --- | --- |
| Llama REPRO | 61398 | 없음 |
| GPTJ M1 | 61399 | 없음 |
| GPT2XL M1+M2 | 61400 | 없음 |
| CPU collector | 61401 | afterany:61398:61399:61400 |

[봉인/등록 receipt](../../../../../audits/servers/server4/price-ridge-m1-m3-2k-20261008/subset-repair-submission.json). 검토 수준은 소유자 CPU/source 검산이며 별도 reviewer 없음. 새 실제 GPU 검산·W&B 원격 연결·W20 완료는 NOT_OBSERVED다. own branch에 게시하며 main 통합은 기존 GH 절차로 남긴다.
