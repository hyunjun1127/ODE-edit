# Server1 두 lane 자원 의존성 조정

nonce USER-GH-SERVER1-TWO-LANES-20261011-R1. accepted turn `01a127b7-83d4-7af0-a20c-32ecff5c036a`. session/CWD/origin/devbox 정합 확인. 정본7776cb94/envelope SHA55531047634002ac2341d2aa3cbce1d6d9a83fa665a054169bf15037286ebabf 전체 확인. root local cap2 유지, 메모리 및 QoS 변경 없음.

단발 전체 own devbox GPU inventory: 실제 RUNNING63154/GPU1, admitted DAG 폭2. 과거3GPU 초과 구간은 이미 종료되어 현재 초과할당0. FE Qwen 두 head가 모두63154를 기다리는 장벽 때문에 두 번째 lane이 비어 있었다. CAKE6까지 이름이 아닌 원 submission/config/script/lock/source closure/full argv와 scheduler owner/Command/WorkDir로 결속했다. 전체 현재 GPU job9개이며 다른 서버 job은 변경하지 않았다.

| 변경 job | 전 resource dependency | 후 resource dependency |
|---|---|---|
| 63217 Qwen CF FE | afterany63154 | 없음 |
| 63296 Llama CF CAKE | afterany63217:63218 | afterany63217 |
| 63297 Llama zsRE CAKE | afterany63217:63218 | afterany63218 |

모두 변경 직전 PENDING/elapsed0/allocation0을 재확인하고 descendants 먼저 임시 hold, resource-only update 후 실제 scheduler graph 검사 및 release. afterok input gate 삭제0, 새 job0, 취소0, RUNNING 변경0. 원 source/config/script/fullargv 불변 SHA 재검산. 임시hold 잔존0.

두 lane: A `63217 → 63296 → 63298 → 63300`; B `63154 → 63218 → 63297 → 63299 → 63301`. CPU 및 actual scheduler DAG 폭2/cycle0, 남은 edge 모두 현재 유효 job. 기존 raw/CP/frozen runtime 보존.

직후 bounded snapshot:63154 RUNNING,63217 PENDING/Reason=None/dependency없음. 그 외 후속은 lane 내 선행을 기다린다. 두 lane이 scheduler상 열렸다는 뜻이며 두 GPU가 이미 RUNNING이라고 표현하지 않는다. 새 장기 poll/heartbeat/자동retry 없음. 상세 시각·상태·source/hash는 own audits before.json/plan.json/held-updates.json/after.json/receipt.json에 보존. README는 GH 소유, 성능 수치 변경 없음.
