# Baseline 표 최신화 및 Server2 미평가 CF FLU/CON 배정

사용자 요청에 따라 SH1/SH2에 실시간 직접 전달했고 양쪽 명시 OWNER_ACK를 확인했다.
Nonce `USER-GH-S1-S2-BASELINE-REFRESH-S2-FLUCON-20261010-R1`, 정본 `92704580`.

| 담당 | accepted turn | 수행 범위 | 현재 확인 단계 |
| --- | --- | --- | --- |
| SH1 | 01a124dc-4d21-70f0-9fa7-94452f671f11 | 완료 CF/zsRE raw 검산, 기존62581–62584 dedup, 미평가CP 인계 | OWNER_ACK/진행중 |
| SH2 | 01a124db-95bd-7231-981d-429bbc98305c | 완료결과 검산 + 누락 CF W20 FLU/CON eval-only 실제 제출 | OWNER_ACK/입력·runner 준비중 |

각 서버 cap3, 더 엄격한 memory/task/QoS 제한 유지. 기존 Server1 평가를 취소하거나
Server2에 중복 등록하지 않는다. checkpoint/cohort/protocol이 같은 완료생성이 있으면 재사용한다.
필요한 원CP 입력만 exact allowlist/bytes/fullSHA/수신검증/용량확인 후 Server2 신규경로로
비파괴 복사할 수 있으며 원본 삭제·덮어쓰기 권한은 없다. 미완료W20·부족자산은 개별 사유 보고.

zsRE는 공개-query 및 저장 predicted/target token correctness의 요청별 평균을 검산한다.
Loc은 loc_ans 정확도이며 W0agreement가 아니다. CF 생성은 기존 CAKE호환 protocol/reference/seed를
유지하고 raw bits/cosine 보존, 표만 원평균×100 half-up2. 재편집/새W0/별도GPUqualification 없음.

최초 전달 시점에는 실제 신규job 영수증과 새결과를 아직 회수하지 않았다.
README는 GH가 owner의 compactrows/실제ID·상태를 받아 통합한다. 기존 b047b099 완료셀을
추정값으로 변경하지 않았다. 원자료/GPU 완료 대기·recurringmonitor·자동retry 없음.

후속 SH1 publication d6733abb에서62581/62582 완료 generation을 회수하여 history표4수치셀과
62583 RUNNING 상태2셀을 갱신했다. 이어 SH2 중간 결과 publication417a12e7에서
완료23행 검산을 회수해 새Qwen3건의10수치셀과 author62531 RUNNING 상태1셀을 갱신했다.
후속 publication0e30ad58에서 SH2 신규FLUCON13GPU+1CPU의 실제held검사/release영수증을 회수했다.
17:28:13 KST snapshot은62864/62865 RUNNING,62866–62877 Dependency PENDING이다.
README의13행Flu/Con26상태셀을 갱신하고 기존factual/zsRE/완료생성수치를 보존했다.
이로써 본 지시의 검산결과 회수·미평가분 제출·등록상태 표 반영을 완료했다.
새평가의 최종성능완료는 아직 관측하지 않았으며 장기GPU완료대기/monitor는 하지 않는다.
[SH1 완료 결과 통합](server1-integration.md).
[SH2 중간 결과 통합](server2-results-integration.md).
[SH2 실제 평가 등록 통합](server2-submission-integration.md).

SH1 첫 연결은 thread/resume에서 timeout이었으며 미전달로 기록했다. 이후 idle 및
turn/start 미발행을 확인해 공식 direct로 전달했고 같은 accepted turn에서 명시 ACK를 확인했다.
[전달 증거](../../../audits/global/baseline-refresh-s2-flucon-20261010/dispatch.json).
