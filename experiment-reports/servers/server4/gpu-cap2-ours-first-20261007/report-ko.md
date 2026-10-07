# server4 GPU cap2 / OURS 우선 대기 DAG

최신 사용자 요청 “gpu cap 2로 줄이자. 지금 돌아가는건 놔두고 dependency 걸어놓은것들 ours 먼저 돌리고 그다음 baseline들 올려”를 적용했다. server4 합산 cap2가 과거 cap3을 대체한다. 다른 서버·과학 task 범위·hparam·frozen source/config는 변경하지 않았다.

## 실행 중 작업 보존과 대기 순서

단발 검산 시 60105(Llama OURS Alpha CAP075), 60617(GPT-J OURS MEMIT CAP100), 60619(GPT-J OURS Alpha CAP075)가 각 GPU1로 RUNNING이었다. 기존 합계3은 사용자 지시대로 보존하며, 현재 할당이 2라고 주장하지 않는다. 세 RUNNING 모두 종료된 이후에만 아래 대기 OURS의 두 lane head가 시작한다. 종료는 afterany이며 과학 성능 통과 gate가 아니다.

```text
기존 RUNNING 60105 + 60617 + 60619 (그대로 보존)
  └─ 모두 종료
       ├─ OURS 60618 → 60106 → 60107
       └─ OURS 60620 → 60621
            └─ 모든 OURS 종료 → baseline 60917 MEMIT
                 ├─ 60918 PRUNE → 60921 BLUE → 60920 AlphaEdit
                 └─ 60919 RECT  → 60922 CAKE
```

60917의 기존 전체 OURS dependency를 유지했고, 60920도 기존 60917과 추가 60921을 함께 기다린다. OURS가 baseline을 기다리는 역방향 edge는 없다. 현재 세 RUNNING을 제외한 미래 GPU DAG의 최대 antichain은 2, cycle은 0이다. 기본 작업별 GPU1/CPU8/59392MiB/QoS/48h 상한은 그대로이며 더 엄격한 task 제한을 확대하지 않았다.

## 실제 변경 및 검산

정확 own PENDING/elapsed0/startUnknown/allocation0/Command/WorkDir/owner를 변경 직전 확인했다. 60618·60620·60106·60920만 임시 hold → resource-only dependency 수정 → 검산 → 같은 ID release했다. 영구 hold0, 취소0, 신규 제출0, RUNNING 변경0이다. GPU0 collector60108·60622·60923의 coverage 의존성도 변경0이다.

source/config/lock/archive/submission/launcher 29개 파일의 SHA를 전후 비교해 모두 동일했다. GPT-J source298be5da, Llama Alpha52263371, Llama baseline e3019677의 원 실행 bytes와 W&B identity/raw/RAM 진행을 보존했다. root dirty와 원 cap3 제출 receipt는 역사 그대로 유지했다. own root/admission/정책·baseline WT 및 이번 clean WT의 ignored cap 파일은 server4 GPU 열만2로 맞추고 node/memory/job pattern을 유지했다.

리뷰 수준: SH4가 실제 scheduler/파일 SHA 검산을 수행했다. 별도 `resume_registration_review` agent는 CPU DAG/source-path만 검토했으며 scheduler 실행·모델 재구성·tensor replay·과학 결과 리뷰는 하지 않았다. helper의 이름 패턴만으로 0 GPU라고 판단하지 않고 전체 same-owner 자원 queue를 exact server4 owner/source/receipt로 결속했다.

[변경 전후 소형 receipt](../../../../audits/servers/server4/gpu-cap2-ours-first-20261007/scheduling-receipt.json), [현재 상태](../../../../tasks/status/gpu-cap2-ours-first-20261007/server4.json). 전체 scheduler stdout/argv는 ignored local에 보존했다. compact receipt만 Git, raw/model/credential/SDK spool 전송0. `NO_BROADCAST_NOT_REQUIRED`: 같은 host 설정과 소형 Git 기록만 필요하다. bounded post-release 확인 후 STOP; 새 recurring monitor/heartbeat/automatic retry/resume/실험 완료 대기0.
