# Server2 checkpoint 설정 확인 및 PRUNE·RECT 취소

사용자: “지금 server2에 올라간 실험들 모두 checkpoint 저장하도록 한건가? prune과 rect는 일단 job내리자”.
관측:2026-10-09 02:23:01 KST(2026-10-08 17:23:01 UTC).

**모두 checkpoint 저장형은 아니다.** 실제 등록된 frozen source/config를 대조했다.

| 등록 그룹 | GPU jobs | Checkpoint 설정 |
| --- | --- | --- |
| 기존 native MEMIT/AlphaEdit/CAKE/BLUE | 61534/61535/61536/61537 | `noCP=true`, `exact_resume=NOT_AVAILABLE`; 실제 runner도 noCP를 강제 |
| official FT/MEMIT/AlphaEdit/BLUE/FE/SPHERE qualification | 61619/61624..28 | 실제 save 연결, FP32 W/native H/RNG/context/cursor/identity 최신1개 유지 |

Native source는 `1db8d947ce77123599dd78003560d71facd3af68`, official source는
`18e7fbd99bc2692c9da9ebdc9f4598caba79a1f8`이다. Official 현재 단계는 continuous B3와 durable
B2→B3 재개 qualification이며 W20 본실행이 아니다. 본 chain 코드에는 매 batch 저장 및 W20
보존이 연결돼 있다. 저장 설정과 실제 파일 생성/완료는 구분하며 이번 source 확인은 tensor를
load하거나 실험 진행률을 읽지 않았다. 기존 noCP job에 checkpoint 저장을 hotpatch하지 않았다.

PRUNE **61538**, RECT **61539**는 exact owner/server2/Command/WorkDir/source/config와
현재 unallocated PENDING을 검산한 뒤 `scancel 61538 61539`로 취소했다. Exit0;
scontrol `CANCELLED`, sacct `CANCELLED by 1025`, elapsed00:00:00/AllocTRES없음,
정확 target의 post active queue0을 확인했다. 파일/source/raw/log 삭제0이다.

취소 전 official 대기 FT61619/MEMIT61624는 두 취소 target의 afterany에 묶여 있었다.
USER cap2를 유지하려고 **아직 미할당 PENDING 두 job의 자원 의존성만** surviving CAKE61536와
BLUE61537 afterany AND로 먼저 연결하고 실제 dependency를 검산했다. 다른 과학 소스/config/
checkpoint/W&B identity 및 원 submission receipt는 변경하지 않았다.

post snapshot은 native MEMIT61534/Alpha61535 RUNNING2, CAKE61536/BLUE61537 PENDING,
official6+collector PENDING이다. Native 공유 collector61540은 남은 네 arm도 수집하므로 유지했다.
다른 실험/서버/OURS/W0/FE 취소0, 신규 제출/fit/model forward/CP 생성0이다.
상세 identity/hash/취소·자원 scheduling receipt는
`audits/servers/server2/prune-rect-stop-20261009/receipt-r1.json`에 기록했다.
새 recurring monitor/heartbeat/auto retry는 없다.
