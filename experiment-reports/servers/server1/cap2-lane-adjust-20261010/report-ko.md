# SH1 cap2 실제 적용

nonce `USER-GH-S1-S2-CAP2-LANE-ADJUST-20261010-R1` 직접 OWNER_ACK 후 수행.
정본 ae2f3eb4 FULL_READ, envelope SHA eb5c622c7970539f39521dceead39d81485356a8a9e9fa9c39218cccf71d7c2a 일치.
session 01a04939-f93a-7b50-bca0-65438eab2062 / devbox / /mnt/raid5/janghj/ODE-edit / hyunjun1127/ODE-edit 확인.
accepted turn은 현재 수신 turn이며 별도 UUID는 이 보고서에서 추정하지 않는다.

root/live admission `servers/local/gpu-caps.tsv`의 server1 GPU 열만3→2로 수정했다. node/memory183296/job patterns/다른 server 행은 그대로다. 새 전용 준비 WT에도 같은 ignored local cap2 파일을 결속했다. 이전 own author/cap3/refresh 준비 WT에는 독립 local cap 파일이 없고 root 파일을 사용한다. 기존 frozen execution directory는 수정하지 않았다.

## 실제 queue / CPU graph

owner 전체 queue를 확인하고 ReqNodeList/NodeList로 자기 서버를 구분했다. job 이름 필터는 사용하지 않았다. 다른 서버의 job은 변경하지 않았다.

| job | 상태 | GPU | 기존 dependency / 조치 |
|---|---|---:|---|
| 62529 author CF | RUNNING | 1 | KEEP |
| 62530 author zsRE | PENDING, elapsed0, allocation없음 | 1 | afterany62529 그대로 |
| 62583 Qwen history FluCon | RUNNING | 1 | KEEP |
| 62584 collector | PENDING, elapsed0 | 0 | afterany62583 그대로; 원 완료62581/82 coverage 보존 |

lane1=62529→62530, lane2=62583. GPU DAG 최대 antichain 폭2, cycle0, 실제 할당2, legacy overcap없음. 따라서 새 resource edge/hold/release가 필요하지 않았고 scheduler mutation0이다. 임시 hold 잔존0, 취소/재시작/새제출0.

최종 read-only scheduler receipt에서 owner/Command/WorkDir/job명/resource/requeue를 원 제출과 대조했다. 원 execution lock source member 전체 및 script/config SHA 일치. 실행 source/config/raw/CP/W&B 불변. CPU graph/identity 검사 PASS는 GPU qualification이 아니다. 독립 reviewer는 사용하지 않았다.

정확 시각·source SHA·local cap SHA·job snapshot은 `audits/servers/server1/cap2-lane-adjust-20261010/receipt.json`, 재현 검사는 같은 경로 `check.py`다. root dirty/README/다른 작업자 변경 보존. cap2 이후 admission에 적용하며 frozen 과거 cap 기록은 역사 그대로 둔다.

NO_BROADCAST_NOT_REQUIRED. 소형 Git 영수증만 게시하고 GH에 전달; 장기 monitor/자동 retry/전송/삭제 없음.
