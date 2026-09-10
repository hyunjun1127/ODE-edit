# GH 승인: L4 progress barrier SH2 입력 수신
instruction_id: ODEEDIT-S06-BLUE-L4-PROGRESS-PRESERVING-BARRIER-SH2-V1

사용자가 server2에서 지정 설계의 기존 코드·자산을 사용한 구현·실행·분석을 요청했다. 이 요청의 필수 입력 준비로 SH2 단독 pull을 승인한다. 승인 범위는 봉인된 완료 ABC fixture 및 참조 자산뿐이다.

- 발신 server1 root: /mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/local/single-layer-cumulative-risk-abc/20260910-v1/
- 이 root의 허용 하위범위: input.lock.json; A/Early/native-r4/, A/Middle/native-r1/, A/Late/native-r4/에서 prepared.pt 및 W0/ENTRY/N 평가·구조·generation·identity; imports/entries/B010,B050,B090의 W-method-state.pt와 B011,B051,B091의 native-targets.pt/entry metadata; imports/config.json, imports/blue-source/, imports/historical/의 필요한 readonly helper.
- 보고서/code는 d2c808015d8e1b34a039c125139d6d66bbca6c73 Git bytes로 우선 공유. 문서·metadata reference만 필요할 때 불필요한 raw tensor를 받지 않는다.
- 필요한 경우만 추가 허용 자산: /mnt/raid5/janghj/EasyEdit/examples/null_space_project_Meta-Llama-3-8B-Instruct.pt 및 /mnt/raid5/janghj/EasyEdit/examples/data/stats/Meta-Llama-3-8B-Instruct/wikipedia_stats/model.layers.4.mlp.down_proj_float32_mom2_100000.npz.
- 목적지 server2: /mnt/raid5/janghj/ODE-edit/local/blue-l4-progress-barrier/<attempt-id>/imports/ . 기존 동일 자산은 SHA/shape/model/layer/정밀도 결속 후 read-only 재사용한다. Native baseline/code/environment를 server2 임의 local path로 대체하지 않는다.
- owner=SH2 한 명. exact selected file list/size/SHA를 transfer manifest로 만든 뒤 rsync --files-from=<allowlist> --relative --partial <registered-server1>:<validated-root>/ <new-import-stage>/ 형태로 전송한다. Repo-external P/C0는 별도 exact-file 수신으로 기록한다.
- 세 prepared 약10.27GB 외 실제 checkpoint/history/reference 자산 총량은 SH2가 allowlist 작성 시 계산한다. free disk/reserve 확인 후 수행한다.
- 목적지 create-once staging→수신전체 SHA 검증→seal; 충돌 파일 overwrite/--delete/cleanup 금지. Source 읽기 외 mutation 금지. 다른 task의 live output, shared credentials, full pretrained 재전송은 승인하지 않는다.
- 정상 완료 입력의 비파괴 전송이므로 반복 사용자 승인 대기를 추가하지 않는다. 범위 밖 asset·경로·overwrite 필요 또는 인증 실패는 보고한다.
- 수신 증거: transfers/verifications/2026-09-11-blue-l4-progress-barrier-sh2-*.json 및 task local receipt. 재사용·전송·미가용을 분리한다.
