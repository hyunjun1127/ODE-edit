# GH 승인: A/B 공통 fixture·source·baseline 공유

approval_id: ODEEDIT-MULTILAYER-AB-S1-S2-INPUT-SHARING-20260911-V1
instructions: ODEEDIT-S06-MULTILAYER-JOINT-EDIT-A-SH1-V1 / ODEEDIT-S06-MULTILAYER-DAMAGE-COMPENSATION-B-SH2-V1
사용자는 SH1/SH2가 동일 sample/checkpoint로 비교하도록 요청했다. 필수 입력과 공통 준비/결과 공유를 아래 범위로 승인한다.

1. SH2 단독 rsync owner. S1 source는 다음의 봉인된 정확한 필요 파일만:
- /mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/local/single-layer-cumulative-risk-abc/20260910-v1/ 의 input.lock, imports/entries/B010/B050/B090 checkpoint, B011/B051/B091 target/context/entry/commit, imports/config.json 및 readonly native/helper source, A 세 prepared와 reference receipts.
- /mnt/raid5/janghj/.codex/worktrees/odeeditsh1-l4-two-memory-routing-v2/local/l4-two-memory-conflict-routing/20260911-v2/ 의 확정 teacher/entry/asset manifests 중 실제 필요한 참고만. v2 bank/위험 calibration을 새식에 자동 재사용하지 않는다.
- 신규 SH1 worktree의 local/multilayer-joint-compensation/20260911-v1/common/ 안 봉인 fixture/history/bank/teacher/reference 및 원본 source closure.
- 첨부 원문 /mnt/raid5/janghj/.codex/attachments/cb28c4e5-2f55-4a68-ab49-2691d5de5559/pasted-text.txt 는 exact single-file read/pull 허용. 지정SHA/bytes를 봉인한다.
2. S2 receiving create-once root:
 /mnt/raid5/janghj/ODE-edit/local/multilayer-joint-compensation/20260911-v1/imports/
신규worktree local루트는 stable root를 참조하며 임의 overwrites하지 않는다.
3. 이미 S2에 있는 migrated CP/ABC/EP imports는 current SHA/schema/reference 확인 후 그대로 재사용. SH2 원본은
 /mnt/raid5/janghj/ODE-edit/local/checkpoint-archives/server4-migration-20260911-v1/
 /mnt/raid5/janghj/ODE-edit/local/blue-checkpoint-downstream/20260909-v1/imports/initial6-v1/
 /mnt/raid5/janghj/ODE-edit/local/blue-l4-progress-barrier/attempt-v1/
안의 대상3checkpoint/targets/정확한 BLUE311b076a native-source 및 hparams/manifest로 한정한다.
필요한 S2 exact native-source/baseline/common rows를 SH1으로 보낼 때 destination은 신규 SH1 worktree local/multilayer-joint-compensation/20260911-v1/imports/ 또는 동일 server1 root이다. 양방향 payload를 모두 SH2만 전송한다.
4. 필요한 P/C0/momit-cov가 destination에 없을 때:
 /mnt/raid5/janghj/EasyEdit/examples/null_space_project_Meta-Llama-3-8B-Instruct.pt
 /mnt/raid5/janghj/EasyEdit/examples/data/stats/Meta-Llama-3-8B-Instruct/wikipedia_stats/model.layers.{4,5,6,7,8}.mlp.down_proj_float32_mom2_100000.npz
의 실제 존재 exact-file manifest/SHA/size를 먼저 확정하여 신규 imports/assets/로 nonoverwrite copy만 허용. Brace/glob 자체를 copy/delete target으로 삼지 않는다. 공용자산 원본 수정0/재생성0. Full model은 local identity reuse 우선; 새 대형 pretrained 이동이 필요하면 별도 보고.
5. source owner가 manifest/READY 봉인→SH2 space/inodes/reserve 및 existing copies 확인→--files-from exact allowlist/--relative staging pull/push→destination fullsize/SHA/schema/closure verify→atomic READY seal. --delete/--remove-source-files/overwrite0. 큰 source tree 통째로 미러0.
6. 실험중 raw/임시 files를 공유하지 않고 완료한 common/endpoint bundle만 사용. Source/report는 Git 우선. Raw 전체3서버 broadcast는 NO_BROADCAST_NOT_REQUIRED(이번S1/S2 exact common sharing 및 localretention)로 대체. Server4 source/reserved/free-space/삭제작업 변경0.
7. Native baseline source가 S1/S2 봉인 imports에 빠져 있으면 SH2가 registered rke-server4:/data/janghj/BLUE의 pinned commit311b076a92e4ed0f14f5c8b4909732da781bc5f7/treef3c933c31cba2fe979c5c34546a99a72e6beb763에서 필요한 tracked native code/hparam/config와 commit manifest만 read-only 조회·exact archive할 수 있다. 해당 Git object 우선, live dirty 파일 치환 금지. 이것은 source-only 예외이며 S4 checkpoint/raw/model/cache 재전송·감사재개·source변경은0이다.
8. 위 범위 밖 repo-external paths, 인증 재설정, 데이터/비밀파일, 다른task raw는 승인하지 않는다. Samehost imports의 재해시를 넘는 GH 중복 감사0. SH2는 transfers/verifications/2026-09-11-multilayer-joint-compensation/server2/에 perbundle receipt; SH1은 source seal/수신ACK.
