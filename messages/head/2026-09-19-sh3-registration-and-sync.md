# GH → SH3: 신규 등록 및 repo 최신화

Instruction ID: `ODEEDIT-GH-SH3-REGISTRATION-SYNC-20260919-V1`
Nonce: `ODEEDIT-GH-SH3-REGISTER-SYNC-20260919-R1`
사용자 요청: 지정 SH3 task 확인·repo 등록·최신화. 원 bootstrap 범위를 계승한다.

## 실제 경계와 권한

대상 session `01a0b9c8-d12c-7133-b336-cc1b5fc2a6b3`, app host lab123,
실제 ubuntu/janghj, CWD `/data/janghj/ODE-edit`, repo `hyunjun1127/ODE-edit`.
GH session `01a04939-8873-7673-8dca-4c7fc5e31af0`.
GH는 app read/SSH hostname·origin/socket과 SH3 M0를 대조했다.
`ACK nonce=ODEEDIT-SH3-BOOTSTRAP-GH-0eeb54f4756c` — 해당 M0와 요청을 정확 GH에서 수신했다.

현재 등록은 control-plane 활성화이며 과학 제출 허가가 아니다.
Tracked project ceiling은 이미 **2GPU**, host 요청 상한은 **119GiB/GPU (121856MiB)**다.
기존 local cap0는 미제출 상태로 유지한다. 별도 실험 task 없이 모델/GPU/eval/Slurmwrite0.
다른 job·pause·실험 상태 변경0. 설치/upgrade/대용량 복사/삭제0.

## Repo 및 local boundary 최신화

1. 본 게시 commit을 fetch하고 registry·server3 active record·PROTOCOL·no-checkpoint 정책을 읽는다.
2. 기존 root와 bootstrap worktree의 dirty/branch를 확인한다. **clean root main만**
   `git merge --ff-only origin/main`으로 갱신하며, dirty면 보존하고 전용 clean worktree를 사용한다.
   원 dirty를 stash/reset/clean하지 않는다. Bootstrap report local commit은 별도 branch로 보존한다.
3. 존재하지 않았던 root ignored `/data/janghj/ODE-edit/servers/local/session-boundary.env` 작성 허용.
   `ODEEDIT_REPOSITORY_ID=hyunjun1127/ODE-edit`,
   `ODEEDIT_REPOSITORY_CWD=/data/janghj/ODE-edit`,
   `ODEEDIT_CODEX_SESSION_ID=01a0b9c8-d12c-7133-b336-cc1b5fc2a6b3`,
   `ODEEDIT_CURRENT_SERVER=server3`.
   사용자 모델/profile 고정0. 해당 값으로 helper 실행 후 실제 PASS/FAIL을 기록한다.
   Dedicated bootstrap worktree에도 필요한 경우 그 **실제 절대 CWD**를 사용하는 같은
   ignored boundary 파일만 작성한다. 기존 다른 내용이 있으면 보존 후 충돌을 보고한다.
4. Shared `user.* / agent.*` 변경 없이 command-scoped git `-c` identity로 commit 가능하다.
   Worktree identity가 이미 합법적으로 동작하면 그것을 사용한다. Global config는 수정하지 않는다.
5. 전용 branch에 원 bootstrap 허용 report/ACK/audit만 commit·nonforce push하고 정확 SHA를 GH로 전달한다.
   **Main 통합은 GH 담당**이며 동일 registry를 SH3가 중복 수정하지 않는다.
6. 최신 source 확인은 metadata-only; 모델·자산 전체 hash·새 GPU 검사를 이 등록의 선행조건으로 만들지 않는다.

## 환경 요청에 대한 현재 답변

- EasyEdit 위치 `/data/janghj/EasyEdit`와 HEAD3488a66ee988d83ee7891a8abbbe6bcb24a77daf는 관측값이다.
  실제 checkout에 alphaedit/memit/native/hparams 수정 및 다수 삭제/untracked가 있으므로
  이를 새 task의 승인된 baseline이라고 지정하지 않는다. 기존 bytes read-only 보존.
- `/usr/bin/python3` 3.10.12는 제어/metadata용으로 확인됐다. 새 과학 task의 interpreter/overlay는 아직 미선정.
  EasyEdit .venv/venv와 janghj miniconda3/anaconda3 후보 python은 GH 한정 점검에서 부재했다.
  다른 환경이 없다는 결론은 아니다.
- `/data/janghj/.cache/huggingface/hub` 존재만 확인했고 pinned snapshot/shards는 미결속.
- fixed10k 예정 경로는 `/data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/`.
  현재 부재하며 정확 고정순서 정책은 `plans/global/fixed-counterfact-10k-policy.md`를 따른다.
- Context/P/stats는 실제 task가 쓸 source/model/order와 결속 전이므로 승인된 공통 경로로 가정하지 않는다.
  기존 EasyEdit data/hparams/P 후보는 삭제하거나 덮어쓰지 않는다.
- 없거나 미결속인 환경/asset은 **다음 준비 계획에서 SH3가 정확 inventory/용량/원본·대상·명령 제안,
  GH가 별도 설치/선택 전송 승인**을 담당한다. 이번 요청에는 실행 승인 없다.

## 통신·보고

원 bootstrap peer nonce 확인은 유지하되 active unrelated peer를 강제로 interrupt/steer하지 않는다.
보류는 미연결 확정과 구분한다. 실제 SSH/agent ACK/rsync 준비 여부를 따로 적는다.
GH는 이번 등록 자체의 관련 M0/ACK를 수신할 수 있다. 실패한 transport는 무한 재시도하지 않는다.

허용 write: 원 `local/state/sh3-bootstrap-20260919-v1/**`,
`messages/acks/server3/2026-09-19-bootstrap.md`,
`messages/server-heads/server3/2026-09-19-bootstrap.md`,
`audits/servers/server3/2026-09-19-bootstrap/**`,
위 root/전용 bootstrap worktree의 ignored session-boundary 파일.
Raw/secret/fullstdout Git0. NO_BROADCAST_NOT_REQUIRED.

최종 ACK에 root source/clean/aheadbehind, session-helper, no-checkpoint 적용,
미결속 환경/asset, report branch/SHA를 기록한다. Control-plane 등록 완료와
과학 runtime 미준비를 분리하고, 보고 후 실험·monitoring을 자동 재개하지 않는다.
