# Server2 official 코드 동기화 적용

Nonce: `GH-SH2-OURS-HPARAMS-MAIN-SYNC-20261009-R1`.

- 준비 checkout: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh2-cf-six-checkpoint-only-20261009`
- branch: `codex/server2-cf-six-checkpoint-only-20261009`
- 적용·검증 HEAD / fetched main: `67d00d4b166a9a186c898dec49f369ad081a875a`
- official tree: `927c6e237d0a1338f387b7a49dedad8fb8cc94c3`
- `cc2d7190e4860ae2934be86f615fef37e84ca4ba` 및 fetched main ancestor 모두 PASS.

원 own commit을 보존하는 fast-forward로 적용했으며 충돌/미커밋 변경은 없었다.
지정 보고서, 세 모델 writer/price JSON, schema 및 resolver를 읽었다.
`python3 -m official.tools.verify`: source157 SHA PASS, Python233, external task imports0.
기존 EasyEdit 환경에서 GPU를 숨기고 `official/tests` **70 PASS**. 설치·환경교체 없음.
실제 pretrained/GPU/W&B online PASS가 아니다. GPU-SMOKE-PLAN은 읽기만 했다.

이번 작업에서 Slurm 조회/등록/취소0, 온라인 W&B 검사0, 공통 source/README 수정0.
dirty root와 frozen runtime/source/config/archive, 모델·자산·raw·CP는 그대로다.
직접 USER server2 cap3와 CF FLU/CON 연기 준비도 보존했다. 이전 공유 tracking schedule
미지원 차단 요인을 이 코드 동기화만으로 해제하거나 실제 job 제출로 표시하지 않는다.

이 영수증은 own branch에 별도 커밋한다. 준비 worktree 최신화와 main 추가 게시는 별개이며,
이번 동기화에 추가 main push나 과학 실행은 하지 않았다.
