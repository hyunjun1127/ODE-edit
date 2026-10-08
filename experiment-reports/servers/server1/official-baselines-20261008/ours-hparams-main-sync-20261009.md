# SH1 ours hparams main 동기화

nonce `GH-SH1-OURS-HPARAMS-MAIN-SYNC-20261009-R1` 실제 적용 완료.
준비 WT `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-official-baselines-20261008`,
branch `codex/server1-official-cf-checkpoint-20261009`에서 clean merge했다.
적용 HEAD `adc1bda2ef755adcad6af7317b9e596d46e6859d`, official tree
`7384f9bd9d7383a1d60df30f35e19733d8a8acc6`.
fetched main `67d00d4b166a9a186c898dec49f369ad081a875a`와 필수
`cc2d7190e4860ae2934be86f615fef37e84ca4ba` 모두 ancestor PASS.
기존 own source `db56d7c0`도 ancestor PASS이며 merge 충돌0이다.

지정 README, 세 모델 writer/price, schema 및 official.ours.config를 전부 읽었다.
baseline 및 legacy PRICE JSON은 bytes 불변. GPU smoke 문서는 계획이며 실행하지 않았다.
기존 환경에서 CUDA_VISIBLE_DEVICES=''로 source verifier 157 SHA/Python257/
external task imports0 PASS, official/tests 113개(76.791초) PASS.
이는 실제 GPU 또는 online 검증이 아니다.

원 dirty root, 이미 등록된 job의 source/config/archive/runtime, asset/raw/CP/model은
변경하지 않았다. Slurm/W&B online 작업0. server1 cap4와 W20 FLU/CON 연기 수정은
보존했다. 이후 sourcefreeze는 이 통합 이후 exact source/config SHA를 별도 기록한다.
준비 WT 동기화 완료이며, own 미통합 W0/runner source의 main 게시 완료를 뜻하지 않는다.
이번 동기화는 main을 push하지 않는다.
