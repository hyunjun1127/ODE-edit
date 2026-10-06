# GPT-J PRICE 6-arm — 제출 준비

사용자가 W&B 비교 요건 반영 후 제출을 다시 승인했다. 대상은 MEMIT/AlphaEdit 각각 CAP075, CAP100, FREE100의 독립 cold first2000, BS100×20이다. **아직 Slurm 제출 0건**이며 actual B1/W20은 NOT_OBSERVED다.

## 구현과 확인

- EasyEdit GPT-J 모델·native stats·projector·hparams를 결속하고 native parallel attention/MLP·fc_out bias·readout27에 ours hook을 적용했다. 기존 과학 계수와 cap 설정은 유지한다.
- native context 5개는 첫 실제 arm의 입력 준비에서 한 번 생성·고정한다. 별도 fit/pilot 없음. GPT-J exact W0가 없어 첫 arm에서 first2000을 관측하고 나머지는 identity가 일치할 때만 재사용한다.
- 기존 60001의 봉인 B4/B5/W0 raw CPU 검산에서 current 분모 R100/P200/N1000, B5 누적 R500/P1000/N5000, W0 R2000/P4000/N20000을 확인했다. 원 raw/stored aggregate/payload 일치. B4에는 all-seen key 없음. 기존 job이나 run 변경 없음.
- 새 caller는 scientific run에 current/pre·current/post·실측 all_seen/post·W0_first2000 및 W0 같은-cohort N 비교를 직접 기록한다. R/P/N 경로 분리, 백분율, harmonic, true-new margin, edits 축과 pre-state 위치를 구분한다. 중복 companion daemon을 자동 실행하지 않는다.
- SH1 job-identity helper `bc63425e` 채택. 실제 job 이름/Config 검증을 유지한다. immutable startup identity와 transport 상태를 분리하고 finish 후 bounded readback을 준비했다. 실온라인 검증은 NOT_OBSERVED.
- CPU fake-SDK/metadata 검사 25개 중 24 PASS, 1 SDK환경 검사 SKIP 후 task-local readback 검사 1개 추가 PASS. 소스 31개 AST/import/config 검사 통과. 이는 과학 toy/GPU/model PASS가 아니다. owner audit이며 별도 reviewer 없음.

## 남은 제출 조건

SH1 소유 shared helper의 비교 metric/config whitelist 및 axis capability 확장이 아직 필요하다. 현재 helper에 새 key를 넣으면 거부되므로 `freeze()` 시작에서 제출을 차단한다. SH4는 공통 helper를 수정하지 않았다. 정확 요청은 [GH/SH1 전달문](../../../../messages/server-heads/server4/wandb-job-id-recording.json)에 기록했다. 사용자가 GH에 전달하겠다고 답했다.

helper 도착 후 caller/schema 원자 검산, 최종 source/config 재봉인, fresh cap3·task2·storage/resource admission, 여섯 GPU job과 GPU0 collector의 held 검사/release가 남아 있다. 과거 임시 freeze는 stale evaluator SHA로 **Slurm 호출 전** 실패했으며 원본은 보존했다. 최신 tracking source는 아직 실행 봉인되지 않았다.

noCP/exact resume NOT_AVAILABLE. 기존 Qwen STOP·Llama job 보존. 원자료/credential/model/SDK spool Git 업로드 없음. NO_BROADCAST_NOT_REQUIRED: 소형 source/receipt만 공유한다.
