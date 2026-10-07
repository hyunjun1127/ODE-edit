# GPT-J CAKE / AlphaEdit-BLUE / PRUNE / RECT — 등록 인계

상태는 `SUBMISSION_HANDOFF`다. 신규 네 GPU job과 CPU collector를 모두 held 검사 후 release했다. 최초 snapshot은 모두 PENDING이며 과학 2k 완료·actual B1→B2·W&B startup은 아직 미관측이다. agent 반복 모니터/자동 retry는 중지했고 등록 runner/collector는 자연 진행한다.

승인 nonce: `USER-GH-SH2-GPTJ-CAKE-ALPHAEDIT-BLUE-PRUNE-RECT-2K-20261007-R1`. [정본 계약](../../../../plans/global/2026-10-07-gptj-cake-blue-prune-rect-2k/contract.json)에 따른 신규 scoped 실행이며 다른 task·기존 job은 변경하지 않았다.

## 실제 등록

| Arm/job | ID | afterany dependency | 요청 |
| --- | ---: | --- | --- |
| CAKE | 60769 | 60656, 60657 | 1GPU / 6CPU / 59392M / 48h 상한 |
| ALPHAEDIT_BLUE | 60770 | 60656, 60657 | 동일 |
| PRUNE | 60771 | 60769 | 동일 |
| RECT | 60772 | 60770 | 동일 |
| CPU collector | 60773 | 60769, 60770, 60771, 60772 | GPU0 / 6CPU / 24576M / 4h |

등록 직전 기존 Server2 ODE-edit allocation은 60656/60657의 2GPU였다. 새 두 lane은 이 complete GPU frontier 뒤에 놓았다. PRUNE/RECT는 각각 CAKE/BLUE 뒤이며 collector는 네 부모 뒤다. afterany는 성능 조건이 아니다. job owner/source/argv/script/resources/dependency를 검사했고 export=NONE/Requeue=0를 유지했다. 최초 queue reason은 `None`으로 반환됐고 이를 다른 이유로 바꾸어 보고하지 않는다. 기존 job 취소·hold·재시작은 0이다. 48h는 ETA가 아니며 실제 신규 peak·GPU 비용은 아직 미측정이다.

## Source · 입력 · 구현

- 실행 source: `3a4a107b7ae4c66f2f7a7f0cb441d26c5a639f68`, tree `60f2be99325236e8f197f51fdd57c3d0e8760579`.
- Config SHA: `4cddaf03c88a7364e2622d1f8e3c7e0dd0d796cc545e73177f6f9951dd933790`.
- Lock SHA: `e08af8a5febe136b727eb8e8c0e01a0dfb6afecad9c23aaccaaf00a959e0d87d`.
- 원 authority publication: `0c5763837605980e1876fd50836913bbd01f47a1`. 실행 source와 후속 보고 publication은 별개다.
- 실제 host/session: server2 / `01a0493a-074c-7f91-9a13-769116326fef`. 전용 worktree: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh2-gptj-cake-blue-prune-rect-2k`.
- GPT-J revision `47e169305d2e8376be1d31e765533382721b2cc1`, FP32/eager/autocast=false/TF32=false, torch2.9.1+cu128 / transformers4.57.1.
- fixed dataset `/mnt/raid5/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json` ordered first2000, seed20261002, BS100×20/arm, 총 8000 offered edits. 새 B21/seed/sweep는 없다.
- 기존 L3–L8 C0의 FP32 mom2/count54924275, threshold.02 P6를 이전 SHA/finite/schema receipt와 현재 stat로 결속했다. 모델/C0/P 신규 생성·다운로드·복사·대형 재해시는 0이다. P physical slot=[3,4,5,6,7,8], BLUE는 slot0/5와 H2를 사용한다.
- 기존 fresh GPT-J W0 first2k raw는 exact scalar reference로 결속한다. 원 W0 비용은 신규 청구하지 않고 H/state/editor resume로 사용하지 않는다. 현재 raw chunk schema의 rows/optimizer_feedback를 보존했다.

| Arm | Layers | Native 고정값 | 완료 시 target fits / solves / H append |
| --- | --- | --- | --- |
| CAKE | L3–L8 | L2=30, temperature=.1 | 2000 / 120 / 120 |
| ALPHAEDIT_BLUE | L3,L8 | blue=true, L2=95, P slot0/5 | 4000 / 40 / 40 |
| PRUNE | L3–L8 | ordinary MEMIT C15000; terminal base fix | 2000 / 120 / 0 |
| RECT | L3–L8 | ordinary MEMIT C15000; native relative mask40% | 2000 / 120 / 0 |

표의 계수는 계약상 완료 예상치이며 실제 측정 결과가 아니다. 공통 lr=.5/최대25 forward·24 Adam/readout27/clamp.75/norm.5/KL.0625/subject_last를 유지한다. target fitting은 arm별 자기 state를 사용하며 공유하지 않는다.

원 CAKE/BLUE/EasyEdit는 읽기 전용이다. 과학 함수 AST/source SHA 및 parser 값과 작은 private import/path 수정의 diff를 결속했다. native context generator 차이와 BLUE block8 ln_1 추가 forward를 유지·계측한다. PRUNE은 원 spectral 함수의 dtype/threshold/log 식을 유지하면서 명시 `PRUNE_TERMINAL_BASE_FIX`로 finalW=coldW0+compressedD를 사용한다. W5/10/15는 dense이고 W20 current/all-seen만 terminal transform 후다. RECT의 provisional dense lower planning→restore→masked public commit과 tie 수를 유지한다.

## 검사와 관측 경계

원 CPU 회귀 72개 PASS. 마지막 telemetry·launcher 변경의 metadata 34개 PASS; 원 native/input tests는 unchanged-source receipt로 재사용했다. 이는 별개 독립 106개 검사나 Server2 실제 GPU PASS가 아니다. bounded 구현 worker 세 명과 owner 검사를 사용했고 별도 independent red reviewer는 0명이다.

실제 B1 finite/operator/materialization/H append·observer 비변이와 B2 own-entry 연결은 각 본 trajectory 안에 포함된다. 별도 small pilot/B1 refit/FD campaign은 없다. 낮은 성능은 중단 gate가 아니다. 기술 mismatch/nonfinite/IO/resource 오류는 typed failure와 원 raw를 남기며 자동 retry하지 않는다.

매 batch current/pre·post는 R100/P200/N1000, W5/10/15/20 all_seen은 실제 prefix이고 최종 R2000/P4000/N20000이다. first500/birthcohort/at-write retention·paired와 TF token-micro/prompt-macro/strict, true/new NLL nats·margin·harmonic을 저장 row로 집계한다. N desired=true, R/P desired=new이다. 현재 신규 지표는 미측정이다.

W&B entity=`wkdguswns2256`, project=`layer allocation`, online/scalar-only. actual job ID/run.name 및 immutable source/config/model/writer를 runner 내부 init에서 결속한다. fit/global_candidate는 평가 edits 축과 구분한다. 기존 shared helper/PRICE mapping을 읽기 전용으로 재사용하고 raw prompt/token/tensor/code/stdout/credential을 업로드하지 않는다. 현재 startup/run URL은 미관측이며 SDK 접수≠remote readback≠과학 완료다.

## 저장 · 실행 · 인계

NoCP, z disk cache=None, checkpoint_saved=false, exact_resume=NOT_AVAILABLE. PRUNE coldW0/SVD 및 transaction rollback은 RAM only다. 기존 CP/raw/source/model/dirty root는 보존한다.

Authoritative local root: `/mnt/raid5/janghj/ODE-edit/local/gptj-cake-blue-prune-rect-2k/attempt-r1/`. source archive·config·lock·held inspection·submitted/released receipts와 향후 arm raw/collector 결과는 이 root에 있다. 입력 준비는 sibling `preparation-r1/`, CPU 원 로그/receipts도 local이다. 신규 raw/tensor/전체 stdout Git0, `NO_BROADCAST_NOT_REQUIRED` — same-host 입력 재사용과 원본 local 보존이다.

재현 entrypoint는 [README](../../../../project/run_scripts/gptj_cake_blue_prune_rect/README.md)에 있다. 실제 prepare → CPU tests → source commit → submit 순서를 사용했다. 모든 job의 full argv/launcher SHA는 local held receipt와 [소형 등록 manifest](../../../../audits/servers/server2/gptj-cake-blue-prune-rect-2k/submission.json)에 연결된다.

CPU collector는 저장 rows·count·state hash를 독립 reducer로 검산하고 partial/FAILED prefix를 보존한다. collector 성공이 네 arm의 과학 완료를 뜻하지 않는다. 현재는 등록 인계만 완료했으며 후속 상세 결과 리뷰는 사용자 recall 시 수행한다.
