# JLZ native BS100 10回 누적 편집 실행 계약

Instruction `ODEEDIT-GH-SH1-JLZ-BS100X10-20261001-R1`. 사용자 handoff `ODEEDIT-JLZ-BS100X10-TO-GH-20261001-R1`.
이 실험은 **고정 첫1000 요청을 BS100×10으로 같은 W/H에 누적**하는 한 경로다. 기존 pilot의 수렴 전용 commit을 새 사용자 지시의 고정예산 반환 후보 commit으로 바꾼다. 원 pilot 설계·결과는 수정하지 않는다. 서버3 MEMIT-HJ와 독립이며 신규 baseline·sweep·Qwen 실행은 없다.

## 방법과 입력

Llama3-8B-Instruct revision8afb486c1db24fe5011ec46dfbe5b5dccdb575c2의 L4–L8 down_proj만 공동 최적화한다. 동일 fixed10k 순서 앞1000, seed20261001, 요청 및 context hash는 contract.json과 cells.csv에 고정한다. W0와 zero FP32 H에서 한 번 시작하고 B2부터 자기 직전 누적 상태를 사용한다.

각 batch entry에서 native 6-context NLL, layer block-output anchor, target-prefix 없는 key와 KL teacher를 포착한다. Key는 group 평균의 평균(.5 + .1×5), NLL은 target-token 평균 후 여섯 context 평균이며 활성 요청 간 합이다. Teacher/key/adj는 최적화 중 고정한다. KL은 `KL(current || entry)`, 계수.0625다. Norm decay .5||R||/||anchor||²와 clamp .75||anchor||, entry NLL<.05의 고정0 block을 유지한다. 비활성 요청의 key는 Gram에 남고 평가 분모에도 남는다.

C0는 npz raw moment/count를 **FP32에서 나눈 뒤 FP64**로 승격한다. `adj=solve(15000*C0+H+KK.T,K)`; `W_eff=W_entry+(R.double()@adj.T).float()`. 각 batch에 다섯 adj solve, FP32 W/activation/R, FP64 solve, TF32/autocast off다. Inner trajectory는 physical W/H를 바꾸지 않으며 native z fitting을 별도로 먼저 하지 않는다.

## 고정 예산과 commit

Optimizer는 **batch당 최대120 whole-batch oracle**이다. 초기·수락·탈락·최종 반환점 재평가를 모두 포함한다. 전체 본실험 상한1200이며 prompt microbatch 호출 수와 구분한다. 최종 재평가1회는 반드시 예약한다. tol1e-4는 수렴 표시용 임시 기준이지 production 교정값이 아니다. 목적·예산·tol을 결과에 맞춰 바꾸지 않는다. Pilot의1200초 batch/3300초 process soft limit은 규모가 달라 상속하지 않으며 새 batch wall stop을 과학 예산 대신 사용하지 않는다.

CONVERGED/POLICY_ZERO_STEP뿐 아니라 BUDGET_STOP, STALLED_AT_PRECISION, NOT_CONVERGED, LINESEARCH_FAILED도 **반환점이 마지막 수락점 또는 초기점이고, 새 평가가 완결·finite·feasible·commit 정합을 통과하면 반영하고 다음 batch로 간다**. 탈락 trial을 반영하거나 수렴 PASS로 이름을 바꾸지 않는다. 수락점0/zero update도 별도 표시하고 history는 규약대로 append한다. 어떤 trial에서든 NONFINITE가 발생했다면 뒤의 finite 재평가로 숨겨 계속하지 않는다.

반환 평가가 사용한 실제 FP32 W_eff 다섯 개를 그대로 commit하고 full-forward NLL/KL 및 L4 key 불변을 확인한다. 그 뒤 현재 모델의 post-write key로 H를 층마다1회 append한다. 후보·probe·observer는 append하지 않는다. 기술 실패는 W/H/RNG/context/ledger를 entry로 rollback하고 evidence를 보존한다. Commit 이후 observer/IO 예외까지 transaction에 포함하거나 이미 durable commit된 상태를 정확히 구분해 다음 batch를 중단한다. H만 또는 W만 남는 partial commit, 중복 append는 금지한다.

## 실제 BS100 최소 확인과 효율

기존 run.py는 BS2/첫4개 전용이며 Oracle이 microbatch마다 W_eff와 autograd를 다시 만든다. 새 namespace에서 BS100×10 runner, metrics, transaction을 구현한다. 같은 oracle에서 한 번 materialize한 weight의 재사용, GPU gradient 누적, 필요한 prediction/KL 위치만의 full-vocabulary head를 검토한다. Teacher/head dtype·FP64 matmul→FP32 cast→FP32 add 순서·global loss 가중치·입력 길이를 바꾸지 않는다. Prefix는 batch entry L4 입력까지만 재사용하며 다음 batch에서는 새로 만든다.

CPU 단위·실제 production 함수 통합 검사 후 실제 BS100 한 entry에서 고정 nonzero feasible R의 legacy full/suffix와 새 경로 loss/gradient를 대조한다. 추가 whole-batch oracle는 합계 최대6회이며 science120회에 숨기지 않는다. 이전 BS2 PASS는 BS100 PASS가 아니다. 동일 W0 entry에서 probe 후 상태를 복원·결속하고 본 B1에 entry/key/adj를 재사용하여 불필요한 native fit이나 과학 batch 반복을 하지 않는다.

고정 허용치는 loss abs1e-3, grad relative1e-3(분모max(1,||g_ref||)), per-request NLL/KL abs1e-3, commit NLL/KL abs1e-3, exact W_eff 및 L4 key, clamp slack1e-6, adj solve relative1e-7이다. 검사 전 봉인하며 실패 후 완화하지 않는다. 광범위 FP64 교정·새 gate 캠페인은 추가하지 않는다. Microbatch2를 출발점으로 2/4/8 중 기술 시간·메모리·고정 parity로만 정하고 첫 과학 fit 전에 동결한다. 자원 최적화가 동일성 검사를 통과하지 못하면 pilot 경로 microbatch2로 돌아갈 수 있으나 BS100·목적·예산은 바꾸지 않는다.

## 평가와 보고

최적화 입력과 official observer를 분리한다. W0에서 같은1000의 R1000/P2000/N10000을 평가한다. 각 batch Current R100/P200/N1000과 native6-context fit을 남기고, all-seen R/P는 매 batch, all-seen R/P/N은 W5와 W10에서 평가한다. 동일 endpoint/문항 관측은 재사용해 중복 forward를 피한다.

R/P는 new NLL<true NLL, N은 true NLL<new NLL, tie는 실패다. R/P는 target_new, N은 target_true의 TF token-micro/prompt-macro/strict 및 true/new/desired NLL을 별도로 보고한다. TF를 자유생성이라고 부르지 않는다. W5 first500→W10 same500, at-write→W10, W0-correct N 유지, active/superseded를 paired lost/gained로 분리한다. 모든 요청은 분모에 유지한다. 기능 점수를 continuation gate로 쓰지 않는다.

10개 batch의 solver status/commit authorization, 호출·accepted/rejected·잔차·KKT·활성 block·clamp·층별R/실현norm, W/H hash와9개 연결, 모든 stage 시간·token work·memory·I/O·allocation을 보고한다. 호출1200은 single-request native1200회와 같은 비용이 아니다. BS2 pilot225GPU초는 과거 비용이고 새 비용과 분리한다.

## 실행과 종료

Server1 devbox의 본task1GPU/8CPU/131072MiB, projectcap2, host hardceiling183296MiB, exportNONE/Requeue0. 최초wall48h는 예약상한이지 ETA·GPUh hardbudget이 아니다. SH1은 실제 cap·partition·disk 및 BS100 시간/peak를 확인하고 예상 총시간을 기록한다. 기존 job과 server3는 조회·취소·변경하지 않는다(본 서버 admission resource-only는 허용).

새 checkpoint·전체W/H/R/Δ/optimizer resume bundle 저장0, RAM transaction만 허용한다. 원 자산은 KEEP, exact crash-resume NOT_AVAILABLE. 이 실험에는 MEMIT-HJ의 임시CP 예외가 없다. Raw scalar/per-case/source/config/ledger는 local에 보존하고 Git에는 코드·소형 표·요약·hash만 게시한다.

SH1은 한 GPU persistent runner로 최소 preflight와10batch를 연결하고 CPU afterany collector로 실패/완료 자료를 수집한다. Source freeze→held owner/argv/resource/dependency 검사→release를 소유한다. 대표 B1 실제 commit/history/observer→B2 entry까지 bounded 확인한 뒤 agent monitoring을 pause한다. 자원부족으로 정상 released job이 pending이면 근거와 미관측을 남겨 먼저 인계할 수 있다. Runner는10회와최종평가/collector를 자연 진행한다. 완료 상세 리뷰는 사용자 recall 때 수행하며 GH는 초기 수신 ACK만 회수한다.
