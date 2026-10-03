# SH1 실행 지시: JLZ v10 A의 B1 단일 fit 진단

Instruction: `ODEEDIT-USER-GH-SH1-JLZ-V10-A-B1-DIAGNOSTICS-20261003-R1`

사용자 원문: “이 부분 실험이 필요하다. SH1에게 task 진행하도록 전달하자. batch 1개만 실험하는 것을 보자”. 첨부 리뷰 §5 D1·D2를 **B1 fit 한 번**으로 확인한다. 해당 리뷰의 원인 확정 표현은 검증 대상 가설이며 실행 결과를 미리 정하는 지시가 아니다.

## 범위와 담당

- GH가 이 지시를 SH1/server1에 전달하고 SH1의 직접 수락 ACK를 회수한다. 이전 모든 실험 중지 지시의 예외는 이 신규 진단 task뿐이다.
- **v10 T′ arm A, BS100, 기존 fixed order 첫100, W0 cold start/H0, 1 batch, 1 scientific fit, 25 candidates/24 Adam updates.** B2·arm B·F1/F2/F3 학습 변경·계수 sweep·baseline 재fit·reference 추가는 이번 범위에 포함하지 않는다.
- 대상 모델은 원 v10과 동일 Llama3-8B-Instruct revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`. 모든 edit layer L4–L8을 사용한다.
- 원 실행 source `c2d5fb107a0435491d8b4705b43b75f6177bbb5c`, `project/run_scripts/jlz_realized_subject/`. Frozen source와 원 input/runtime metadata를 읽기 전용 재사용한다.
- native 문장·context·token·lookup·mean key·KL current∥entry·NLL readout·norm0.5·allocation0.1·KL0.0625·ridge15000·q-scale·Adam·예산·terminal25 반환을 보존한다. Warm-up/clamp/pulse/E/replay를 새로 넣지 않는다.
- 사용자 요구는 연구 진단 실행이다. 기존 stop 기록이나 과거 NOT_RUN 상태를 신규 권한 부재로 해석하여 재승인을 요구하지 않는다. 무관한 실험 재개·취소는 허용하지 않는다.

## D1: 후보별 실제 R/P/N 궤적

1. 단일 원래 fit에서 **c9,c13,c17,c21,c25**의 evaluated materialized FP32 W_eff를 graph와 분리해 CPU RAM에 복사한다. 모델 전체 checkpoint 대신 편집한 projection weight만 보관하면 약5.5GiB 수준이며 실제 bytes/RSS를 기록한다.
2. 후보는 `candidate-c` = `c−1` updates 후, 다음 Adam update 전이라는 원 정의를 유지한다. 평가를 위해 추가 Adam update를 하거나 후보를 다시 최적화하지 않는다.
3. Fit 종료 후 각 snapshot을 같은 W0 모델에 임시 적용해 원 evaluator로 **R100/P200/N1000**을 모두 평가한다. W0도 같은 실행 환경에서 한 번 평가한다. 기존 v9/v10/Baseline B1 결과는 역사 참고로 재사용하며 새 baseline fit은 없다.
4. Raw paired IDs, new/true NLL, preference, TF strict, NLL margin, v/a·Q·층별 normalized share, 전체/성분별 loss와 q/R gradient를 기록한다. R/P는 new NLL<true NLL, N은 반대이며 tie는 실패다.
5. 평가 결과를 optimizer, stopping, coefficient, candidate selection으로 반환하지 않는다. 공식 terminal은 c25 그대로다. c9..c21은 관측용 경로다.
6. W/H/optimizer/RNG를 관측이 바꾸지 않았음을 확인한다. Snapshot별 적용은 누적 덧셈이 아니라 정확한 W0→W_eff 교체다. Probe마다 H를 append하지 않는다. 정상 terminal history commit을 검산한다면 전용 ephemeral state에 정확히 한 번만 하고 원본 main을 변경하지 않는다.

보고 질문: PS/target NLL 개선이 어느 구간에서 작아지며 Q와 NS 손상이 어떻게 변하는가? “과잉 진행” 가설을 지지/반박/불충분으로 판정한다. 다섯 점만으로 전체 목적의 균형 부재나 lifelong 최적 강도를 확정하지 않는다.

## D2: 동일 terminal writer의 위치별 개입

c25의 **동일 U/W_eff를 고정**한다. Mask별 key로 writer를 다시 풀거나 R을 refit하지 않는다. 모델 activation/key는 각 마스크의 causal forward에서 자연스럽게 달라지도록 둔다.

네 경로를 같은 이웃 문장에서 비교한다.

| 경로 | Neighbor subject lookup token | 나머지 valid token |
|---|---|---|
| NONE/W0 | W_entry | W_entry |
| SUBJECT_ONLY | W_eff | W_entry |
| NONSUBJECT_ONLY | W_entry | W_eff |
| ALL | W_eff | W_eff |

- Mask는 모든 L4–L8에 동일한 위치 정의를 사용한다. Subject는 **이웃 문장의 subject**이며 편집 요청의 subject를 억지로 찾지 않는다. Native lookup 계약과 같은 마지막 subject token 하나를 사용한다. Padding은 제외하고 원 teacher-forced target-prefix token의 유효 위치 정의는 명시한다.
- 원 데이터의 subject metadata 또는 relation template과 정확히 대응하는 문자열에서 subject span을 결정하고, 원 token IDs/offset/native token locator에 결속한다. 값/성능을 보고 span을 고르거나 tokenizer를 달리하지 않는다. Subject를 신뢰성 있게 찾지 못하는 행은 조용히 first/last-token으로 대체하지 않는다.
- Span/lookup/모호성 목록을 관측 전에 확정한다. D1은 항상 N1000 전체를 유지한다. D2에 식별 불가 행이 있으면 네 경로 모두 **동일한 식별 가능 subset**에서 평가하고 coverage, 제외 ID/이유, 전체 ALL NS와 subset ALL NS를 따로 보고한다. 이 경우 N1000 전체의 경로 attribution이라고 주장하지 않는다.
- `U k`를 별도로 더하는 근사 대신 `F.linear(k,W_entry)`와 `F.linear(k,W_eff)`의 **행 선택**을 이용해 원 materialized FP32 산술을 보존한다. ALL은 정상 actual evaluator, NONE은 W0와 일치해야 한다.
- N의 new/true NLL과 margin을 primary continuous 지표로, NS·lost/gained·TF strict를 함께 기록한다. 같은 c25 ALL·W0 관측은 D1 결과를 재사용할 수 있다.
- 네 결과는 2×2 위치 개입이다. 비선형 때문에 SUBJECT_ONLY+NONSUBJECT_ONLY=ALL이 아니다. Margin에 대해 `interaction = margin_ALL−margin_SUBJECT−margin_NONSUBJECT+margin_NONE`도 보고한다. 이를 원 모델 손상의 유일한 가법적 원인 분해라고 부르지 않는다.

보고 질문: 비subject-only 개입이 얼마만큼 손상을 만들며, subject-only 및 두 경로의 상호작용과 비교하면 어떤가? 이웃을 학습 loss나 모델 선택에 사용하지 않는다.

## 구현·자원·검증 계약

- SH1 session: `01a04939-f93a-7b50-bca0-65438eab2062`; GH: `01a04939-8873-7673-8dca-4c7fc5e31af0`; repo `hyunjun1127/ODE-edit`. App의 live CWD와 오래된 registry의 CWD가 다를 수 있으므로 session/repo 검증 후 전용 non-main `codex/` worktree를 만들고 boundary를 결속한다. 모델/profile은 사용자 설정을 따른다.
- SH1 write ownership: 신규 `project/run_scripts/jlz_realized_subject_diagnostics/**` 및 그 tests; 원 namespace 변경이 필요하면 전용 branch의 **snapshot callback/observer instrumentation만** 허용하고 source diff로 계산 불변을 증명한다. EasyEdit·기존 sealed source·다른 session 변경을 덮어쓰지 않는다.
- Raw output: ignored `local/jlz-v10-a-b1-diagnostics/20261003-v1/**`. Reports: `experiment-reports/servers/server1/jlz-v10-a-b1-diagnostics-20261003-v1/**`; SH1 status/audit는 해당 task 전용 이름을 사용한다.
- Slurm 제출 **허용**. Task 동시 GPU cap1, CPU8 기본, host RAM96GiB 기본, GPU walltime6h 상한. GH/SH1이 최신 server1 project cap을 확인해 더 엄격한 한도를 적용한다(기존 cap 증액 없음). 메모리·queue cap 밖이면 `RESOURCE_PENDING`, 무관한 job 취소나 CPU에서 model fit 우회 없음.
- FP32 model/FP64 geometry 유지, TF32 off. GPU 메모리상 MB1이나 observer chunk 축소는 whole-B solve와 loss normalization을 유지한 기술 조정으로 가능하며 기록한다. Scientific batch100·context·vocab·평가 표본 축소는 하지 않는다.
- 사전 CPU 검증: 네 마스크의 complement/identity, neighbor lookup coverage, row/target identity, snapshot 적용·원복, 원 reduction/adjoint 유지. GPU technical check는 같은 allocation의 고정 후보에서 dense/direct 또는 원 route와 instrumented route의 필요한 parity만 확인하며 별도 scientific fit으로 늘리지 않는다.
- Finite/shape/identity/restore/ALL-NONE mask parity 오류는 중단. 수치 비교는 v10 기존 명시 tolerance와 scale을 사용하고 실패 기준을 사후 변경하지 않는다. 낮은 NS, 균등 share, 예상과 다른 부호 같은 **과학 결과는 중단 gate가 아니다.**
- 원 H200/Transformers4.44.2와 SH1 runtime 차이를 명시한다. 가능한 한 pinned runtime을 재사용하고, c25의 역사 v10 B1 대비 parity는 수치/성과 차이와 환경을 함께 기록한다. 다른 GPU에서 성과가 완전히 같아야만 통과한다는 gate는 만들지 않는다.
- Git은 SH1 소유 source·tests·compact report의 commit/비강제 push 및 GH 정본 통합을 허용한다. Raw per-case 텍스트/모델/dataset/weight snapshot은 Git 제외. Snapshot은 RAM만 사용하고 실행 후 해제한다.
- Artifact broadcast는 현재 protocol의 `scripts/rsync-artifact-broadcast.sh`로 승인된 ODE-edit peer raw 위치에 수행하며 삭제 sync는 금지한다. 불가 시 이유와 GH가 접근 가능한 로컬 경로를 exception receipt에 남긴다. Source/docs/기존 input manifest 및 seed/context cache의 필요한 재사용 전송은 이 task 범위에서 허용하며 새 모델·C0 재다운로드/재계산은 하지 않는다.
- SH1은 접수 nonce ACK → 구현/technical check → Slurm job ID 또는 resource-pending → 완료 결과를 구분한다. B1 평가 후 종료하며 B2/추가 arm/자동 과학 재fit으로 확장하지 않는다. 작업에 필요한 초기 제출 상태 확인과 완료 회수는 허용하되 새 반복 automation은 만들지 않는다.

## 첨부 리뷰 해석의 교정

- 1차 동차 norm도 task와 합치면 내부 최적점을 가질 수 있다. “크기를 정하는 장치가 없다”는 정리가 아니라 현재 계수·optimizer·예산에서의 관측 가설로 둔다.
- R-gradient 크기만으로 q-Adam의 영향이나 배분 무효를 확정하지 않는다. q 좌표·성분 방향·실제 step을 함께 본다.
- Raw Q의26 대99는 anchor normalization을 거친 정책 비용 차이가 아니다. `c_l=√(Q_l/(Bσ_l²))` 및 실현량/방향을 함께 비교한다.
- 기존 subject 기저 gap·key 상대차는 평균/분포를 구분한다. 작은 key norm 차이만으로 출력 영향 경로를 배제하지 않는다.
- D1·D2만으로 “object 수준 편집”이나 lifelong 원인을 확정하지 않는다. D3/F1/F2는 다음 설계 후보이며 이번 실행 대상이 아니다.

## 전달 근거

- 원 첨부: `/mnt/raid5/janghj/.codex/attachments/051f6b4b-3b18-435b-acdf-82369366c985/붙여넣은 텍스트.txt`.
- 독립 감사: `experiment-reports/global/2026-10-03-jlz-v10-a-review/report-ko.md` 및 `code-audit/v9-v10-code-delta-ko.md`, `metrics-audit/v9-v10-report-ko.md`.
- Local frozen source: `local/jlz-v10-a-review/20261003-r1/snapshot/attempt-v2/source/`; 원격 root는 server3 `/data/janghj/ODE-edit/local/jlz-realized-subject-v10/20261003-v1/attempt-v2`.
- `input-b1.json`은 원 v10 A B1의 정확한100 IDs와 input identity이며 후보 성능으로 표본을 바꾸지 않는다.
