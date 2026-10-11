# Llama3 zsRE 2K — 수정 FE + author hparams + history

사용자 요청으로 **63357 / official-s4-llama3-zsre-fe-author-hparams-2k** 한 개를 server4에 제출했다.
실제 held script·source·config·자원을 검사하고 release했다. 최신 사용자 정정 **“아니 이거 cap 2에 맞추자.”**에 따라
같은 63357을 잠시 hold하고 `afterany:63284`로 변경한 뒤 다시 release했다. 신규 제출·취소·실험 설정 변경은 없다.
GPU 1개, CPU 8개, RAM **58 GiB**, 48시간, `lab_gpu_s4`, no-requeue다.

현재 **PENDING / BLUE 63284 종료 후 실행**이다. 두 lane은 다음과 같다.

- MEMIT **63283** → AlphaEdit **63262**
- BLUE **63284** → 새 FE **63357**

가능한 시작·종료 순서 16개 상태를 열거해 동시 project GPU **최대 2개**를 확인했다.
기존 두 replay 및 AlphaEdit의 의존성은 보존했다. server4의 local cap과 오래된 clone의 정책값도 2로 맞추고,
새 `official-*` job이 admission 계산에 포함되도록 local pattern을 추가했다.
실제 cap 검사에서 현재 2개 사용 중 추가 1개에 대해 `DENY pending_resource_cap`을 확인했다.
GPU 실행·W&B 온라인·2K 결과·실제 실험 checkpoint 생성 완료는 아직 관측하지 않았다.

최초 등록 당시 dependency 없는 cap-3 영수증은 이력으로 남기며, 현재 배치는
[cap-2 변경 영수증](../../audits/servers/server4/llama-zsre-fe-author-hparams-20261011/cap2.json)과
[로컬 설정·admission 검증](../../audits/servers/server4/llama-zsre-fe-author-hparams-20261011/cap2-settings.json)이 우선한다.

## 실행 버전

기존 **62530**의 수정 EasyEdit `MEMIT_FE_HISTORY_FE_AUTHOR_HPARAMS` Llama zsRE 조건을 cold W0/H0에서 다시 실행한다.
FE 저자 repo `fe_original`/W0 전체2K z 고정 경로와 구별하여 README에 별도 행을 만들었다.
저자 repo 완료 수치는 해당 기존 행에 보존한다.

- 코드: `official.baselines.memit_fe_history`와 vendored EasyEdit FE.
- 원저자 hparam overlay: **clamp 0.75, 35 steps**. 기존 profile의 lr 0.1, decay 0.5,
  layers 4–8, loss layer 31, mom2 weight 15000, KL 0.0625를 그대로 사용한다.
- history **ON**: native `lambda_C*C0 + H_entry + K@K.T`, FP64 solve, final-model mean key를
  성공한 batch마다 layer별 한 번 누적한다. native 편집·history 수치 함수는 변경하지 않았다.
- 각 batch 진입 모델에서 해당 100건의 z를 새로 계산한다(`cache_template=None`).
- first-2K zsRE, batch 100 × 20, seed 0, FP32/eager, TF32 off.
- context는 이전 config와 같은 cold-native generation 정책이다. 사전 생성 context를 새로 넣지 않는다.
  이전 server1 context SHA는 `66384123…`였으며, GPU 환경이 달라 새 context/출력의 bitwise 일치는 주장하지 않는다.
- 평가는 W0 및 W5/W10/W15/W20 prefix의 공개-query zsRE request-macro Eff/Gen/Loc,
  Loc은 loc_ans 정답 비교다. zsRE에는 CF FLU/CON이 없다.

원 config SHA `70b5d963836a778ccc0f64e1919897a60a08a9280642bc83e582825ab0a9ee55`,
원 source `9c3fe23282bcaf2e353b1494913a323e2f037d39`.
핵심 FE/history/profile/context-generator 파일 7개는 새 source와 **바이트·SHA 동일**하다.
새 config SHA `b35dde639824300ece65b38b518e7a112655f0c1600b83f5c077e2c043a53dc2`,
실행 source `c16c9612f49dec543fcc602efd64a4edb74b7f8a`는 제출 전에 main에 게시했다.

## 저장과 검증

server4 디스크 여유는 준비 시 약 21.5 GiB여서 새 tensor payload는 server1으로 직접 전송한다.
매 100 edit의 최신 checkpoint를 원자적으로 교체하고, W0 및 500/1000/1500/2000 state를 별도 보존한다.
송신/수신 SHA와 수신 파일 전체 SHA 확인 후 commit한다. 전송 중 실패하면 이전 checkpoint가 남는다.
ACK 유실 및 중간 batch 재개도 지원한다. 보존 state에는 편집층 가중치·H·context·RNG·평가 cursor가 들어간다.

server1 archive:
`/mnt/raid5/janghj/ODE-edit/local/llama-zsre-fe-author-hparams-s4-20261011/checkpoint-archive`

- `checkpoint/`: 매 100 edit의 최신 native checkpoint.
- `milestones/edit-0000`, `edit-0500`, `edit-1000`, `edit-1500`, `edit-2000`: 영구 보존.
- `states/batch-XX`: 최신 non-milestone 상태 하나. 다음 commit 이후 교체한다.
- server4 `runs/zsre/checkpoint`와 `milestones`: remote 위치 및 SHA 포인터.

checkpoint 한 개 예산 4.95 GiB, 원자적 쓰기 중 최대 6개와 server1 reserve 64 GiB를 포함하여
약 93.72 GiB를 요구했고 준비 시 약 309.97 GiB를 확인했다. server4에는 raw 2 GiB·다른 작업 증가 8 GiB·여유 8 GiB를 확보했다.
기존 모델/C0를 그대로 재사용하고 전체 SHA 및 C0 shape/count/finite 검사를 통과했다. 모델·C0 복사나 재계산은 없다.

최초 96 GiB 메모리 요청은 server4의 GPU당 60 GiB 정책으로 **job 생성 전에 거절**되었다.
이전 동일 FE 실행 62529/62530의 최대 RSS 약 24.7/24.6 GiB를 확인하고 **58 GiB**로 맞춰 63357 한 개를 등록했다.
거절 영수증은 보존했고 중복 GPU job이나 자동 재시도는 없다.

CPU 회귀 **46개 PASS**: author profile, history solve/누적, public zsRE dispatch,
checkpoint의 매 batch 복원·4개 milestone 보존·손상/ACK 유실 회복 및 기존 CAKE/local checkpoint 회귀.
2,000건 **24,535 query**를 공개 evaluator AST와 비교하여 input/target 불일치 0이다.
실제 server4→server1 SSH B0/B1 저장·SHA 확인·rsync 재개도 synthetic tensor로 통과했다.
pretrained GPU qualification은 사용자 방침에 따라 실행하지 않았다.

[CPU 자산·설정·query 검증](../../audits/servers/server4/llama-zsre-fe-author-hparams-20261011/cpu-preparation.json) ·
[실제 제출과 held 검사](../../audits/servers/server4/llama-zsre-fe-author-hparams-20261011/submission.json) ·
[최초 scheduler 관측](../../audits/servers/server4/llama-zsre-fe-author-hparams-20261011/observed-63357.json).
