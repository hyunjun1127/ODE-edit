# Server2 GPT-J CAKE / AlphaEdit-BLUE / PRUNE / RECT

승인 `USER-GH-SH2-GPTJ-CAKE-ALPHAEDIT-BLUE-PRUNE-RECT-2K-20261007-R1`의
전용 실행 profile이다. 원 CAKE/BLUE/EasyEdit는 읽기 전용이며 네 arm은
각각 cold GPT-J W0에서 fixed-order first2000, BS100×20을 처리한다.

| Arm | Physical layers | 배치당 target / solve / H append |
| --- | --- | --- |
| CAKE | L3–L8 | 100 / 6 / 6 |
| ALPHAEDIT_BLUE | L3, L8 | 200 / 2 / 2 |
| PRUNE | L3–L8 | 100 / 6 / 0 |
| RECT | L3–L8 | 100 / 6 / 0 |

CAKE의 H6/P6와 BLUE의 독립 H2/P physical slot0,5를 구분한다.
원 native25 evaluations/24 Adam updates 및 GPT-J parser/문맥 생성기를 유지한다.
BLUE block8 ln_1 capture extra forward를 계측하며 제거하지 않는다.

PRUNE은 20 dense native MEMIT write 후 한 번만 native spectral compression을
수행한다. 명시 예외 `PRUNE_TERMINAL_BASE_FIX`는 finalW의 base를 RAM coldW0로
고정한다. W5/10/15는 dense, W20 current/all-seen은 이 terminal transform 후다.
RECT는 native dense lower planning→restore→40% relative mask public commit을
사용한다. 동률의 실제 retained 수를 기록한다.

NoCP, z disk cache 없음, exact_resume=NOT_AVAILABLE. RAM rollback과 PRUNE W0/SVD는
영속 checkpoint가 아니다. 원 입력과 기존 실행 source/raw/job은 변경하지 않는다.

## 실행 순서

전용 clean worktree에서 다음을 사용한다. Python은
`/mnt/raid5/janghj/EasyEdit/.venv/bin/python`이다.

```bash
python -m project.run_scripts.gptj_cake_blue_prune_rect.prepare
python -m project.run_scripts.gptj_cake_blue_prune_rect.preflight
python -m project.run_scripts.gptj_cake_blue_prune_rect.preflight --metadata-r2
# 검사된 own source를 commit한 후:
python -m project.run_scripts.gptj_cake_blue_prune_rect.submit
```

prepare는 이전 fullSHA/finite/schema receipt와 현재 stat를 재사용한다.
모델/C0/P를 다시 load/hash/생성하거나 복사하지 않는다. CPU 검사는 actual GPU PASS가
아니다. 실제 B1 write/history/observer 및 B2 entry 검사는 각 본 trajectory에 포함된다.

submit은 기존 Server2 ODE-edit owner/source frontier를 단발 확인하고 cap2 안에서
CAKE→PRUNE, BLUE→RECT afterany lane을 등록한다. 네 GPU job과 CPU collector 모두
held owner/argv/source/resource/dependency 검사 후 release한다. 각 GPU job은
1GPU/6CPU/59392M/48h 상한, CPU collector는 GPU0/6CPU/24576M/4h다.
48h는 ETA가 아니다. 최초 상태 인계 뒤 agent 반복 모니터/자동 retry는 없다.

W&B는 기존 shared helper와 PRICE scalar mapping을 읽기 전용으로 재사용한다.
actual job_id/run.name, 모델·writer·source/config를 결속한다. current/pre와
current/post는 매번 R100/P200/N1000이며 W5/10/15/20 all_seen은 실제 prefix다.
N desired=true, percentage/nats 및 token/prompt 분모를 보존한다. SDK 접수는
remote delivery나 과학 완료를 인증하지 않는다. 원문·token·tensor·코드·stdout 업로드는 없다.

collector는 저장된 rows/counts/state hash를 CPU에서 독립 집계한다. partial/실패 prefix를
보존하며 CPU scheduler 성공과 네 scientific arm의 완료를 구분한다. raw는 local에
유지하고 Git에는 source와 소형 보고/manifest만 게시한다.
