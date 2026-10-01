# JLZ sequential B100×10 — 실제 초기 경계 인계

Instruction/ACK ODEEDIT-GH-SH1-JLZ-BS100X10-20261001-R1.
**MONITORING_PAUSED_AWAITING_USER**, 2026-10-01 12:28 UTC.

원 GPU56684와 CPU collector56685는 source/config/자원을 변경하지 않고 계속 실행한다.
Frozen source c91962dd4952eb58a16b6b00dcaa3018be71693f,
tree620a5ed510569a09bac405fc220e11afa0724182.
Lock SHA3fb6601ac22352f214c17d3b0a3ebcf495c4897fba0e9523a9afa36bf8a396cf.
Run `/mnt/raid5/janghj/ODE-edit/local/jlz-sequential/20261001-v1/attempt-r1`.

실제 B1 commit, 층4–8 history 각1회(합5), R/P/N 및 TF observer 저장과
B2 entry의 W/H hash가 B1 post와 정확히 연결됨을 확인했다.
관측 파일의 SHA는 B1 commit receipt와 일치했다. 전체 model byte 인증이 아니라
지정 W/H hash 및 원 nonselected pointer/version/source-scope guard 범위다.

| B1 항목 | 실제 기록 |
|---|---:|
| Solver | BUDGET_STOP / CALL_CAP |
| Scientific whole-batch calls | 120 |
| 별도 actual BS100 preflight | 3 |
| Accepted / backtracks | 97 / 21 |
| Normalized residual | 0.004618457520544859 |
| Fresh final | true |
| FP32 materialization / L4 key | bitwise |
| Commit NLL/KL maxabs | 2.8386712074279785e-6 |
| B1 fit seconds | 5632.271129179746 |
| B1 stage total seconds | 6242.371006421745 |
| Peak allocated bytes | 38853122560 |

미수렴을 수렴 PASS로 변경하지 않는다. 사용자 fixed-budget 반환후보 commit 정책을 적용한
초기 성공이며 10-batch terminal 성공이 아니다. 초기 B1 기록의 Current preference는
R100/100, P134/200, N887/1000; TF strict R100/100,P67/200,N171/1000이다.
이는 저장 receipt의 초기 숫자이며 최종 독립 paired 분석/장기 유지 결과를 뜻하지 않는다.

Preflight3 calls310.714865초, actual shared-selected route MB2가 채택되었다.
Supplied10→20 구간46.942초/oracle에1200을 단순 곱한15.647h는 optimizer-only 가정치이며,
load/entry/geometry/관측/I-O 및 후속 batch 변화를 제외한다. 48h wall은 ETA/예산이 아니다.
Parent allocation 전체 비용과 10-batch 결과는 다음 명시 recall에서 확인한다.

초기 증거 SHA:

- INITIAL_VALID.json: 1b57c34483872c64845e84e678a2cb26cf1af60e1fc1d22ae494ef1a6cf13549
- B001-committed.json: 59e096a9f5d1dc96cd7b07763eaa317e0e033de7de6a3cb7d9fb25b12d33142f
- B002-entry.json: 2d52b1ed04f5d465ff16e428bb1e7febdde0a435bba5ac0ccc06209ac31052ef
- W01-observations.json: 6e61ae5f1882ebcdeedf92b04ae01e16caaceb22f9ee20fcaf7ef6fe3fe047b5

Local owner-initial-pause.json 및 owner-initial-monitor.json을 인계 상태로 기록했다.
이제 원 task의 polling/log/result/terminal 대기를 중지한다. 등록된10회 runner/collector는 자연 진행.
효율화56704는 별도의 최초 small qualification 뒤 이미 pause됐고 두 job 모두 변경0이다.
새 checkpoint 없음, exact_resume=NOT_AVAILABLE, SERVER3 actions0, NO_BROADCAST_NOT_REQUIRED.
