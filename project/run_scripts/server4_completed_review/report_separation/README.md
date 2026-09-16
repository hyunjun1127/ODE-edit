# Server4 독립 보고 v2 publication

이 코드는 완료된 compact CSV/보고만 재구성한다. Scientific raw, tensor,
model, scheduler, GPU 및 기존 runtime에 접근하지 않는다. 지표를 다시 평가하지 않는다.
Native/BLUE-style 표시 이름과 physical layer를 명시하고 원 raw arm ID를 보존한다.

## 재현

원본 `307ba7ae`의 v1 패키지와 publication 기준 `4d9bd910`을 포함하는 clean
worktree에서 실행한다. `build`는 새 v2 경로가 없는 복제 worktree에서만 실행한다.
현재 게시된 v2를 덮어쓰지 않는다. 각 output은 create-once다.

```bash
/usr/bin/python3 -B -m project.run_scripts.server4_completed_review.report_separation.build build
/data/janghj/EasyEdit/.venv/bin/python -B -m project.run_scripts.server4_completed_review.report_separation.plots --output NEW_FIGURES
/usr/bin/python3 -B -m project.run_scripts.server4_completed_review.report_separation.finish figures --source NEW_FIGURES
/usr/bin/python3 -B -m project.run_scripts.server4_completed_review.report_separation.checks --output NEW_CHECKS
/data/janghj/EasyEdit/.venv/bin/python -B -m project.run_scripts.server4_completed_review.report_separation.plots --output NEW_REPRODUCED_FIGURES
/usr/bin/python3 -B -m project.run_scripts.server4_completed_review.report_separation.verify --output NEW_TESTS.json --reproduced NEW_REPRODUCED_FIGURES
/usr/bin/python3 -B -m project.run_scripts.server4_completed_review.report_separation.finish seal --checks NEW_CHECKS/checks.json --tests NEW_TESTS.json
```

Matplotlib 3.10.7로 CAKE/baseline 전용 그림 두 개를 생성한다. 원 scope와 일치하는
CAKE 그림 두 개와 cap 그림 여섯 개는 byte 그대로 재사용한다. Markdown 검사는
markdown-it-py 3.0.0의 GFM HTML 표 cell/행열/링크 검사이며 browser pixel 렌더
검증으로 부르지 않는다. 새 두 PNG는 별도 output에서 byte 재현한다.

초기 build의 문자열 scope 검사 한 건은 instruction ID 안의 `CAKE`까지 금지한
publication 검사 문제였다. 새 cap 본문은 instruction을 manifest에만 연결하도록
수정했다. 실패한 새 출력은 local `staging/build-r1-cake`, `build-r1-cap`에 보존했고
v1/source/runtime/지표를 바꾸지 않았다. 이는 모델 실행 실패나 scientific repair가 아니다.

`analysis-manifest.json`은 frozen publication-code commit을 기록하며 실제 experiment
execution source는 복사한 provenance에서 별도로 유지한다. Rooted receipt의
`PENDING_MAIN_PUBLICATION`은 seal 당시 상태다. 최종 게시 상태는 task receipt와
server-head handoff에 별도 기록하며 sealed family receipt를 사후 덮어쓰지 않는다.
