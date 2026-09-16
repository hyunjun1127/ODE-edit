# CAKE/baseline 및 alpha-cap 독립 정본 v2

두 family 보고와 링크 전용 새 목록을 실제 생성했다. CAKE는 W0/native/BLUE pair·physical-layer singleton만, alpha-cap은 네 정책과 동일1k 참고 baseline만 포함한다. 원 수치/분모와 v1 bytes 불변. GFM HTML의 실제 표 cell/링크를 검사했고 새 명칭의 baseline 그림은 코드 생성했다.

새 scheduler/GPU/model/evaluator/raw tensor/Slurm/rsync/delete0. 검증된 source/report-only payload를 non-force main `557f7e1346ac32e5d04131a63ff7cab3141c397d` / tree `11b3ea5b071475b589c72d28a33bbfe0a25a63eb`에 게시하고 remote exact/clean/ahead-behind0/0을 확인했다. 분석 코드 `78c3097aa4da6079ee1891678b64edad97f4f204`는 실제 experiment source와 별개다. SH2 `8aabac92`와 GH 기준 main `4d9bd910`을 보존했다. 현재 커밋은 이 게시 사실과 STOP 상태만 추가한다.

## 두 독립 정본

절대경로 prefix는 `/data/janghj/ODE-edit/local/report-separation/20260916-v1/worktree/`다. 아래 repo 경로를 prefix에 붙인 실물이 존재한다.

- CAKE+baseline: `experiment-reports/servers/server4/cake-native-lifelong-b100x100-2026-09-15-v1/completed-review-v2/diagnostic-report-ko.md`
  SHA256 `e108438abd51b8d536925ce4ad863e6e8e815df10a105aa1503899c4de65953a`.
- Alpha-cap: `experiment-reports/servers/server4/ep-tw1-alpha-cap-sweep-2026-09-15-v1/completed-review-v2/diagnostic-report-ko.md`
  SHA256 `fa966bc035b866600a72503c9715fa632ee27c1c71b2cffee160db978ea0ef47`.
- 링크 전용 목록: `experiment-reports/servers/server4/completed-experiments-review-2026-09-16-v2/README.md`
  SHA256 `75c4a66ca19346fc6861bdfd86f4fc4f954626893b2f31db623c306ef60d2910`.

기존값/v1 불변, 74개 mapping, 9 CPU tests, 121 manifest SHA/size 참조, 실제 GFM HTML 표/링크 검사를 통과했다. 새 그림2개 byte 재현, 기존 그림8개 exact 재사용. 과학 검증 수준/미측정은 원래대로 유지한다. Family rooted receipt는 seal 당시 상태를 보존하고 최종 게시 상태는 runs/task receipt에 별도 기록했다.

TASK_COMPLETE_STOP / monitoring_active=false / automatic_resume=false. 신규 실험·모니터링·후속 작업은 없다.
