# 완료 CPU 리뷰 재현

원 실행: GPU52823 / CPU52824. 분석은 원 runtime를 import/실행하지 않는다.
원 raw, panels, execution.lock 및 frozen source가 같은 server4 경로에 있어야 한다.
현재 report를 덮지 않고 task-local 새 scratch에서 실행한다. 모델/GPU 호출0.

```bash
cd /data/janghj/ODE-edit/local/native-delayed-write-e3/20260924-v1/completed-review-r1/worktree
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
review_scratch=$(mktemp -d /data/janghj/ODE-edit/local/native-delayed-write-e3/20260924-v1/completed-review-r1/repro-XXXXXX)
export DELAYED_REVIEW_LOCAL="$review_scratch/local"
export DELAYED_REVIEW_REPORT="$review_scratch/report"
export DELAYED_REVIEW_AUDIT="$review_scratch/audit"
/data/janghj/EasyEdit/.venv/bin/python -m unittest project.run_scripts.native_delayed_write_e3_completed_review.test_review -v
/data/janghj/EasyEdit/.venv/bin/python -m project.run_scripts.native_delayed_write_e3_completed_review.reducer
/data/janghj/EasyEdit/.venv/bin/python -m project.run_scripts.native_delayed_write_e3_completed_review.detail
/data/janghj/EasyEdit/.venv/bin/python -m project.run_scripts.native_delayed_write_e3_completed_review.supplement
/data/janghj/EasyEdit/.venv/bin/python -m project.run_scripts.native_delayed_write_e3_completed_review.report
```

`preflight`는 최초 리뷰의 exact accounting snapshot 전용이므로 재현 시 실행하지 않는다.
새 scheduler 조회가 필요 없다. `finalize`는 source commit 및 원 reproduction receipt를
결속하는 일회성 publication 도구이므로 수치 재현 명령에 포함하지 않는다.
Raw와 분석의 create-once JSON/CSV를 보존하며, 생성된 report/PNG만 code 재생성한다.
새 raw/GPU 평가·Slurm write·타 task 재개는 없다.

주요 파일:

- [한국어 상세 보고](report-ko.md)
- [E1 독립 첫 표](first-endpoint-table.csv), [단계 coverage](stage-coverage.csv)
- [전체 paired summary](paired-summary.csv), [W1→later](e1-atwrite-paired.csv)
- [active/overwrite](active-overwrite.csv), [factorial](factorial-summary.csv)
- [모든 corner/patch TF·NLL](factorial-patch-metrics.csv)
- [module](module-summary.csv), [layer drift](layer-drift.csv), [patch 검사](patch-audit.csv)
- [source 적합성](source-conformance.csv), [비용](compute.csv), [raw inventory root](artifact-index-root.json)
- [분석 lineage](analysis-manifest.json), [package](package-manifest.json), [rooted receipt](rooted-receipt.json)

통계/그림은 단일 고정 trajectory의 저장된 실제 관측만 사용한다. Full tensor/General KL
teacher의 사후 독립 재구성은 불가하며 GFM HTML renderer 미설치는 별도 기록했다.
NO_BROADCAST_NOT_REQUIRED. 큰 raw/CP/model/teacher/prompt/log는 Git에 올리지 않는다.
