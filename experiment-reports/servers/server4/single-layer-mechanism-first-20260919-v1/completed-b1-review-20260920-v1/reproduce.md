# CPU 리뷰 재현

대상은 terminal51058의 immutable B1 output뿐이다. 모델/evaluator/Slurm을 실행하지 않는다. 새 빈 분석 namespace를 사용한다. 이미 존재하는 create-once 출력은 덮어쓰지 않는다. 실제 실행 source와 본 분석 source SHA는 analysis-manifest에 분리되어 있다.

아래는 저장된 분석 명령의 재현 형태다. `REVIEW_LOCAL`은 이 task의 허용 local root 아래 새 경로로 지정한다. 원 reproduction을 실행할 필요 없이 봉인된 CSV/JSON도 읽을 수 있다.

```bash
cd /data/janghj/ODE-edit/local/single-layer-mechanism-first/20260919-v1/completed-b1-review-20260920-v1/worktree
REVIEW_PY=/data/janghj/EasyEdit/.venv/bin/python
REVIEW_OUT=/data/janghj/ODE-edit/local/single-layer-mechanism-first/20260919-v1/PROGRAM/b1-fd-waiver-r4/output
REVIEW_LOCAL=/data/janghj/ODE-edit/local/single-layer-mechanism-first/20260919-v1/completed-b1-review-20260920-v1/reproduce-new
REVIEW_ACCOUNTING=audits/servers/server4/single-layer-mechanism-first-20260919-v1/completed-b1-review-20260920-v1/accounting.json
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
"$REVIEW_PY" -m project.run_scripts.single_layer_mechanism_first.review_b1 --output "$REVIEW_OUT" --destination "$REVIEW_LOCAL/first-reduction"
"$REVIEW_PY" -m project.run_scripts.single_layer_mechanism_first.review_completed_b1 --output "$REVIEW_OUT" --destination "$REVIEW_LOCAL/full-reduction-r2" --accounting "$REVIEW_ACCOUNTING"
"$REVIEW_PY" -m project.run_scripts.single_layer_mechanism_first.review_supplement --output "$REVIEW_OUT" --destination "$REVIEW_LOCAL/supplement-r3"
"$REVIEW_PY" -m project.run_scripts.single_layer_mechanism_first.review_tensor_inventory --output "$REVIEW_OUT" --destination "$REVIEW_LOCAL/tensor-inventory.json"
"$REVIEW_PY" -m project.run_scripts.single_layer_mechanism_first.review_local_solver --output "$REVIEW_OUT" --destination "$REVIEW_LOCAL/local-solver.json"
"$REVIEW_PY" -m project.run_scripts.single_layer_mechanism_first.review_provenance --attempt /data/janghj/ODE-edit/local/single-layer-mechanism-first/20260919-v1/PROGRAM/b1-fd-waiver-r4 --repo . --destination "$REVIEW_LOCAL/provenance.json"
```

Publication은 `publish_b1_review`의 create-once package 경로를 사용한다. 기존 package를 overwrite 대상으로 지정하지 않는다. PNG만 byte 재현할 때는 별도 scratch package에 필요한 compact CSV를 정확 복사하고 `--plots-only`로 실행한다. 이번 검사는 원 PNG의 해시를 먼저 보존한 뒤 같은 코드/환경에서 재생성해 byte 일치를 확인했다.

```bash
"$REVIEW_PY" -m unittest project.run_scripts.single_layer_mechanism_first.test_review_b1 project.run_scripts.single_layer_mechanism_first.test_review_completed_b1 project.run_scripts.single_layer_mechanism_first.test_review_tensor_inventory project.run_scripts.single_layer_mechanism_first.test_review_supplement project.run_scripts.single_layer_mechanism_first.test_review_publication
```

HTML은 system Python의 markdown-it-py3.0.0 CommonMark+table renderer로 생성했다. 이는 한국어/표/링크의 실제 HTML 생성·구조 검사이며 GUI browser pixel 검사는 아니다. PNG는 두 장 모두 직접 육안 확인했다. 코드 및 compact output만 Git에 넣었고, 전체 NLL row·native/factor tensor·HTML scratch는 task-local이다.

재현에 원 raw/공통자산이 없어졌다면 NOT_AVAILABLE로 중단해야 한다. 새 GPU로 보완하거나 다른 arm의 raw를 가져오는 재현 절차는 없다. 권한은 최신 사용자 호출에 의해 다시 제한될 수 있다.

