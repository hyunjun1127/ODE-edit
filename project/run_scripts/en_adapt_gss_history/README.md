# SH3 RES / GSS_REC fixed10k

최신 사용자 두-arm/cap2 지시를 구현한다. 각 Slurm job은 독립적인 Llama W0/zero M4에서 시작해 B100×100을 한 process RAM에서 연속 실행한다. B1/B2는 본 stream의 일부이며 별도 pilot나 성능 gate가 없다. `EN_ADAPT_H_GSS`는 실행 대상이 아니다.

부모 `en_adaptive_nullspace`의 실행 검증된 native/z-hook/current geometry/controller와 readiness source를 읽기 전용으로 사용한다. `runtime.py`는 fixed10k horizon만 확장한다. `history.py`는 fact-version ledger/bank/pending, `factors.py`는 at-write cold teachers와 fresh NLL/KL factor VJP, `objective.py`는 R512+선택 history, `block_diagnostics.py`는 동일 SVD에서 블록 교차항을 관측한다. 원 threshold/rank/epsilon/Armijo/ideal-ray curvature는 변경하지 않는다.

GSS sketch는 signed NLL gradient에 고정 Pstar와 PCG64 seed20260920의 독립 Gaussian32×32 map 두 개를 적용한 2048차원이다. 모델/hidden/head FP32, map/projection/sketch algebra FP64이다. RES 또는 pool≤512에서는 selection NLL backward/sketch가 없다. Overflow는 microbatch1 graph마다 별도 NLL/KL VJP를 수행하며 per-fact dense W gradient를 저장하지 않는다. selected history KL의 aggregate만 dense gradient가 된다. 원 at-write full-vocabulary target teacher는 version별 불변이며 repeat/eviction으로 갱신하지 않는다.

`observers.py`의 neural family stream과 microbatch16은 부모 evaluator와 같다. 선형 시간 CPU reducer만 분리했다. 선택 후 entry/WN/selected를 같은 current/first100/hash128 panel 및 지정 full-history 입력에서 관측하며 exact FP32 endpoint alias만 재사용한다. TF accuracy는 자유 생성 정확도가 아니다. 두 arm은 selection과 recency 가중치가 함께 달라 각각의 인과 기여를 분리할 수 없다.

실행 준비 경로:

```bash
cd /data/janghj/ODE-edit/local/en-adapt-gss-history/20260920-v1/worktree
PY=/data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/venv/bin/python
# create-once CPU 입력·자원 계획 (이 task에서 이미 수행됨)
PYTHONDONTWRITEBYTECODE=1 "$PY" -m project.run_scripts.en_adapt_gss_history.prepare
# source commit과 map/input seals 준비 후 단 한 번 freeze
"$PY" -m project.run_scripts.en_adapt_gss_history.freeze
# cap2 admission → 두 held exact 검사 → 두 release, 재등록 금지
"$PY" -m project.run_scripts.en_adapt_gss_history.freeze --submit
```

위 prepare는 생성 경로가 존재하면 재실행하지 않는다. 실제 submission/release receipt가 있으면 submitter도 중복 등록을 거부한다. 최종 인계의 source/lock/job mapping이 실제 실행 근거다. 테스트만으로 GPU PASS를 주장하지 않는다.

`save_checkpoints=false`: W/M/optimizer/delta/resume bundle disk 저장이 없고 exact crash-resume은 불가능하다. Cold teacher/keys는 immutable replay 입력이며 raw JSON은 lossless gzip으로 local에만 저장한다. Root나 공유 EasyEdit, 기존 teacher/model은 수정하지 않는다.

각 job 1GPU/8CPU/121856MiB, ubuntu/gpu, exportNONE, Requeue0, wall7일. 두 job 합산 자원·cold store·출력을 lock에 결속한다. Runtime state는 두 job 간 공유하지 않는다. Release 뒤 scheduler/log/result/initial/B2/terminal 조회·대기·callback은 하지 않는다. 프로그램은 자연 진행하며 agent는 `MONITORING_PAUSED_AWAITING_USER`로 멈춘다. 종료 후 사용자 recall 때만 CPU report 비교와 terminal review를 수행한다.

각 process는 B100 후 `report.build/write`를 호출한다. 이후 사용자가 review를 요청할 때의 비교 도구는 `report.compare_attempts(res_output, rec_output)`이며 자동 동료 job 관찰을 하지 않는다. 부모 precision은 `NOT_ESTABLISHED`, 신규 hash4/hash64 diagnostics는 관측이며 성능 gate나 오차를 숨기는 tolerance 조정이 아니다.
