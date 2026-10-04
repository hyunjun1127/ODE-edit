# JLZ v12 SH4 → SH3 이식·등록

상태: **RELEASED_RESOURCE_DEPENDENCY_PENDING / MONITORING_PAUSED_AWAITING_USER**. 실제 S3 GPU qualification·main 초기·완료는 모두 NOT_OBSERVED다.

Nonce `ODEEDIT-GH-SH4-SH3-JLZ-V12-MIGRATION-20261004-R1`. S4 실행 source `7852d66ed5467f94ebda73507b6ccdbf5c921328`의 수학 코드를 그대로 사용하며 S3 전용 prepare/submit과 tests만 추가했다. 원 정본27개 SHA/size 및 CSV2000행의 모든 identity 필드를 검산했다. CPU reference29·production7·이식5 검사가 통과했다. owner audit이며 별도 독립 agent 검토는 수행하지 않았다.

SH4 main58173/collector58174는 source 소유자가 취소했고 할당 GPU초0이다. S4 pilot58172는 완료·410GPU초의 역사 자료이며 S3 qualification으로 쓰지 않는다. 소형 config/lock/provenance6개121342B만 allowlist로 수신했다. 원자료 KEEP, 큰 자산 전송0.

계획 DAG는 S3 pilot BS2×2 → 동일source/config technical READY를 검사하는 fresh V12_MAIN BS100×20 → afterany CPUcollector다. 최대동시1GPU, 각8CPU/59392MiB, pilot4h/main168h 상한, collector0GPU/8CPU/24576MiB/4h이다. Wall은 ETA가 아니다. 최대논리평가 pilot100/main50000, update96/48000은 사전상한이지 실측이 아니다.

S3 기존 Python3.12.3/torch2.9.1+cu128을 재사용하고 task-local dependencies-r1에 transformers4.57.1/tokenizers0.22.1/jsonschema4.10.3/pyrsistent0.20.0을 고정했다. 공유 venv4.44.2는 변경하지 않았다. S4 runtime와 비교한11개 소스 SHA는 일치한다. Model FP32/eager/TF32off, geometry FP64, historyCPUFP32를 유지한다. Native BLUE311b076은 task-local 기존 사본이고 evaluator는 원 jlz_realization/observe.py·inputs.py를 보존한다.

모델 revision8afb486c1db24fe5011ec46dfbe5b5dccdb575c2, fixed10k dataset, native6contexts, L4–8 C0는 기존 S3 fullSHA receipt와 현재 size/inode/mtime·논리 경로로 재결속했다. S4 W0 raw는 수신하지 않는다. 승인된 S3 fresh first2k W0 관측1회를 main 안에서 수행하며 학습에 피드백하지 않는다.

RAM 계획: model CPU loading peak 약34GiB, fit/write phase history+rollback 약8.8GiB, entrycache3GiB, 1층 FP64 system/solve 등12GiB 및 여유4GiB. 두 phase peak를 중복 합산하지 않는다. 과거 S4 pilot MaxRSS33.12GiB/GPU35.62GiB는 참고이며 S3 peak 보증이 아니다. S3 request58GiB를 상향하지 않는다. H200 GPU 계획 model30+write3+activation20+solve10+여유8=71GiB. 디스크 출력·temp·여유24GiB를 reserve하고 최신 실제 free/inode는 admission receipt에 기록한다.

`save_checkpoints=false`, exact resume 불가. 설정·스칼라·평가 raw만 저장하고 W/H/delta/RNG 복원 bundle은 저장하지 않는다. v10/v11과 다른 중지 task는 재개하지 않는다. Release 후 resource/dependency snapshot1회에서 agent monitoring을 중단한다. 본 프로그램은 pilot qualification 후 W20와 CPUcollector까지 자연 진행한다.

## 실제 제출과 인계

2026-10-04 **19:44:38 KST** 단일 resource/dependency snapshot:

| 단계 | Job | GPU/CPU/host | dependency | 관측 상태 |
|---|---:|---|---|---|
| S3 BS2×2 pilot | 58178 | 1 / 8 / 59392MiB | 없음 | PENDING Resources |
| fresh V12_MAIN BS100×20 | 58179 | 1 / 8 / 59392MiB | afterany:58178 + 동일 source/config CHAIN_COMPLETE READY | PENDING Dependency |
| CPU collector | 58180 | 0 / 8 / 24576MiB | afterany:58178:58179 | PENDING Dependency |

모든 job은 owner `janghj`, node `ubuntu`, partition `gpu`, QoS `lab_gpu_s3`, exportNONE/Requeue0이다. 세 job 모두 held 상태에서 full argv/source/launcher SHA/resources/dependency를 확인한 뒤 collector→main→pilot 순으로 release했다. Manual hold는 남기지 않았다. 제출 직전 본인 GPU job0 / projectcap1 helper PASS였다. 이후 agent의 scheduler/log/result 조회는 하지 않는다.

Pilot technical failure 시 main은 source/config-bound READY 부재를 기록하고 model load 전에 차단된다. afterany는 실패도 collector로 모으기 위한 자원 순서이고, 기술 gate를 생략하지 않는다. 자원이 배정되면 등록된 프로그램이 자동으로 pilot→fresh main W20→collector를 수행한다. 원source 및 raw 보존, 자동 retry/다른 task 재개0이다.

실행 source `1d27a830274aaee49a713bd07e463b0513591c9e`, tree `5be499f7ddc68983c47144009b5f0b35117ac54b`. Source archive SHA `32901e2a1011a1488c722afdcb2941d7203078c44c59f90572393608cafe04b8`. Config SHA `b09cfae0d6a8a29071818c114017b9f350ec9c58ec3143ac211b42179145458b`, execution lock SHA `959ce59d104713869bccfa8db20ad88c2e9d4de3d9e39215d1cae9bd0996b244`. 분석/게시 commit은 이 실행 commit과 구분한다.

실물 attempt: `/data/janghj/ODE-edit/local/jlz-v12-shared-budget/20261004-v1/attempt-s3-r1`. 원 모델 `/data/janghj/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`, dataset `/data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json`, context `/data/janghj/ODE-edit/local/memit-history-fixed10k/20260928-v1/inputs/baseline/contexts.json`, C0 `/data/janghj/EasyEdit/examples/data/stats`. 층별 C0/hash·native closure·실행runtime 절대 경로는 [port-binding.json](../../../../audits/servers/server3/jlz-v12-shared-budget-bs100x20-20261004-v1/port-binding.json)에 기록했다.

Snapshot free `76007530496` bytes (약 70.79GiB), inode `224328254`. 출력·atomic temp·여유24GiB 계획. 제출 시 S3 실제 GPU allocation/과학 초기/실행 비용은 미관측이며 S4 pilot410GPU초를 S3비용에 합산하지 않는다.

## 검산 재현과 한계

원 source 수학9개 파일 exact 보존 테스트, 위조 cessation SHA 거부, cap1 의존성 및 node/memory/CPU/export, launcher shell 문법/실제 Python/overlay, main READY가 model load보다 앞서는지의 5개 이식 검사가 통과했다. CLI import도 CPU로 확인했다. 원 production7개에는 작은 무작위 Llama의 실제 fit→write→history→rollback과 terminal schema가 포함되며, Llama-3-8B GPU qualification과 다르다. reference29개는 S3 NumPy/SciPy 버전으로 별도 local output에 재검산해 원 정본을 덮지 않았다.

```bash
cd /data/janghj/ODE-edit/local/jlz-v12-shared-budget/20261004-v1/worktree
PYTHONPATH=/data/janghj/ODE-edit/local/jlz-v12-shared-budget/20261004-v1/dependencies-r1 \
  /data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/venv/bin/python -B \
  -m unittest project.run_scripts.jlz_shared_budget.test_cpu project.run_scripts.jlz_shared_budget.test_server3 -v
```

전체 repo Slurm-memory audit에는 기존 무관 S4 launcher6개의 상한 초과가 남아 있다. 이식 namespace는 request58GiB와 typed held 검사를 통과했으며 무관 파일을 수정하지 않았다. S3 target-model 수치 결과와 W20의 완료 검산은 아직 없다. 완료 상세 리뷰는 사용자 recall 때 수행한다. 이번 인계는 구현·등록·release 사실만 다룬다.
