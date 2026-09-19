# Server2 checkpoint mechanism audit

Instruction: `ODEEDIT-S06-S2-CHECKPOINT-MECHANISM-AUDIT-20260920-V1`.
이 package는 기존 singleton L4 BLUE checkpoint의 관측/재구성 분석이다. 새 z 최적화,
편집 chain, history append, 모델 checkpoint 저장은 없다.

## 현재 운영 상태

2026-09-20 사용자: “gpu 자리가 없으니 구현까지만 완료하고 user의 호출 기다려”.
구현/CPU 검증만 완료한 뒤 대기한다. 최초 gate job51071은 실행 전 PENDING에서
JobHeldUser로 hold했다. 최초 execution source3f65d170과 archive/lock은 불변이다.
새 source는 토큰 provenance/후속 analysis/CPU tests를 추가한 별도 lineage이며
이미 제출한 job의 source를 바꾸지 않는다. Release, 새 제출, scheduler polling,
GPU/model 분석은 명시 USER recall 전 하지 않는다. Submit helper는 pause receipt가
있으면 recall receipt 없이는 scheduler 조회 전 거부한다.

CPU PASS는 실제 Llama/FP32 solve/evaluator parity PASS가 아니다. 현재 GPU 결과는
NOT_RUN이다. 실제 최종 H1–H4 기전 판정과 Z00 report도 아직 미완료다.

## 구현 구성

|코드|역할|
|---|---|
|staging / prepare / common|exact allowlist 수신·재해시, SOURCE_KEEP, 원 source isolated assembly|
|archival|100 current +12 seen identity/분모/strict/overwrite, at-write anchor, 2000 cluster bootstrap|
|geometry|W0·actual12 W/M, FP64 subtraction, 11구간 공통256-vector sketch|
|validation|원소/열/요청/평가 row 최대오차 및 10% 반복spread, zero-reference 규칙|
|model_runtime / model_gate|원 native group packing, full-forward/prefix, original MB16 evaluator, B1 actual-delta gate|
|key_bank|C01 PASS 후 native700+geometry512, actual packing/positions/lookup, entry-h affine 대조, F00|
|operators / operator_lane|비대칭 LU, history별 RHS 재사용, native100별 S/B, B1/B91 dense parity, 직접 thin SVD, 대수적 counterfactual|
|activation_lane|고정 first100 NS의 2구간 panel, s=0/.5/1과 all-token gradient·E/D 교차항|
|reporting|local raw에서 raw-free CSV/PDF/PNG/한국어 보고; 미측정 수치 대입 없음|
|control / submit / *.sbatch|source archive/lock, explicit resource, held inspection; autonomous cascade 없음|

## CPU 재현

```bash
PYTHONPATH=/mnt/raid5/janghj/ODE-edit/local/fixed10k-preedit-eval/attempt-v1/deps-transformers-4.44.2 \
PYTHONDONTWRITEBYTECODE=1 /mnt/raid5/janghj/EasyEdit/.venv/bin/python \
  -m unittest discover -s project/run_scripts/checkpoint_mechanism_audit -t . -p 'test_*.py' -v
python3 scripts/slurm_memory_policy.py audit \
  project/run_scripts/checkpoint_mechanism_audit/run.sbatch \
  project/run_scripts/checkpoint_mechanism_audit/analysis.sbatch
```

기존 staging/CPU 결과 root는
`local/checkpoint-mechanism-audit/20260920-v1/attempt-v1/`다.
Create-once 산출물에 명령을 그대로 재실행하면 overwrite를 거부한다.
CPU 결과는 archival/geometry receipts의 source/hash/count로 재사용한다.

## USER recall 뒤 실행 DAG (현재 실행 금지)

1. 기존 held gate51071/source/lock을 정확히 확인한 뒤 사용자 지시 범위에서 처리한다.
   C00/C01 actual PASS 전 key-bank/operator/suffix를 시작하지 않는다.
2. Key bank는 `python -m ...key_bank --gate GATE/terminal.json --output NEW`.
   W0 h0와 actual entry h를 별도 기록한다. Geometry512는 stream ID disjoint이며
   `sha256('20260920|'+case_id)`와 numeric case-id tie-break로 사전 고정했다.
3. Pilot lane0 histories `0,10`, lane1 `1,100`: 각1GPU/8CPU/60416M. 공통 bank를
   읽되 history별 output/factor single writer. 모든 pilot O PASS 뒤에만 extension.
4. Extension lane0 `5,20,30,40,50`, lane1 `60,70,80,90`. Native demands는
   해당 entry history에 결속하며 누락 target은 해당 claim만 차단한다.
5. G001_010/G050_100은 C01/A01/F00 PASS 뒤 독립 lane으로 가능하다. 기존 operator
   jobs가 있으면 project cap2 내에서만 admission한다. 다른 job 변경 금지.
6. 신규 실행 source는 `control freeze --mode keys|operator|activation --args-json ARGS`
   로 고정한다. ARGS는 해당 module의 CLI 인자 문자열 배열이며 shell expansion 없음.
   `--analysis-output`은 create-once 목적지다. 공개된 source HEAD와 source archive를 분리한다.
7. 모든 cell의 PASS/FAILED/BLOCKED/SKIPPED를 실제 증거로 수집한 후에만 최종 report를 생성한다.

전체 작업은 project cap2, task max2, 각1GPU/8CPU/host60416M/exportNONE/Requeue0이다.
설계 RAM64GiB 대신59GiB 요청을 적용한다. 실제 비용은 아직 미측정이며8h gate 예약을
속도/완료시간 추정으로 쓰지 않는다. 추가두번째 slot을 중복 작업으로 채우지 않는다.

## 수치·해석 경계

- Original Torch2.9.1+cu128 / Transformers4.44.2 / FP32/eager / matmul TF32false,
  cuDNN TF32true. Writer/evaluator BOS 차이와 original manual-left MB16을 보존한다.
- Key/block all-elements 1e-6+1e-5|ref|, solve max-RHS relative1e-5,
  actual-delta/response1e-3, row NLL/margin1e-4, zero-reference≤1e-12의 abs1e-7.
  동일경로 repeat는 그10%; 성능에 따른 완화 없음.
- 원 solve는 FP32; operator LU와 잔차/해석은 명시 FP64. P/M 임의대칭화, inverse,
  CG/Cholesky, raw score clipping을 하지 않는다.
- 인접 singular gap≤1e-3*largest_sigma는 사전 고정 descriptive grouping이며 gate/
  절단/방법 변경이 아니다. 모든 raw mode는 유지한다.
- B1 이외 demand는 RECONSTRUCTED_NATIVE_WRITE. 저장평가 anchor는 first-write-N이며
  미실행 W0를 대체하지 않는다. All-case와 conflict-free/active-version은 별도다.
- Per-item/prompt/tensor/teacher/cache/raw logs는 local-only. 원 CP 불변. 논문식 해석은
  task-local 사용자 허용 범위 내 H1–H4 근거와 한계로만 하며 새 실험을 만들지 않는다.

일반 Slurm audit의 기존 server4 6 launcher policy 위반은 unrelated historical source다.
본 package의 두 launcher 별도 감사와 분리하며 타 source를 고치지 않았다.
