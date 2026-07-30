# Session 01 Motivation — locked MV-2 execution preflight

- 점검일: 2026-07-31 KST
- 점검 역할: 독립 execution red preflight
- 대상: `odeedit_mv2refresh_pair_v1` 한 건
- 허용 범위: metadata, code, test, wrapper, session/runtime/resource 경계만
- 금지 범위: 과학 outcome 수치, raw outcome row, `direct_z`, Git 변경, Slurm 제출

## 네 범주

### Proposal에서 온 내용

- W0에서 고정한 guide 아래 W1 state refresh가 fixed-direction continuation보다
  나은지 작은 matched-C diagnostic으로 검증한다.
- 이 실행은 Motivation mechanism kill-test이며 ODE-Edit의 최종 성능 또는
  paper claim을 확정하지 않는다.

### Repo/protocol에서 확인한 사실

- 선행 pair audit에는 exact
  `MV2 PREPARE` verdict가 정확히 1개 있다.
- canonical spec은 model당 12 case, salted rank `[100:112]`, seed `17`,
  `q=1/256`, `h=1/2`, MEMIT layer `4,5,6,7,8`을 잠근다.
- 실행 identity는 Llama/Qwen 두 child를 하나의 2-GPU parent allocation에서
  동시에 시작하는 one-shot pair다.
- server1 session, runtime, offline, resource cap 및 Git exclusion은 아래
  기계 검사와 일치한다.

### GH 추정

- 두 `srun`을 같은 parent allocation에서 연속 background launch하고 그 뒤
  `wait -n`하는 구조는 현재 wrapper가 제공할 수 있는 최소 same-start
  보장이다.
- 이 preflight의 PASS는 실행 재현 경계가 닫혔다는 뜻이며, refresh의 과학적
  유효성이나 기대효과 크기를 예고하지 않는다.

### 사용자 확인 필요

- 이 exact pair 한 건은 사용자가 Llama/Qwen 동시 제출과 빠른 Motivation
  검증을 명시했으므로 추가 확인이 필요하지 않다.
- retry, case 확대, resource 증가 또는 다른 session/server 실행은 이
  preflight 범위 밖이며 새 gate가 필요하다.

## 필수 체크 결과

### 선행 gate와 untouched selection metadata

- `audits/global/2026-07-31-mv1mix-untouched-pair-v1.postrun.md`:
  exact `MV2 PREPARE` verdict 1개, ambiguous verdict 없음.
- 두 untouched manifest의 `selected_case_ids`는 각 20개이며 배열이 exact
  동일하다.
- 두 manifest의 selection `manifest_id`, `order_hash`, source `sha256`,
  `split_hash`가 각각 exact 동일하다.
- 각 summary의 `selection_manifest_id`는 대응 manifest와 exact 일치한다.
- 두 run 모두 planned/attempted/pass `20/20/20`, failure `0`,
  `expected_counts_exact=true`, `all_rollbacks_exact=true`,
  `git_output_written=false`다.
- case ID 값 자체와 outcome 수치는 열거하거나 이 audit에 기록하지 않았다.

### session, repository, tracked blob

- CWD: `/mnt/raid5/janghj/ODE-edit`.
- session boundary:
  repository `hyunjun1127/ODE-edit`, server `server1`, Codex session
  `019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2`; checker `PASS`.
- local Git role: `head-server1-gh` / `global-head` / `server1`.
- branch `main`; 점검 시 HEAD `51d85971f825b237d7a79320c9c8eb9cf9cf92bd`
  및 `origin/main` exact 동일.
- MV-2 runner/analyzer/lineage/trajectory/bridge, 네 test, child/parent/helper
  wrapper, implementation spec는 모두 tracked이고 working blob이 HEAD blob과
  exact 동일하며 해당 path의 staged/unstaged diff가 없다.
- 전체 worktree는 아직 clean이 아니다. untouched 결과·pair audit가
  untracked이고 `messages/inbox/server1.md`가 수정 중이며, 이 audit도 생성
  직후 untracked다. 따라서 submit helper는 현재 fail-closed다.
- GH는 untouched results/audits/inbox와 이 preflight를 검토해 commit하고
  `origin/main`에 push한 뒤, 제출 직전에 전체 clean 및
  `HEAD == origin/main`을 다시 확인해야 한다.

### runtime과 offline 경계

- `EASYEDIT_ROOT=/mnt/raid5/janghj/EasyEdit`
- `KR_UV_BIN=uv`
- `HF_HOME=/mnt/raid5/janghj/.cache/huggingface`
- `HF_HUB_CACHE=/mnt/raid5/janghj/.cache/huggingface/hub`
- `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`
- `PYTHONPATH=/mnt/raid5/janghj/EasyEdit`
- child wrapper는 `uv run --offline --frozen --no-sync`, cache/download 차단,
  fixed session/CWD/model/run/job identity, tracked-clean blob을 재검사한다.

### selection, code, syntax

- focused MV-2 unittest: `39/39 PASS`.
- exact
  `MV2SelectionAndEnvelopeTests.test_next12_is_deterministic_and_disjoint_from_first100`:
  `PASS`; next 12의 결정성, exact rank slice, first 100과 disjoint를 검사한다.
- MV-2 Python runner/analyzer/lineage/trajectory/bridge 및 네 test:
  `py_compile PASS`.
- child sbatch, pair sbatch, submit helper: `bash -n PASS`.

### Slurm과 resource

- child 및 parent 각각 `sbatch --test-only --export=NONE PASS`; 실제 제출 없음.
- parent: `2 GPU`, `2 task`, task당 `8 CPU`, `130000M`, `08:00:00`.
- 각 child step: `1 GPU`, `1 task`, `8 CPU`, `65000M`; Llama와 Qwen은
  exact model/run ID로만 허용된다.
- parent가 두 child를 같은 allocation에서 background로 시작하며,
  한 child 실패 시 sibling을 종료·회수하는 fail-fast가 있다.
- `check-slurm-resource-cap.sh server1 2 130000`:
  GPU `2 <= 3`, memory `130000 <= 396234 MiB`, `ALLOW`.
- 두 output directory와 one-shot marker는 모두 absent이고, 동명 active
  job 수는 `0`이다.

### artifact와 credential 경계

- `local/results`, `local/logs`, `local/state`는 Git ignore가 적용된다.
- `servers/local/`도 Git ignore가 적용되며 tracked file이 없다.
- `local/`의 tracked file은 안내용 `local/README.md` 한 개뿐이고 raw
  artifact는 tracked되지 않는다.
- MV-2 code/spec/wrapper의 private-key, access-key, password/token/secret/cookie
  literal scan은 `PASS`다.
- credential 파일, raw outcome row, `direct_z` artifact는 열람하지 않았다.

## Authorized scope

- 선행 audit와 이 audit가 각각 exact verdict 1개를 유지하고 전체 repo가
  clean이며 `main == origin/main`일 때에만
  `submit_session01_mv2refresh_pair_server1.sh` 한 번을 허용한다.
- child direct submit, 개별 model submit, run ID 변경, output 재사용,
  EasyEdit/cache/dataset/covariance/projector write는 금지한다.
- 실행 후 raw artifact는 `local/`에 유지하고 Git에는 compact metadata,
  model별 독립 분석, pair red audit와 재현 instruction만 남긴다.

## 중단 조건

- exact gate가 없거나 중복됨
- session/CWD/origin/branch/tracked blob/clean boundary 불일치
- untouched selection metadata의 model 간 불일치
- next-12 overlap, 누락, 중복 또는 fixed identity 변경
- output directory, one-shot marker 또는 동명 job 존재
- GPU/memory cap 초과, child 비동시 실행, fail-fast 상실
- online access, cache recompute/download, projector load, credential 노출
- receipt-before-outcome, equal-C, lineage, rollback/RNG test 또는 runtime
  technical gate 실패

- 최종 판정: `PASS` — locked MV-2 Llama/Qwen 동시 pair 한 건에만 유효
