# Session 01 MV-0 실행 전 red-team gate 및 GH 직접 제출 예외

- 작성일: 2026-07-30
- 작성자: `head-server1-gh` (`global-head`)
- 독립 검토: `red_execution_preflight`, `easyedit_impl_audit`,
  `mv0_runner_impl`
- 최종 판정: `PASS` — 아래 exact MV-0 smoke 한 건에만 유효
- 허용 run: `mv0_llama_smoke_v1`
- 허용 job: `odeedit_mv0_llama_smoke`
- canonical plan:
  `plans/global/2026-07-30-session-01-motivation-validation.md`

## 1. 네 범주

### Proposal에서 온 내용

- fixed direct-z 아래 layer proposal을 비교하려면 먼저 native EasyEdit MEMIT과
  repo-local adapter가 같은 update를 만드는지 검증해야 한다.
- evaluation field는 edit process에 들어가면 안 되며, 실패 case를 결과를 본 뒤
  denominator에서 제외하면 안 된다.
- 큰 Motivation 실험보다 작은 fidelity/neutrality gate를 먼저 통과해야 한다.

### Repo/protocol에서 확인한 사실

- `PROTOCOL.md:506-509`는 명시적 사용자 지시, emergency scheduling,
  time-critical exploratory work에 GH의 직접 Slurm 제출을 허용한다.
- 사용자는 이 task에서 실제 Motivation kill-test 실행을 명시적으로 지시했다.
- server1의 등록 GH session은
  `019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2`, CWD는
  `/mnt/raid5/janghj/ODE-edit`, repository identity는
  `hyunjun1127/ODE-edit`이다.
- server1은 `devbox`, project GPU cap 3, requested GPU당 host-memory request
  cap `198117 MiB`다.
- 별도 server-head Codex session과 다른 active ODE-Edit clone은 아직 없다.
- EasyEdit는 read-only external root이며 ODE-Edit code만 hook/adapter를
  제공한다.

### GH 추정

- 최초 1-case run은 연구 가설 검증이 아니라 implementation fidelity와
  trace-neutrality를 확인하는 최소 smoke다.
- server1 onboarding 전체를 완료하지 않고도 local GH host의 exact one-shot
  job은 사용자 지시 예외로 제한 실행할 수 있다.
- 별도 SH가 생기기 전에는 GH가 scheduler state와 local artifact integrity만
  최소 monitoring하고, 결과 해석과 post-run validity audit은 실행 구현을
  담당하지 않은 별도 agent에게 분리하는 것이 가장 보수적이다.

### 사용자 확인 필요

- 없음. 이 audit은 현재 사용자의 실제 실행 지시를 exact one-shot 권한으로
  해석한다.
- 재시도, Qwen smoke, case 2–3 확장, MV-1 진입은 이 승인에 포함되지 않는다.

## 2. 직접 제출과 onboarding waiver

일반 lifecycle에서는 SH가 제출·monitoring·보고·artifact broadcast를
담당한다. 현재 SH가 없으므로 다음 두 예외를 한 번만 적용한다.

1. **GH direct-submit exception**: 사용자의 명시적 실행 지시에 근거해 아래
   helper가 허용하는 Llama 1-case job 한 건만 GH가 제출한다.
2. **server1 limited-onboarding waiver**: local Slurm controller/node, exact
   CWD, runtime, source/data/cache path, session boundary, cap은 확인됐지만
   SSH/rsync mesh와 SH assignment는 미완료다. 이 미완료 항목은 local-only
   smoke에 필요하지 않으므로 이번 한 건에 한해 waive한다.

이 waiver는 server1의 일반 제출 권한, 다른 model, 다른 repo, remote SSH,
rsync, external transfer 권한을 만들지 않는다.

## 3. 독립 implementation gate

### 해결된 blocker

- ordered MEMIT path는 frozen direct-z와 exact precomputed covariance pin을
  우회하지 않는다.
- covariance miss, dataset load, force-recompute는 즉시 중단된다.
- EasyEdit raw factor의 `adj_k @ resid.T`, shape match, `.float()` add 순서를
  `TemporaryExactMemitApplication`이 재생한다.
- model mode, `use_cache`, CPU/CUDA RNG, `requires_grad`, global context,
  `compute_z`, `COV_CACHE`, target weights를 복원한다.
- 승인된 EasyEdit source byte만 SHA/size를 재검사한 뒤 직접 compile하며
  `.pyc`를 읽거나 쓰지 않는다. 승인되지 않은 `easyeditor*` import와
  preloaded module은 차단한다.
- import 또는 post-import provenance 검사가 실패하면 bridge binding,
  finder, `easyeditor*` module을 모두 제거한다.
- preflight 직후 source가 사라지는 TOCTOU에서도 inert namespace와 finder를
  제거하고 unverified code 실행 없이 중단한다.
- EasyEdit proposal 호출 동안 Python `random`, NumPy, Torch CPU/CUDA RNG를
  보존·복원한다. Import는 model/random seed 설정 전에 수행한다.
- ordered와 synchronous proposal은 entry model state를 공유하되 서로 다른
  semantics/provenance를 명시한다.
- generated context 원문은 compact manifest/JSONL에 저장하지 않는다.
- fatal contract/state/rollback/recompute/OOM/OSError는 sanitized failure
  event를 기록한 뒤 즉시 중단한다.
- summary는 planned, attempted, not-run denominator를 모두 보존한다.
- `KeyboardInterrupt`와 `SystemExit`는 일반 technical failure로 삼키지 않는다.

### 검증

```text
/mnt/raid5/janghj/EasyEdit/.venv/bin/python -m py_compile \
  project/run_scripts/ode_edit_motivation/*.py \
  project/run_scripts/ode_edit_motivation/tests/*.py
exit=0

/mnt/raid5/janghj/EasyEdit/.venv/bin/python -m unittest discover \
  -s project/run_scripts/ode_edit_motivation/tests -p 'test_*.py' -v
Ran 38 tests
OK
```

독립 `mv0_runner_impl` 재감사 결과는 `PASS`, 잔여 blocker 없음이다. 독립
`easyedit_impl_audit`는 최종 commit 전 dirty-worktree gate가 의도적으로
실행을 막는 것을 확인했다. 제출 helper는 tracked clean `main == origin/main`
조건을 충족하기 전 `sbatch`에 도달하지 않는다.

실제 pinned EasyEdit tree에 대한 model-free import 검증에서는 approved
source 13개만 closure에 들어왔고, 전부 `_VerifiedSourceLoader`,
`__cached__=None`이었다. 두 번째 `load()`는 같은 verified binding을
재사용했으며 다섯 package namespace는 inert였다.

## 4. 고정 input과 runtime preflight

Model을 올리지 않은 full size/SHA-256 preflight 결과:

| model | provenance ID | selection manifest ID | 결과 |
| --- | --- | --- | --- |
| `llama3-8b-inst` | `273d06af367904aac61931331e6aa2ea09204b0db1f6c7557bca16a9e2c84d2b` | `186372487e0a79d64a4895dfa5ad142a5e9b1dd3f675c9a3468205147c38c4ce` | PASS |
| `qwen2.5-7b-inst` | `d247b27edf638b7a9625954e8bf2aeac745b61247230f8bcf733e991a3ed9d16` | `186372487e0a79d64a4895dfa5ad142a5e9b1dd3f675c9a3468205147c38c4ce` | PASS |

각 scope는 다음을 full hash/size로 검사했다.

- algorithm-critical EasyEdit source 13개
- `pyproject.toml`, `uv.lock`, `.python-version`
- 양 model MEMIT YAML
- CounterFact 21,919 rows
- 해당 model의 precomputed Wikipedia `mom2` 5개
- 해당 model의 precomputed AlphaEdit projector 1개

Projector는 MV-0에서 deserialize하지 않는다. Wikipedia moments는
read-only로 load하며 download/recompute하지 않는다.

고정 runtime:

```text
Python 3.12.0
torch 2.9.1+cu128
CUDA runtime 12.8
transformers 4.57.1
numpy 2.2.6
accelerate 1.13.0
```

버전 mismatch, model/tokenizer revision mismatch, non-float32, 둘 이상의 visible
GPU, network/cache miss는 모두 fail-closed다.

## 5. exact Slurm envelope

| 항목 | 고정값 |
| --- | --- |
| target | `server1` / `devbox` |
| partition | `gpu` |
| allocation | node 1, task 1, `gpu:a6000:1`, CPU 8 |
| host memory | `65000M` |
| time | `06:00:00` |
| model | `llama3-8b-inst` |
| cases | 1 |
| seed | 17 |
| selection seed | `ode-edit-motivation-counterfact-v1` |
| run ID | `mv0_llama_smoke_v1` |
| job name | `odeedit_mv0_llama_smoke` |
| raw output | `local/results/raw/session01_motivation/mv0_llama_smoke_v1/` |
| local log | `local/logs/slurm/session01_motivation/%x-%j.{out,err}` |
| one-shot marker | `local/state/slurm-submissions/session01_motivation/mv0_llama_smoke_v1.submitted/` |
| tracked sbatch | `project/run_scripts/session01_mv0_server1.sbatch` |
| submit helper | `project/run_scripts/submit_session01_mv0_server1.sh` |

실제 제출 명령은 이것 하나다.

```bash
project/run_scripts/submit_session01_mv0_server1.sh \
  llama3-8b-inst mv0_llama_smoke_v1 1
```

helper는 allocation 전에 다음을 순서대로 검사한다.

1. helper와 in-job envelope 양쪽의 exact arguments:
   `llama3-8b-inst mv0_llama_smoke_v1 1`, exact job name
2. `agent.id=head-server1-gh`, `agent.role=global-head`,
   `agent.hostname=server1`
3. 관련 sbatch/helper/audit와 runtime 핵심 module이 Git tracked
4. branch `main`, untracked를 포함한 repo 전체 clean,
   `HEAD == origin/main`
5. 이 audit의 exact PASS line
6. exclusive run path와 same-name active job 부재
7. `scripts/check-session-boundary.sh
   019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2`
8. `scripts/check-slurm-resource-cap.sh server1 1 65000`
9. mode 700 local log/output root 생성
10. ignored local state root에 exclusive `mkdir`로 durable one-shot marker 생성
11. `sbatch --parsable --export=NONE` 한 번과 returned job ID를 marker에 mode
    600으로 기록

`sbatch --test-only`는 이 envelope를 `devbox`, partition `gpu`, CPU 8로
수용했고 실제 job은 생성하지 않았다.

## 6. credential, artifact, broadcast boundary

- Slurm은 `--export=NONE`으로 interactive credential/environment를 상속하지
  않는다.
- ignored env file은 `source`하지 않고 strict key/value parser와 allowlist로만
  읽는다.
- `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `WANDB_API_KEY`,
  `OPENAI_API_KEY`는 runner 전에 unset한다.
- `/usr/local/bin/uv --offline --frozen --no-sync`와
  `UV_PYTHON_DOWNLOADS=never`를 고정한다.
- `set -x`, environment dump, raw upstream exception text를 금지한다.
- raw event, generated direct-z, model/data/stats/projector, full log는
  `local/`에만 둔다.
- Git에는 job/commit/config/provenance ID, compact metrics, denominator,
  resource summary, 한국어 분석/audit만 남긴다.
- active peer clone/SH가 없으므로 이번 run은 artifact broadcast를 수행하지
  않는다. 임의 SSH/rsync 대신 post-run metadata에
  `broadcast_exception=no_active_peer_clone_or_sh`를 기록한다.

EasyEdit untracked example shell에서 발견된 literal HF credential은
복사·실행·전송하지 않는다. 해당 credential은 사용자가 별도로 rotate/폐기할
대상이다.

## 7. 중단 조건

Helper submit gate에서 다음 하나라도 실패하면 job을 만들지 않는다.

- session/CWD/repository/Git agent mismatch
- audit/helper/sbatch untracked 또는 `main != origin/main`
- untracked를 포함한 dirty worktree/index
- red PASS line 부재
- same run path/job 또는 durable submission marker 중복
- session boundary 또는 dynamic GPU/memory cap failure
- unsafe log path 또는 environment

Allocation 뒤에도 model을 load하기 전 Python preflight에서
source/data/cache/runtime identity를 다시 검사한다. 이 검사가 mismatch면
GPU allocation은 이미 생겼더라도 model weight는 올리지 않고 즉시 종료한다.

Runtime에서 다음 하나라도 발생하면 run은 `all_pass=false`로 즉시 중단한다.

- source/data/cache/runtime preflight mismatch
- model/tokenizer/runtime pin mismatch
- one-visible-GPU/float32/eval/`use_cache=False` mismatch
- download, dataset load, stats/projector recompute 요청
- provenance/direct-z/context/request/state mismatch
- forbidden evaluation field 또는 raw payload persistence
- non-finite factor/metric
- target weight concurrent mutation 또는 rollback mismatch
- CUDA OOM, scheduler cancel/signal, nonzero process exit
- trace-no-write/native-adapter equivalence failure

1-case smoke가 `exit=0`, `all_pass=true`, exact rollback, artifact firewall pass가
아니면 Qwen smoke, 3-case 확장, MV-1을 제출하지 않는다. relaxed flag,
recompute, download, overwrite, 같은 run ID 재사용으로 우회하지 않는다.
`sbatch` 자체가 실패하더라도 one-shot marker를 자동 제거하지 않으며, 새
red audit 없이 marker를 삭제하거나 같은 run을 재시도하지 않는다.

## 8. 후속 책임과 보고 경로

- GH 최소 monitoring:
  `squeue`/`sacct` state, compact stdout, output file existence/size/hash만 확인
- raw artifact:
  `local/results/raw/session01_motivation/mv0_llama_smoke_v1/`
- compact run metadata:
  `runs/mv0_llama_smoke_v1/`
- 독립 result analysis:
  `experiment-reports/global/2026-07-30-mv0-llama-smoke-analysis.md`
- 독립 post-run red audit:
  `audits/global/2026-07-30-mv0-llama-smoke.postrun.md`
- evidence index:
  `experiment-reports/global/2026-07-30-session-01-motivation-validation.md`

실행 담당과 결과 분석 agent를 분리한다. Qwen 또는 3-case 확장은 위 두 독립
post-run 문서가 smoke를 승인하고 새 cap/preflight가 통과할 때 별도 audit와
새 run ID로만 허용한다.
