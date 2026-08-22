# S4-M1 server4 artifact path seal audit

## 판정

- server/agent: `server4` / `server4-server-head`
- task base: `9e1943d29e19bf1429a99ec86fce0b13c878b513`
- observed_at: `2026-08-22T18:22:14+09:00`
- artifact content: Llama `READY_REUSE`, Qwen `READY_REUSE`
- path seal / launcher: `BLOCKED_SHARED_SOURCE_CHANGE`
- scientific submit: `HOLD`
- data/source/model/GPU/Slurm mutation: 0

S4-M1은 physical path와 scientific content identity를 분리하려는 task다.
`/data/janghj/EasyEdit`의 content는 다시 검증됐지만, 현재 shared guard와 launcher는
server4 소유 local config만으로 physical root를 바꿀 수 없다. 지시대로 shared
scientific source를 수정하기 전에 차단했다. Symlink, copy/transfer, lock 수정,
projector/stats 재계산·다운로드·수정과 submit/cancel/retry는 수행하지 않았다.

## Focused content identity test

현재 BF/base lock의 member SHA와 size를 `/data/janghj/EasyEdit`의 실제 파일에
전체 rehash로 대조했다. Projector만 `torch.load(map_location="cpu",
weights_only=True, mmap=True)`로 metadata smoke했다. Model/GPU load는 없었다.

| alias | projector result | covariance result | accepted bundle identity |
| --- | --- | --- | --- |
| `llama3-8b-inst` | SHA/size MATCH; FP32 `(5,14336,14336)` | layer 4–8, 5/5 SHA/size MATCH | `2cd8517f65f476e3a8be4edddc81fc345ecd96a49634e41769bbacd4206af77e` |
| `qwen2.5-7b-inst` | SHA/size MATCH; FP32 `(5,18944,18944)` | layer 4–8, 5/5 SHA/size MATCH | `55a6c2557b3a89f6880a7c56f65fb2487bafee53476dd248b57a3a48185a6638` |

Bundle hash는 accepted S4-M0 canonical contract다. 이번 focused test는 그
bundle을 구성하는 projector와 Wikipedia covariance member identity가 현재도
exact-match임을 확인했다. 두 alias의 revision, native model name, dimension,
projector SHA와 covariance SHA는 서로 달라 교환할 수 없다.

## Current resolution 및 negative tests

기존 host-specific mechanism은
`servers/templates/method-runtime.env` -> ignored
`servers/local/method-runtime.env`이며 `EASYEDIT_ROOT`/HF root를 담을 수 있다.
그러나 server4에는 active `method-runtime.env`가 없고, 더 중요하게 현재 ODE-BF
submit/sbatch/guard는 이 파일을 읽지 않는다.

Focused negative test 결과:

1. Canonical BF lock으로 `ODEBFArtifactGuard`를 구성하면 Llama/Qwen 모두
   missing `/mnt/raid5`에서 fail-close했다.
2. Base lock을 메모리에서 `/data/janghj/EasyEdit`와 server4 HF cache로 바꾸고
   `validate_artifact_lock_structure`를 호출하면
   `ODEAllocContractError: EasyEdit artifact root differs`로 fail-close했다.
3. `/mnt/raid5/janghj/EasyEdit`는 missing이고 `/data/janghj/EasyEdit`는 symlink가
   아닌 실제 directory임을 확인했다.

이 negative behavior는 안전하지만 server4 launcher readiness를 만들지는 못한다.

## Exact shared source/API blockers

### Base artifact guard

`project/run_scripts/ode_alloc/p0_artifacts.py`:

- `validate_artifact_lock_structure` lines 51–59가 legacy EasyEdit/HF absolute
  root를 정확히 요구한다.
- `P0ArtifactGuard.__init__` lines 140–149는 별도 runtime root override 없이
  lock root를 즉시 `resolve(strict=True)`한다.

필요 API는 lock의 logical/scientific identity를 유지하면서 physical root만
별도 seal에서 받는 optional fail-close override다. 예:
`P0ArtifactGuard(lock_path, alias, *, runtime_easyedit_root=None,
runtime_hf_hub_cache=None)`. Default `None`은 기존 `/mnt/raid5` semantics를 그대로
유지해야 한다.

### ODE-BF artifact guard

`project/run_scripts/ode_bf/artifacts.py::ODEBFArtifactGuard.__init__` lines
113–139는 BF lock의 EasyEdit/evaluator/HF root를 즉시 resolve하고 base guard에도
override를 전달할 API가 없다.

필요 API는 hostname/agent role과 server4-only path seal을 검증한 뒤 physical
roots를 BF/base guard에 일관되게 전달하고, alias별 member SHA/size와 bundle
identity를 model load 전에 검증하는 optional resolver/seal parameter다. Default는
기존 동작이어야 하므로 server1/server2와 legacy lock semantics는 바뀌지 않는다.

### Launcher integration

대표적으로
`project/run_scripts/session04_ode_bf_submit_p1_adaptive.py` lines 72–73,
594–599는 shared artifact locks를 고정해 guard/forecast에 전달한다. Session05
sbatch 계열 50개는 `readonly EASYEDIT_ROOT="/mnt/raid5/janghj/EasyEdit"`를
포함하며, 예시
`project/run_scripts/session05_ode_bf_p1r52_joint_pc_full_fp32_b100.sbatch`
lines 19, 33–40도 EasyEdit/HF root를 고정한다.

따라서 target launcher가 정해진 뒤 최소한 다음 shared 연결이 필요하다.

1. ignored server4 seal 또는 기존 `servers/local/method-runtime.env`를 fail-close
   resolver가 읽는다.
2. physical hostname, `agent.hostname=server4`, `agent.role=server-head`, canonical
   EasyEdit/HF roots를 검증한다.
3. 두 bundle identity와 alias/revision/layer/representation, P/stats SHA/size,
   projector shape/dtype를 model load 전에 검증한다.
4. server4일 때만 override를 선택하고 다른 host는 legacy default를 유지한다.

이 연결은 server4 소유 path만으로는 구현할 수 없으며 target scientific
launcher도 아직 특정되지 않았다.

## Seal state 및 handoff

- approved server4 physical root: `/data/janghj/EasyEdit`
- launcher-consumed canonical seal location: `NONE` (구현 전 차단)
- shared source changed: 0
- local seal created: 0
- launcher verdict: `BLOCKED`
- next owner: global-head

GH가 위 shared API와 target launcher 범위를 승인·지정하면 server4는 가장 작은
server4-only selection으로 구현하고 focused positive/negative tests를 다시 수행할
수 있다. 그 전까지 scientific submit은 `HOLD`다.
