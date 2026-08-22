# S4-M1-R1 shared path-seal interface audit

## 판정

- server/agent: `server4` / `server4-server-head`
- implementation base HEAD/tree:
  `50e3af94f50b00cae24793d06a55c08b13b4b4e1` /
  `22eece0c16231cc22bf1e231133d757608f60923`
- observed_at: `2026-08-22T18:44:51+09:00`
- EasyEdit shared interface/seal: `PASS`
- HF mapping: `WITHHELD_EXACT_SET_MISMATCH`
- evaluator mapping: `WITHHELD_MISSING_CANONICAL_ROOT`
- full launcher: `BLOCKED_HF_EVALUATOR_MAPPING`
- scientific submit: `HOLD`
- EasyEdit/artifact/model/GPU/Slurm mutation: 0

GH가 승인한 범위에서 legacy scientific/content locks는 수정하지 않고 exact root
prefix remap interface를 추가했다. Default `None`은 legacy root 검증, raw/canonical
lock identity와 기존 receipt schema를 유지한다. Runtime seal receipt는 scientific
artifact receipt와 별도 객체이며 logical locked root, resolved runtime root, seal
identity, host/role, member manifest와 accepted bundle identity를 분리 기록한다.

## Canonical server4 seal

- tracked path: `agents/server4/alphaedit-runtime-path-seal.json`
- schema: `ode-edit-alphaedit-runtime-path-seal/v1`
- seal id: `ODEEDIT-S4-M1-R1-SERVER4-ALPHAEDIT-PATH-SEAL-V1`
- file SHA256:
  `176018298d691a1b67ae7ca787c7dcd2bd77d8606044790833d361fd451ddeb3`
- root digest:
  `00e99949cc53bb5a8ad85aafbff0965e26f0d1d7fd84783feed7a9ea5595d06e`
- physical hostname / Git hostname / role:
  `server4` / `server4` / `server-head`
- exact active mapping:
  `/mnt/raid5/janghj/EasyEdit` -> `/data/janghj/EasyEdit`

Arbitrary prefix substitution은 제공하지 않는다. Guard가 전달한 logical root가 seal의
exact legacy root와 다르면 거절한다. Runtime root는 absolute real directory여야 하며
root 또는 member path의 symlink, root 탈출, unresolved/special member를 거절한다.

| alias | bundle SHA256 | member manifest SHA256 | actual preflight |
| --- | --- | --- | --- |
| `llama3-8b-inst` | `2cd8517f65f476e3a8be4edddc81fc345ecd96a49634e41769bbacd4206af77e` | `5b982fd7d35cd33d7dde7fe483dfd1a48f765d1492491533d72a13db14cb020a` | 7/7 SHA/size/shape/dtype MATCH |
| `qwen2.5-7b-inst` | `55a6c2557b3a89f6880a7c56f65fb2487bafee53476dd248b57a3a48185a6638` | `915d87cd96cebee5eb2757dfa3d1734ba53ae267460b0b157be43f43a6a18a57` | 7/7 SHA/size/shape/dtype MATCH |

각 7개 member는 AlphaEdit hparams 1, null-space projector P 1, layer 4–8
Wikipedia covariance 5개다. Projector는 CPU mmap metadata로, NPZ covariance는
Numpy header만 읽어 shape/dtype를 확인했다. Model/GPU load는 없었다.

## Shared API

### `project/run_scripts/alphaedit_runtime_path_seal.py`

- immutable `AlphaEditRuntimePathSeal`, `RootMapping`, `ModelSeal`,
  `ArtifactMember`, `RuntimePathSealReceipt`
- `load_alphaedit_runtime_path_seal(path, repo_root=...)`
- exact `resolve_root`, legacy-lock member contract checks, alias preflight
- focused source-manifest verifier

### `project/run_scripts/ode_alloc/p0_artifacts.py`

- `validate_artifact_lock_structure(..., runtime_path_seal=None)`
- `load_artifact_lock(..., runtime_path_seal=None)`
- `P0ArtifactGuard(..., runtime_path_seal=None)`
- seal이 있으면 exact EasyEdit/HF logical root와 P0 model/covariance identity를
  비교하고 runtime root로 해석한다. Preflight receipt는
  `runtime_path_seal_receipt`에 별도 보관한다.

### `project/run_scripts/ode_bf/artifacts.py`

- `ODEBFArtifactGuard(..., runtime_path_seal=None)`
- seal을 EasyEdit/evaluator/HF root resolution과 base `P0ArtifactGuard`에 동일
  객체로 전달한다.
- ODE-BF alias/revision/layers/hparams/P identity를 seal과 비교하고 base
  preflight의 path receipt를 별도 전달한다.

기존 `ArtifactReceipt`와 `ODEBFArtifactReceipt` dataclass field는 변경하지 않았다.
따라서 path deployment identity가 기존 scientific receipt/canonical identity에
섞이지 않는다.

## Exact launcher/session integration

`project/run_scripts/session05_ode_bf_server4_artifact_path_dry_plan.py` 한 곳이
tracked server4 seal을 고정 경로에서 읽는다. CLI seal-path override는 제공하지
않는다. 실행 순서는 다음과 같다.

1. focused source manifest 검증
2. physical hostname, Git `agent.hostname`, `agent.role` 검증
3. legacy BF/base lock의 alias/member identity와 seal 비교
4. 두 alias의 actual member SHA/size/shape/dtype 검증
5. mapping readiness 계산 및 raw-free JSON receipt 출력

Dry-plan output은 EasyEdit logical/runtime path를 각각
`/mnt/raid5/janghj/EasyEdit`와 `/data/janghj/EasyEdit`로 기록했고
`model_load=false`, `gpu_use=false`, `slurm_submit=false`였다.

Focused source manifest:

- path:
  `project/run_scripts/ode_bf/locks/source_manifest_s4_m1_r1_server4_path_seal.json`
- file SHA256:
  `491ac1de132f92b4dc8e73e9808b4c1e3fef26c52c31ad7a7e94b75dbe2daa34`
- root digest:
  `10efecc8c03418793c819b06f2e7910a95ab17f825820bd9a7e9eebe7eafa539`

## HF read-only audit 및 mapping 보류

Candidate `/data/janghj/.cache/huggingface/hub`는 real directory, owner
`janghj:janghj`, mode `0700`이다. Base lock의 두 pinned revision에 대해 locked
member는 모두 존재하며 symlink target과 resolved size가 MATCH했다. SHA가 있는
member도 Llama 6/6, Qwen 7/7 MATCH했다.

그러나 current `P0ArtifactGuard.preflight`는 snapshot member name set의 exact
equality를 요구하며 candidate에는 다음 unsealed extras가 있다.

- Llama: `.gitattributes`, `LICENSE`, `README.md`, `USE_POLICY.md`, `original`
- Qwen: `.gitattributes`, `LICENSE`, `README.md`

따라서 identity 증거가 current exact-set contract를 만족하지 않아 HF mapping을
seal에 넣지 않았다. 다른 exact cache는 `/data/janghj` bounded search에서 없었다.
파일 삭제/이동/복사/symlink는 수행하지 않았다.

## Evaluator mapping 보류

Legacy BF lock은
`/mnt/raid5/janghj/00.KE/00.Experiment/00.EAIR_parametric/AlphaEdit` evaluator
root도 요구한다. server4에서 승인·검증된 canonical equivalent가 없으며 task의
승인 mapping 범위에도 포함되지 않았다. 따라서 이 mapping도 withheld다.

## Focused gates

- `py_compile`: PASS
- focused unittest: 10/10 PASS
  - default-none lock identity 및 legacy negative behavior
  - server4 seal positive와 P0/ODE-BF guard 동일 seal 전달
  - wrong host/role/logical root/runtime symlink/unresolved member
  - wrong SHA/shape/bundle/alias/locked member
- tracked seal actual full-member preflight: Llama/Qwen PASS
- focused source manifest: PASS
- target no-model dry-plan: PASS; `/data` resolution PASS
- `git diff --check`: PASS
- `scripts/check-agent-access.sh --all-changed`: PASS on isolated branch

## 최종 상태

- EasyEdit path seal/interface: `PASS`
- HF mapping: `WITHHELD`
- evaluator mapping: `WITHHELD`
- `LAUNCHER_READY=false`
- blocker: `BLOCKED_HF_EVALUATOR_MAPPING`
- scientific submit: `HOLD`

GH가 HF extra-member disposition과 evaluator canonical path identity를 별도로
결정하기 전에는 model load 또는 Slurm submit으로 진행하지 않는다.
