# ODE-Edit

`ODE-Edit`는 sequential knowledge editing을 한 번의 고정 weight jump가 아니라,
현재 model state에서 target과 layer별 write direction을 다시 계산하는 edit
trajectory로 다루는 연구 저장소다.

현재 primary design은 **ODE-BF Cold-FR-E8**이다. 최신 proposal과 완료된 실험은
다음 방향을 지지하지만, 아직 `ODE-BF Full`, formal CBF safety 또는 lifelong
superiority를 확립하지 않았다.

## 현재 방법 한눈에 보기

```text
W0의 z_base에서 cold start
        ↓
target-only bootstrap은 joint clock 밖에서 1회
        ↓
request별 shared terminal residual을
6개 controller context와 5개 writer layer에 공통 적용
        ↓
K=8, h=1/8의 고정 Euler joint write로 tau=1까지 진행
        ↓
target-new NLL progress + Neutral/Soft layer routing
        ↓
W64 low-rank solve + BF16-authoritative virtual transaction
        ↓
terminal endpoint만 atomic commit 후보
```

핵심 설계 원칙은 다음과 같다.

- Native direct-z는 cold controller 초기값으로 사용하지 않는다.
- 모든 writer layer에는 같은 request-specific full residual을 제공하고, routing이
  실제 layer 분배를 결정한다.
- first-hit은 기록하지만 fixed-E8 design-validation run을 조기 종료하지 않는다.
- 임의의 historical/pretrained H/P budget은 scientific hard veto로 사용하지 않는다.
  H/P는 soft routing signal 또는 raw audit observable이다.
- 실제 model parameter와 persistent history는 inner trajectory에서 변경하지 않는다.
- efficacy, generalization, locality, new/old NLL, layer concentration, realization
  fidelity, capacity와 compute를 step별로 함께 기록한다.

현재 구현 중인 common cold-coordinate gate는 absolute target replacement와
layer별 `z-H_l` residual을 제거하고, canonical terminal residual을 context와
writer layer에 additive하게 공유한다. 이 gate의 결과가 나오기 전에는 해당 설계를
완료된 scientific method로 취급하지 않는다.

## 현재 evidence

### 확립된 기술 기반

- Llama/Qwen genuine B10에서 W64 reduced solve certificate를 검증했다.
- 동일 W64 virtual endpoint와 committed endpoint의 parameter bytes, logits와
  event identity를 검증했다.
- injected rollback과 최종 W0 restore가 exact였다.
- Native dense와 W64는 BF16 byte-exact하지 않으므로 W64를 Native의 exact
  replacement라고 부르지 않는다.

### 가장 강한 완료 결과

Warm target initialization을 사용한 no-budget `FR-A8-NEWNLL-ALLOFF`의 matched
B10에서 두 모델 모두 `Eff 10/10`, `Gen 20/20`, `Loc 80/100`을 기록했다. 이
결과는 hard H/P veto가 under-edit를 만들 수 있음을 보여주는 강한 causal reference지만,
cold fixed-E8 최종 method의 성능 증거는 아니다.

최신 완료 cold fixed-E8 R8에서 Llama Neutral은 `Eff 8/10`, `Gen 12/20`,
`Loc 89/100`이었다. Qwen Neutral trajectory는 tau=1까지 완료했지만 Soft routing의
수치 certificate failure가 post-freeze panel을 막아 paired endpoint 지표가 남지 않았다.
따라서 common cold-coordinate 수정 후 same-seal 재실행이 현재 다음 gate다.

Historical-H benefit은 아직 검증되지 않았다. 관련 B10-1 실험의 active history가
비어 있었기 때문에 sequential claim은 ordered B10-2 이상에서 별도로 검증해야 한다.

## Follow-up 읽기 순서

1. [현재 ODE-BF proposal](project/proposals/ODE_BF_Dynamic_Layer_Proposal.md)
2. [Main-table 승격과 계산량 절감 future work](plans/global/2026-08-08-ode-bf-main-table-and-compute-future-work.md)
3. [전체 실험 파이프라인과 실행 계보](experiment-reports/global/2026-08-08-ode-bf-experiment-pipeline.md)
4. [SH1/SH2 실험 종합 리뷰](experiment-reports/global/2026-08-08-ode-bf-sh-experiment-review.md)
5. [저비용 coefficient-space 대안 ODE-Alloc](project/proposals/00.ODE_Alloc_Proposal_Report.md)
6. [초기 BF-ODE-Edit proposal 원문](project/proposals/00.proposal.md)
7. [관련 연구와 novelty boundary](project/proposals/sections/02-related-work-and-novelty-boundary.md)
8. [운영 규칙](PROTOCOL.md)

## Repository map

- `project/proposals/`: 현재 proposal, 역사적 원문, method/related-work sections
- `project/run_scripts/ode_bf/`: ODE-BF runtime, routing, transaction, evaluator와 tests
- `project/run_scripts/ode_alloc/`: coefficient-space ODE-Alloc component track
- `experiment-reports/global/`: raw-free 실험 결과와 causal interpretation
- `plans/global/`: preregistered experiment contract
- `scripts/`: session/resource/static gate utilities
- ignored `local/`: raw result, full log, dataset, checkpoint와 private runtime state

## Git과 실행 경계

Git은 proposal, source, lock, compact receipt와 report를 위한 control plane이다.
Model/GPU/Slurm 실행과 raw artifact는 ignored execution plane에 둔다. Active experiment
worktree의 미완성 source는 main에 섞지 않으며, 완료 checkpoint도 source/history가
정리되고 필수 gate를 통과한 뒤에만 main으로 승격한다.

현재 main documentation은 최신 proposal과 완료 evidence를 안내한다. 완료된 R8
실험 source 계보와 진행 중인 common-coordinate R10은 별도 branch에서 보존하며,
R10 terminal review 전에는 active 변경을 main으로 가져오지 않는다.

원격 서버 접속 정보, raw IP, username, port, key, token, password와 private dataset
secret은 저장소에 기록하지 않는다.
