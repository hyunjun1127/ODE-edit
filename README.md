# ODE-Edit

`ODE-Edit`는 sequential knowledge editing을 한 번의 고정 weight jump가 아니라,
현재 model state에서 target과 layer별 write direction을 다시 계산하는 edit
trajectory로 다루는 연구 저장소다.

현재 primary research direction은 **Fixed-z Functional Safe Write**다. 2026-08-30
research reset에 따라 이전 ODE-BF Cold-FR-E8 결과는 새 가설의 근거로 자동 승계하지
않는다. Same-z endpoint multiplicity, functional signal의 key-space 대비 추가 가치,
multi-step feedback의 static constrained solve 대비 필요성을 순서대로 반증 가능하게
검증한다.

## 현재 연구 방향 한눈에 보기

```text
Gate 0: frozen linear split = closed form 확인
        ↓
Gate 1: 동일 z를 만족하는 W 후보의 functional spread 검증
        ↓
Gate 2: key-risk를 통제한 functional signal의 추가 가치 검증
        ↓
Gate 3: native low-rank coefficient-space constrained writer
        ↓
Gate 4: static QP 대 relinearized feedback의 compute-matched 비교
        ↓
Gate 통과 시에만 Alpha/strong baseline 및 sequential 확장
        ↓
Gate 4 통과 시에만 CBF-ODE-Write 명칭 승격
```

핵심 설계 원칙은 다음과 같다.

- 기존 editor가 산출한 direct-z는 W-write 비교에서 고정한다.
- 단순 Euler subdivision은 negative control이며 contribution으로 보지 않는다.
- functional contract의 controller, calibration, gate-validation, final audit bank를 완전히 분리한다.
- KL의 initial zero-gradient 때문에 1차 CBF만 사용하지 않고 quadratic trust model과
  actual finite candidate check를 결합한다.
- 초기 구현은 full W-space가 아니라 약 5차원의 native factor coefficient space다.
- first-hit은 algorithmic endpoint일 뿐 canonical optimum으로 부르지 않는다.
- static constrained solver가 같거나 더 좋으면 ODE claim을 폐기한다.
- Gate 0--4 전에는 formal CBF, long-horizon 또는 lifelong superiority를 주장하지 않는다.

## Pre-reset evidence

아래 결과는 구현 자산과 historical negative evidence로만 보존한다. 새 proposal의 Gate를
통과한 scientific evidence가 아니며, 새 방법 성능으로 재해석하지 않는다.

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
Common cold-coordinate 수정 후 same-seal 재실행은 당시의 다음 gate였지만,
2026-08-30 research reset 이후 scientific execution priority에서는 내려갔다.

Historical-H benefit도 당시 검증되지 않았다. 관련 B10-1 실험의 active history가
비어 있었던 한계는 pre-reset limitation으로 남기며, 새 proposal의 sequential Gate를
통과하기 전에는 해당 claim을 재개하지 않는다.

## Follow-up 읽기 순서

1. [2026-08-30 current research reset proposal](project/proposals/2026-08-30-fixed-z-functional-safe-write-proposal.md)
2. [Proposal index와 pre-reset 문서 상태](project/proposals/README.md)
3. [이전 ODE-BF proposal](project/proposals/ODE_BF_Dynamic_Layer_Proposal.md)
4. [전체 pre-reset 실험 파이프라인](experiment-reports/global/2026-08-08-ode-bf-experiment-pipeline.md)
5. [SH1/SH2 pre-reset 실험 종합 리뷰](experiment-reports/global/2026-08-08-ode-bf-sh-experiment-review.md)
6. [이전 coefficient-space ODE-Alloc](project/proposals/00.ODE_Alloc_Proposal_Report.md)
7. [관련 연구와 novelty boundary](project/proposals/sections/02-related-work-and-novelty-boundary.md)
8. [운영 규칙](PROTOCOL.md)

## Repository map

- `project/proposals/`: 현재 proposal, 역사적 원문, method/related-work sections
- `project/run_scripts/ode_edit_method/`: low-rank factor, hook, transaction,
  controller/runtime와 tests
- `project/run_scripts/ode_edit_motivation/`: editor bridge, diagnostic math와
  pre-reset motivation/evaluator 자산
- `experiment-reports/global/`: raw-free 실험 결과와 causal interpretation
- `plans/global/`: preregistered experiment contract
- `scripts/`: session/resource/static gate utilities
- ignored `local/`: raw result, full log, dataset, checkpoint와 private runtime state

## Git과 실행 경계

Git은 proposal, source, lock, compact receipt와 report를 위한 control plane이다.
Model/GPU/Slurm 실행과 raw artifact는 ignored execution plane에 둔다. Active experiment
worktree의 미완성 source는 main에 섞지 않으며, 완료 checkpoint도 source/history가
정리되고 필수 gate를 통과한 뒤에만 main으로 승격한다.

현재 main documentation은 2026-08-30 research reset proposal을 primary로 안내한다.
Pre-reset R8/R10 source와 evidence는 별도 역사 계보로 보존하며, 새 Gate의 근거로 자동
승계하지 않는다.

원격 서버 접속 정보, raw IP, username, port, key, token, password와 private dataset
secret은 저장소에 기록하지 않는다.
