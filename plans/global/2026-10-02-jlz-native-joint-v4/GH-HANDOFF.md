# GH 인계: JLZ v4 method + A/B 2k 실험 설계

2026-10-02 KST. Scope: v4 두 arm만. **설계 전달 패키지이며 생산 runner·GPU 검증·실험 제출 완료를 뜻하지 않는다.**

사용자 요청을 다음 세 가지로 반영했다.

1. 검토한 연산 최적화를 method 설계에 반영한다.
2. Method 설계문과 실험 설계문을 별도로 제공하여 GH가 함께 SH4에 넘길 수 있게 한다.
3. Method는 가변 logical batch size와 model/benchmark adapter를 사용한다. BS100/Llama3/CounterFact는 이번 실행 profile이다.

## 읽는 순서와 정본

| 문서 | 역할 |
|---|---|
| `method-ko.md` | 연구 목적·일반 수식·공동 계산 그래프·write·최적화·불변조건 |
| `portability-contract.json` | 가변 B/층별 차원·model/benchmark interface·capability/fallback |
| `contract.json` | 이번 Llama3 profile을 결속한 method machine contract와 compute-r1 |
| `docs/methods/jlz-native-joint-v4.tex` | 같은 method의 TeX source |
| `experiment-2k/experiment-ko.md` | 이번 A/B BS100×20의 pilot/main·평가·자원·완료 조건 |
| `experiment-2k/experiment.json` | 이번 실행의 machine contract |
| `experiment-2k/gh-sh4-handoff-ko.md` | GH→SH4용 실행 담당·범위 인계문 |
| `handoff/package-manifest.json` | 위 파일과 CPU 근거·검증 코드·작은 source 의존성의 SHA/size |

`compute/reviewed-optimization-proposal.txt`는 검토한 원 제안이며 실행 지시의 정본이 아니다. 채택·보류·조건은 최신 method/contract를 따른다.

`experiment-ko.md`를 method 정본으로 취급하지 않는다. 실험의100/20/2k/평가분모는 instance 값이며 일반 method의 shape 상수가 아니다. Method 변경은 method/TeX/contract에 먼저 반영하고 실험 profile을 연결한다.

## 현재 합의된 실행

- A eta=0, B eta=1만 새로 실행. 각자 W0/H0에서 같은 첫2k를 BS100×20으로 처리한다.
- 두 arm 모두 전체 eligible L4–L8, native6+KL1,25후보/24Adam, 마지막 유한 clamped 후보, 동일한 공동 step schedule.
- 작은 A/B pilot 및 BS100 timing 후 main. Baseline 신규 main/pilot/fallback 없음. 기존 결과는 선택적인 읽기 전용 참고다.
- 성능·집중도·미수렴은 진행 gate가 아니다. 입력/gradient/state/commit 오류는 수정한다.
- SH4 project cap2/job1GPU/host60416MiB 기준으로 최신 점유를 재확인. 영구 checkpoint/복원 동등 payload 없음.

## 최적화 구현 요구

Direct delta oracle → batched selected full-vocabulary head/crop → 정확한 causal prefix cache/entry fusion → 같은 A의 factor 재사용 → full-token layer streaming/history key 재사용 순서다. 각 capability의 reference parity를 확인해 활성화하고 실제 route를 기록한다. 빠른 경로를 검증하지 못하면 동일 method의 reference 경로를 사용한다. 목적·층·문장·예산을 줄이는 것은 이 fallback에 포함되지 않는다.

큰 B에서는 primal/dual backend 또는 exact tiled operator로 Q의 모든 요청 결합을 유지한다. 마지막 partial batch는 실제 B_t로 처리한다. 모델별 causal/alias/write-map 조건이 다르면 해당 cache를 끄고 full reference를 쓴다. Native intervention과 write의 additive 연결 자체가 없는 모델에는 별도 adapter가 필요하다.

이번 CPU 근거는 입력/token 계수와 작은 FP64 causal-attention/geometry 검산이다. 실제 Llama FP32 kernel 정합, GPU 가속률, 다른 모델·benchmark 성능은 아직 검증하지 않았다. 숫자25.55%,6.58%,1.70GiB는 이번 profile의 처리량/저장량이며 속도 향상 배율이 아니다.

## 검증과 전달 방식

패키지 archive의 경로와 SHA는 `handoff/archive-receipt.json`에 기록한다. Archive는 repository-relative 경로를 유지하며, 모델·데이터셋·checkpoint·secret을 포함하지 않는다. 전용 디렉터리 또는 검토용 worktree에 풀어 SHA를 확인하고 필요한 source/doc만 GH의 배포 경로로 승격한다. 수신 worktree의 기존 파일에 무조건 덮어쓰지 않는다.

원 worktree:
`/mnt/raid5/janghj/.codex/worktrees/odeedit-jlz-subject-rebuild-20261002-v1`

Repository root에서:

```bash
python3 plans/global/2026-10-02-jlz-native-joint-v4/handoff/verify_package.py
python3 plans/global/2026-10-02-jlz-native-joint-v4/experiment-2k/build_plan.py --check --dataset /data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json
python3 plans/global/2026-10-02-jlz-native-joint-v4/inputs/check_exact_native_inputs.py --dataset /data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json --contexts <same-SHA-contexts.json>
```

첫 명령은 archive 안의 파일만으로 검증한다. 뒤의 두 명령은 SH4의 기존 dataset/context 자산 경로와 원 SHA가 필요하다. `math/validate_native_joint.py`, `math/validate_compute_reuse.py`는 torch가 있는 CPU 환경에서 재현할 수 있다. `compute/count_native_work.py`는 이번 profile의 tokenizer-only 감사이며 모델 weight를 로드하지 않는다. 로컬 경로가 다른 경우 `--model`, `--dataset`, `--contexts`를 사용하며 `--output`에는 새 local 결과 파일을 지정해 봉인된 CPU 근거를 덮어쓰지 않는다. Archive SHA 영수증은 archive 외부의 sidecar이며 내부 manifest는 풀린 파일의 SHA를 검증한다.

실제 v4 CLI는 아직 구현되지 않았다. SH4가 source/asset/runtime closure와 실제 argv를 결속하고 pilot 검증 후 제출한다. 전달 준비·전달 수락·제출·실행·완료는 별도 상태로 보고한다. 이 패키지 작성이 GH에 실제 메시지를 전송했다는 뜻은 아니다.
