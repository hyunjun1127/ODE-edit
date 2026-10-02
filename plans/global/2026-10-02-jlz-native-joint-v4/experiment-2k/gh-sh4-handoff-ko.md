# GH → SH4용 JLZ v4 2k 구현·실험 명세

2026-10-02 KST. Task ID: `jlz-native-joint-v4-bs100x20-20261002-v1`.

**이 파일은 인계 내용이며 직접 전달/수락/Slurm 제출 영수증이 아니다.**

패키지 진입점은 `../GH-HANDOFF.md`다. **일반 method는 `../method-ko.md`·`../portability-contract.json`·TeX**, 이번 실험은 `experiment-ko.md`·`experiment.json`을 함께 읽는다. Compute-r1 최적화는 method에 반영되어 있고 구현 정합 후 선택한다. Batch size·model·benchmark를 adapter/profile로 분리하며 이번 BS100/Llama3/CounterFact 상수를 공통 runner에 고정하지 않는다.

최신 범위 수정: **“v4 두 arm만 실험 돌리는 걸로 해”**. 신규 실험은 A/B만이다. Baseline 신규 main/pilot/fallback과 별도 baseline wrapper 구현을 모두 제외한다. 아래의 이전 담당 지정은 GH→SH4 역할만 유지하며, baseline 실행 범위는 이 최신 지시가 우선한다.

현재 사용자 요청: “실험 진행하도록 실험 설계 진행하라. 2000edit을 기준으로 일단 해보자.”
앞선 실행 담당 지정: “이거 간단한 pilot test 돌려보고 바로 bs100, 20 step sequential로 진행해서 baseline과 비교하는 것으로 진행하고 싶다. gh에게 task 전달시키고 sh4에게 진행시키도록 하는 것 user 명령으로 전달한다고 하자”. 이후 method를 재구축했으므로 **새 v4 명세**를 사용한다. 과거 v2 실험을 재시작하거나 이미 실행 중인 실험을 변경하는 지시가 아니다.

정본 worktree:
`/mnt/raid5/janghj/.codex/worktrees/odeedit-jlz-subject-rebuild-20261002-v1`

읽을 파일:

1. `plans/global/2026-10-02-jlz-native-joint-v4/contract.json`, `method-ko.md`, method artifact manifest.
2. 같은 경로의 `experiment-2k/experiment-ko.md`, `experiment.json`, `evaluation-schedule.json`, `case-schedule.csv`, artifact manifest.
3. `docs/methods/jlz-native-joint-v4.tex`.

GH는 이 파일들의 SHA를 검토하고 자신의 task/소스 배포 경로로 SH4에 제공한다. 본 설계 worktree의 다른 변경을 함께 게시하지 않는다. SH4는 전용 namespace `project/run_scripts/jlz_native_joint/`에 구현하고 현재 source/config/raw를 hotpatch하지 않는다.

핵심 조건:

- Fixed10k 첫 2k, BS100×20, native rewrite 6개 + KL 1개, 모든 L4–L8의 직접 delta 공동 Adam 25후보/24update.
- A eta=0, B eta=1의 추가 ridge optimal-value 비용만 목적 차이. R norm, full/subject 혼합, G/E prompt loss, 기존 SPG120은 사용하지 않는다.
- Fit 이후 delta를 고정하고 current-key local writer를 적용한다. 각 chain은 cold W0/H0에서 독립 시작하여 20 batch를 RAM에서 연속 실행하고, 모든 층 write 후 H를 append한다.
- 신규 2k chain은 A/B만이다. 기존 baseline 산출물은 읽기 전용 참고 비교만 허용하며 없거나 호환되지 않아도 새로 실행하지 않는다.
- BS2×1 두 arm pilot 및 B2 entry handoff 검증 → B100 timing(arm당 warm-up 1회 + 측정 3회) → main. 단일층 native parity는 구현 검증용이며 main 편집층을 일부만 선택하는 것이 아니다.
- 매 batch current 전량, W5/W10/W20 all-seen 전량 평가. 같은 endpoint의 중복 평가는 제거한다. Observer와 collector는 동일 schedule 파일을 사용한다.
- 낮은 strength/PS/NS, 집중도, 미수렴, 실현 잔차는 진행 중단 gate가 아니다. 잘못된 입력·gradient·상태·commit·nonfinite는 기술 오류로 수정하거나 rollback한다.
- SH4 기존 project cap 2, job당 1 GPU, host memory ≤ 60416 MiB이며 최신 점유를 확인한다. Lane 1은 A, lane 2는 B다. Server3와 다른 task는 변경하지 않는다.
- Checkpoint 및 복원 가능한 동등 payload는 저장하지 않는다. Raw 평가, 작은 telemetry, source/config/provenance를 저장하고 exact resume이 불가능함을 명시한다.
- Direct delta, batched full-vocabulary head, exact subject-prefix cache, 같은 A의 factor 공유, full-token layer streaming/history key 재사용을 구현한다. Capability가 성립하지 않거나 정합이 미확인된 경로는 동일 method의 reference로 대체하고 route를 기록한다. 후보/문장/층 축소나 다른 objective로 대체하지 않는다.

구현 전 확인할 경계:

- V4 생산 runner는 아직 없다. 과거 `jlz_two_arm.run`은 다른 method다.
- Baseline의 새 runner/wrapper/pilot은 이번 범위에 없다. Native input/loss 정의와 Tensor/tuple 호환 adapter는 A/B 기술 정합 확인에만 재사용할 수 있다.
- 실제 model/stat/tokenizer/source/runtime SHA, token identity, dry preparation 및 `--help`가 확인된 argv를 launch manifest에 결속한다. 아직 없는 CLI를 실행 가능한 명령으로 보고하지 않는다.

GH/SH4 보고 상태는 `DESIGN_RECEIVED`, `IMPLEMENTING`, `PILOT_TECHNICAL_PASS`, `RESOURCE_PENDING`, `SUBMITTED`, `RUNNING`, `COMPLETED/PARTIAL_FAILURE`를 구분한다. 접수만으로 제출이나 완료라고 하지 않는다. 최초 실행 보고에는 실제 job ID/argv/출력 경로와 B1 commit→B2 entry 또는 정식 resource pending을 담는다. 장기 agent polling/자동 heartbeat를 새로 만들 필요는 없다.

완료 조건은 A/B 각각20회 commit과 필수 평가다. 기존 baseline 참고 결과가 없어도 완료할 수 있다. 한 arm만 끝나면 partial로 보고한다.

CPU 설계 검증 명령(실제 존재하며 GPU/job 호출 없음):

```bash
python3 plans/global/2026-10-02-jlz-native-joint-v4/experiment-2k/build_plan.py --check
python3 plans/global/2026-10-02-jlz-native-joint-v4/inputs/check_exact_native_inputs.py
```

실험 품질과 실행 시간은 아직 측정되지 않았다. CPU 검증을 GPU qualification으로 대체하지 않는다.
