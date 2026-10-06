# Scientific W&B 공통 지표 구현

Nonce `USER-GH-ALL-SH-WANDB-METHOD-METRICS-20261007-SERVER1`.
Owner SH1/session `01a04939-f93a-7b50-bca0-65438eab2062`.
구현 source **`930e46531f38d5fa97a6d860275db73978c7daf4`**.
상태: **IMPLEMENTED_CPU_VERIFIED / ACTUAL_ONLINE_NOT_RUN**.

실제 root CWD `/mnt/raid5/janghj/ODE-edit`, 과거 registry29e4는 사용하지 않았다.
전용 WT `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-wandb-method-metrics-20261007`.
원 root dirty/완료 GPT2 job·source·asset은 변경하지 않았다.

## 구현과 호출 계약

- `schema.COMPARISON_SCHEMA='price-first2k-scalar-v1'`, SH4 `tracking.contract_ready()` PASS.
- current/pre, current/post, all_seen/post, W0_first2000 및 w0/current/N,
  w0/all_seen/N을 구분하는 strict whitelist. R/P/N 각9 scalar,
  pct0..100, 정수 count/success_count, NLL nats, true−new margin 산술 검사.
- performance payload의 edits 및 pre/post state 검사. W0_first2000은 x=0,
  R2000/P4000/N20000만 허용. 다른 W0 horizon은 새 명시 schema 등록 필요.
- eval wildcard는 `step_metric='edits', step_sync=False`;
  fit/optimizer는 `fit/global_candidate`와 별도 batch/candidate를 사용한다.
  batch 리셋으로 감소한 fit 축은 거부, 동일 endpoint의 별도 관측은 허용한다.
- harmonic: pct 입력 `3/sum(1/p)`, fraction 입력 `300/sum(1/r)`.
  missing이면 omit(None), 측정된0이면0. TF strict 조화평균으로 바꾸지 않는다.
- scientific config model/model_family/writer/role/metric_schema/config_sha 필수,
  기존 source/job IDs 유지. model alias는 llama3/gptj/qwen/gpt2xl 등록형.
  role/schema/source URL/observation SHA는 typed allowlist, secret/raw 필드 거부.
- 부모의 `<spool>/identity.json`은 atomic create-once. run UUID/URL/name 및
  job/source/config/model/schema와 startup 확인을 가변 `receipt.json`과 분리한다.
  이전 identity 덮기 실패는 startup block, runtime log/finish 오류는 과학예외와 분리한다.
- finish에서 마지막 실제 eval/fit 각 최대1 row만 한정 API readback한다.
  `edits`/key/value/config/name 검산 범위를 receipt에 명시한다.
  미수신/오류는 UNVERIFIED이며 SDK accepted/flush 또는 uploader 종료를
  full history 전달이나 scientific completion으로 표기하지 않는다.

`init(env_file=..., spool=..., config=...)` API 유지. SH4는 준비한 caller와
이 helper commit을 **같은 신규 source freeze에 결속**한다. scalar mapping이
올바른지/원 sample identity가 일치하는지는 producer 책임이며 helper가 scalar
count만으로 문항 identity를 증명하지 않는다. 추가 forward·logging용 모델평가0.
asset/setup의 legacy scalar caller는 유지하며 GPT2 자산 준비에 과학 schema를 소급하지 않는다.

## 실제 CPU 검사와 재사용 근거

SDK 격리환경 Python:
`/mnt/raid5/janghj/ODE-edit/local/wandb-setup/20261006/venv/bin/python` (wandb0.30.0).
기존25 + 신규12 = **37 tests PASS**, skip0. GPU/model/Slurm/온라인 smoke0.
현재 SDK define_metric/scan_history signature도 CPU introspection으로 확인했다.
재현:

```sh
python -m unittest project.run_scripts.experiment_tracking.test_tracking project.run_scripts.experiment_tracking.test_job_identity project.run_scripts.experiment_tracking.test_method -v
```

신규 검사는 production SH4 reducer·batch_values·w0_subset·contract_ready와
공통 worker.session/부모 receipt 경로를 호출한다. 입력은 synthetic scalar row다.
B4/B5 current 모두 R100/P200/N1000, B4 all_seen key 없음,
B5 all_seen R500/P1000/N5000, N의 true token count/ACC 선택을 확인했다.
마진·단위·harmonic missing/zero, W0 잘못된 horizon, 평가 상태축,
fit 리셋, array task0/signed step, unknown/private config, immutable receipt,
queue-full 축 비변이, readback 누락/오류를 검사했다.

실제 B4/B5 원 raw를 server1로 새로 전송하거나 다시 열지 않았다.
기존 SH4 owner CPU raw 검산을 다음 tracked receipt로 재사용했다:
`audits/servers/server4/jlz-price-gptj-2k/tracking-compatibility.json`,
SHA `d60740039517083828cd3d3fe8f6bd6bf2a8aa7b1f6ba6a09f57e867b6fb4d1e`.
해당 receipt의 raw/stored equality와 이번 local synthetic integration을 구분한다.
독립 reviewer 미사용(owner 직접 검사), 실제 online/next-run readback은 **NOT_RUN**.

## 게시·경계

GH와 SH4 handoff: 위 helper commit/API/capability를 채택하고 SH4 local raw 검사를
새 helper와 재결속할 수 있다. 이 지시는 새 science submit 권한이 아니다.
Frozen/PENDING/running/과거 run rename·backfill·comparison snapshot 생성0.
control 정본/SH4 caller 및 과학 coefficient/input/evaluator 수정0.
NO_BROADCAST_NOT_REQUIRED. Git에는 소형 source/test/receipt/report만 게시한다.
