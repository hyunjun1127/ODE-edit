# Repository-managed Run Scripts

이 디렉터리가 ODE-Edit의 tracked 실행 script와 wrapper의 canonical
location이다. root `run-scripts/`는 사용하지 않는다.

Session 01 Motivation Validation의 첫 실행 gate는
`ode_edit_motivation.mv0_fidelity`다. 이 runner는 두 고정 model snapshot,
EasyEdit source, CounterFact, 기존 MEMIT Wikipedia moments, 기존 AlphaEdit
projector의 byte identity를 먼저 검사한다. Projector는 MV-0에서 deserialize하지
않으며, moments cache miss/download/recompute는 차단한다.

Model을 올리지 않는 preflight:

```bash
/usr/local/bin/uv run --offline \
  --project /mnt/raid5/janghj/EasyEdit --frozen --no-sync \
  python -m project.run_scripts.ode_edit_motivation.mv0_fidelity preflight \
  --easyedit-root /mnt/raid5/janghj/EasyEdit \
  --model llama3-8b-inst
```

server1의 tracked Slurm envelope는 `session01_mv0_server1.sbatch`다. 고정
자원은 `devbox`, `gpu:a6000:1`, CPU 8, `65000M`, 6시간이다. 이번 승인에서
model/run ID/case 수는 각각 `llama3-8b-inst`,
`mv0_llama_smoke_v1`, `1`로 fail-closed 고정한다. Slurm은
`--export=NONE`으로 제출해 interactive session의
credential/environment를 상속하지 않는다. 제출은
`submit_session01_mv0_server1.sh`만 사용하며, 이 helper가 exact argument,
session boundary, resource cap, red-team gate, pushed-clean `origin/main`,
exclusive output, log directory를 allocation 전에 확인한다. Submit 직전
`local/state/slurm-submissions/`에 원자적 one-shot marker를 만들며, 제출이
실패하더라도 새 audit 없이 marker를 지우거나 재시도하지 않는다.

```bash
project/run_scripts/submit_session01_mv0_server1.sh \
  llama3-8b-inst mv0_llama_smoke_v1 1
```

모든 raw event, frozen direct-z, generated context, full log는 `local/`에만
남긴다. Git에는 compact manifest/metric summary/한국어 분석 report와
재현 명령만 승격한다. Model별 smoke가 통과하기 전 case 수를 늘리거나 다음
motivation diagnostic으로 진행하지 않는다.

CPU-only contract/import test:

```bash
/mnt/raid5/janghj/EasyEdit/.venv/bin/python -m unittest discover \
  -s project/run_scripts/ode_edit_motivation/tests -p 'test_*.py' -v
```
