# Session 02 BF16 context-lock probe 구현 감사

판정: `PASS_FOR_GH_CODE_REVIEW / NO_GPU / NO_SCIENTIFIC_OUTCOME`.

## Boundary

| 항목 | 판정 | 근거 |
| --- | --- | --- |
| base | PASS | exact `26aa173a295682360c6747c55dd22f8c74d91548` |
| branch | PASS | `codex/odeeditsh1-bf16-context-probe-v1` |
| session/role | PASS | canonical SH1, server1, Sol Ultra |
| write scope | PASS | probe, no-submit sbatch, one focused test, own report/audit only |
| EasyEdit | PASS | read-only source bridge; authored diff 0 |
| GPU/Slurm/push | PASS | 0 / NO / NO |
| old artifacts | PASS | failed roots/logs와 GH-owned untracked metadata 무변경 |

## Contract review

- CLI alias는 Llama/Qwen 두 fixed alias로 제한된다.
- `load_fixed_model_checkpoint_original`만 import하며 legacy
  `load_fixed_model`은 AST에 없다.
- runtime metadata, floating parameter dtype set, config dtype, model/tokenizer
  revision을 모두 fail-close한다.
- seed reset과 `fresh=True` generation은 같은 process에서 loop count 2로
  고정된다.
- exact comparator는 source/templates/manifest/group size/canonical template
  bytes를 모두 독립 확인한다.
- EasyEdit source manifest는 generation 전후에 검사한다.
- output root는 repository의 `local/results/session02-bf16-context-lock-probe-v1-*`
  direct child만 허용하고 기존 path를 거부한다.
- raw template 출력은 upstream stdout/stderr suppression 및 raw-free summary
  schema로 차단한다.
- script에는 dataset/covariance/direct-z/evaluation import 또는 Slurm submit
  call이 없다.
- sbatch는 1 GPU/8 CPU/65000 MiB/30분과 future authorization token을
  fail-closed 고정한다.

## Tests

Focused `8/8`와 전체 method `70/70`이 `PYTHONWARNINGS=error`로 PASS했다.
Compile, shell syntax, help-only CLI, diff check도 PASS했다. Synthetic tests는
raw context 문자열을 사용하지만 temporary directory 밖에 기록하지 않았고
Git artifact는 만들지 않았다.

## Red-team conclusion

이 구현은 deterministic calibration을 측정할 수 있는 도구일 뿐 BF16 ID,
legacy-FP32 mismatch 원인 또는 method 우월성의 증거가 아니다. 실제 GPU
double-generation이 불일치하면 tolerance나 seed를 바꾸지 않고
`NONDETERMINISTIC_CONTEXT_HOLD`해야 한다. 현재 actual ID는 임의 제안하지
않았으며 `UNOBSERVED`다.
