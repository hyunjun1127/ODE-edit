# MV-0 paired c3 v2 job 15512 동시성 실패 분석

- 작성일: 2026-07-30
- 역할: 실행·취소에 참여하지 않은 독립 분석 agent
- 대상: `15512` (`odeedit_mv0_pair_c3v2`)
- 결론: **동시 실행 요건을 충족하지 못한 Slurm step-memory envelope 결함**

## 판정

job 15512는 Llama/Qwen 두 model을 동시에 실행하지 못했다. 첫 child
step이 GPU 1개·CPU 8개와 함께 parent host memory `130000M` 전체를
점유했고, 두 번째 child step은 생성되지 않았다. 따라서 MV-0
fidelity, Motivation signal, ODE-Edit 기대 개선폭에 집계하지 않는다.

31초 시점의 GH `scancel 15512`는 잘못된 envelope가 GPU 2개를 계속
예약하는 것을 막기 위한 running-job/GPU-reservation 보호 triage다.
영향 범위는 job 15512 한 건이며 v2 이력은 보존한다.

## 구분

- proposal에서 온 내용: 이 job은 proposal의 연구 가설을 검증하지
  못했으며 stale proposal이나 ODE integration의 증거가 아니다.
- repo/protocol에서 확인한 사실: parent는 GPU 2, CPU 16,
  `mem=130000M`; v2의 각 `srun`은 GPU 1·CPU 8만 명시하고 step
  memory를 누락했다. `PROTOCOL.md:506-509`는 running-job triage를
  포함한 GH 직접 조작 예외를 허용한다.
- GH 추정: launch order와 Llama manifest만 존재하는 상태를 함께 보면
  `.0`은 Llama였고, 두 번째 `srun`은 memory 부족으로 step 생성 전에
  대기했다. scheduler는 취소 주체의 user ID까지만 증명하므로 Codex
  session identity는 GH 실행 기록에 따른다.
- 사용자 확인 필요: v3 exact one-shot은 기존 승인 범위 안이다. v3
  동시성 gate가 다시 실패하면 자동 재제출하지 않고 보고해야 한다.

## 핵심 증거

| 항목 | 확인값 |
| --- | --- |
| parent | `CANCELLED by 1025`, elapsed `00:00:31` |
| parent allocation | CPU 16, GPU 2, memory `130000M` |
| child `.0` | `FAILED 143:0`, CPU 8, GPU 1, memory `130000M` |
| child `.1` | scheduler record 없음 |
| v2 marker | `mv0_pair_c3_v2.submitted/job-id = 15512` |
| Llama v2 output | directory와 partial `manifest.json`만 존재 |
| Llama result | `metadata.json`, `summary.json` 부재 |
| Qwen v2 output | directory와 manifest 모두 부재 |

raw log와 partial manifest 원문은 보고서에 복사하지 않았다.

## 원인과 최소 수정

로컬 Slurm `26.05.0`의 `srun --help`는 `--mem=MB`를 지원한다.
`srun --mem=65000M --usage` parser check는 return code 0이었고 invalid
size control은 거부됐다. v2 `.0`의 실제 `AllocTRES mem=130000M`은
memory 미지정이 첫 step의 parent-memory 전체 점유로 이어졌음을
확인한다.

v3의 논리적 수정은 두 child 모두에 exact `--mem=65000M`을 추가하는
것뿐이다. 합계는 parent `130000M`이며 model snapshot, case, EasyEdit
adapter, metric, threshold, moments/projector read-only 정책은 바꾸지
않는다.

| identity | v3 값 |
| --- | --- |
| job | `odeedit_mv0_pair_c3v3` |
| Llama run | `mv0_llama_c3_v3` |
| Qwen run | `mv0_qwen_c3_v3` |
| marker | `local/state/slurm-submissions/session01_motivation/mv0_pair_c3_v3.submitted/` |

pair wrapper, runtime/Python allowlist, test, helper, red gate를 같은 v3
identity로 회전한다. 단위 없는 `--mem=65000` 대신 audit과 일치하는
`--mem=65000M`을 사용한다.

## 보존·재실행 조건

1. v1/v2 marker, log, v2 partial Llama manifest를 삭제·덮어쓰기·재사용하지
   않는다.
2. 이 보고서와 v3 preflight를 tracked commit에 포함하고 clean
   `main == origin/main`, CPU/shell/session-boundary/cap check를 통과한다.
3. v3 output/marker와 same-name active job이 없을 때 helper로 한 번만
   제출한다.
4. 제출 직후 `.0`과 `.1`이 동시에 `RUNNING`이고 각 step이 CPU 8,
   GPU 1, `mem=65000M`인지 확인한다.
5. 한 step만 생성되거나 memory가 다시 `130000M`이면 해당 job만
   취소하고 자동 재제출하지 않는다.

v3 성공도 implementation fidelity gate일 뿐이다. model별 결과 분석과
pair post-run red audit 뒤에만 다음 Motivation diagnostic으로 진행한다.
