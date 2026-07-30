# MV-0 Qwen one-case smoke post-run red audit

- 작성일: 2026-07-30
- 역할: 실행/구현에 참여하지 않은 post-run red agent
- 대상 run: `mv0_qwen_smoke_v1`
- 대상 Slurm job: `15503`
- 최종 판정: **PASS — 이 exact one-case smoke의 실행·rollback fidelity에만 유효**
- 다음 단계: **CONDITIONAL ALLOW — 새 3-case paired calibration gate의 작성·감사·통과 후 1회 실행만 허용**
- 현재 문서가 직접 허용하지 않는 것: 기존 helper를 이용한 3-case 제출, Motivation/MV-0 claim, confirmatory 진행

## 1. 검토 경계

이 감사는 다음 네 파일과 제공된 scheduler fact만 사용했다.

1. `local/results/raw/session01_motivation/mv0_qwen_smoke_v1/manifest.json`
2. 같은 경로의 `summary.json`
3. `audits/global/2026-07-30-session01-mv0-qwen-execution-preflight.md`
4. `project/run_scripts/submit_session01_mv0_server1.sh`

`events`, `direct_z`, raw prompt/context, Slurm log, submission marker, 다른
code/report는 읽지 않았다. 따라서 아래 PASS는 허용된 compact evidence의
내부 일관성과 제공된 scheduler fact에 대한 판정이다. 읽지 않은 artifact의
실제 내용, 로그에 오류 문자열이 전혀 없었다는 주장, 고정 파일 byte를 이
감사자가 다시 hash했다는 주장까지 포함하지 않는다.

## 2. 결론 요약

허용된 증거에서 HOLD를 요구하는 모순은 발견하지 못했다.

- scheduler fact는 job `15503`, `COMPLETED`, `ExitCode 0:0`, elapsed
  `00:04:06`, allocation `gpu=1/cpu=8/mem=65000M`이다.
- manifest와 summary 모두 Slurm identity를
  `15503 / odeedit_mv0_qwen_smoke / devbox / under_slurm=true`로 동일하게
  기록한다.
- 실제 manifest SHA-256은
  `2f80784481c3451dcda991ce3b13fcfcbaa6149a581e4d98be33f2db12e3fc06`이며
  summary의 `artifacts.manifest_sha256`과 정확히 같다.
- 분모는 planned/attempted/pass `1/1/1`, failure `0`, abort 후 미실행 `0`으로
  닫혀 있다.
- Qwen provenance, model/tokenizer revision, selection/context ID, 단일 case와
  request hash가 manifest/summary/preflight 사이에서 일치한다.
- 고정 Qwen mom2 다섯 개는 manifest의 fixed hash/size 목록과 summary의
  loaded 목록에서 계층 4–8로 정확히 대응한다.
- projector는 manifest에서 hash/size 검증 전용으로 고정되어 있지만
  summary의 `projector_files_loaded=[]`이므로 deserialize/load되지 않았다는
  계약과 맞는다.
- reported rollback/teacher equivalence 오차는 모두 0이고 scheduler exit도
  정상이다.

이것은 단 한 calibration case의 plumbing/fidelity smoke 성공이다. 모델 편집
품질, calibration 분포, Qwen 일반성, projector 경로의 동작, Motivation claim을
지지하지 않는다.

## 3. Exact envelope 대조

| 항목 | 사전 허용값 | 관측 증거 | 판정 |
| --- | --- | --- | --- |
| run ID | `mv0_qwen_smoke_v1` | path, manifest, summary 동일 | 일치 |
| job ID | 제출 후 생성 | manifest/summary `15503`; scheduler fact `15503` | 일치 |
| job name | `odeedit_mv0_qwen_smoke` | manifest/summary 동일 | 일치 |
| target | `server1` / `devbox` | helper는 `server1` agent를 강제; manifest/summary node `devbox` | 일치 |
| allocation | A6000 1, CPU 8, `65000M` | scheduler `gpu=1/cpu=8/mem=65000M`; manifest는 visible GPU 1, RTX A6000 | 일치 |
| time limit | `06:00:00` | scheduler fact에는 TimeLimit 필드가 없고 elapsed `00:04:06`만 있음 | **미독립확인** |
| model | `qwen2.5-7b-inst` | manifest/summary 동일 | 일치 |
| revision | `a09a35458c702b33eeacc393d103063234e8bc28` | requested/observed model/tokenizer commit 모두 동일 | 일치 |
| dtype/device | float32, one visible CUDA GPU | `torch.float32`, `cuda:0`, visible count 1 | 일치 |
| case/seed | 1 / 17 | summary case 1; context source `fresh-seed-17` | 일치 |
| output boundary | exact run directory | 검토한 manifest/summary가 exact path에 존재 | 일치 |

TimeLimit의 실제 scheduler 값과 one-shot marker/job-id 내용은 주어진 증거
경계에서 확인할 수 없다. 또한 manifest/summary는 helper를 거쳐 제출했다는
암호학적 증명을 담지 않는다. 다만 helper는 exact
`qwen2.5-7b-inst|mv0_qwen_smoke_v1|1` 외 인자를 거부하고, 외부 scheduler
identity와 결과 artifact identity가 모두 같은 job `15503`에 결속되어 있어
현재 결과 무결성을 뒤집는 충돌 증거는 없다. 다음 gate에서는 scheduler
`TimeLimit`도 post-run fact에 포함해야 한다.

## 4. Manifest/summary 결속과 provenance

다음 값이 서로 닫혀 있다.

- `run_id`: `mv0_qwen_smoke_v1`
- `model_alias`: `qwen2.5-7b-inst`
- `slurm.job_id`: `15503`
- `slurm.job_name`: `odeedit_mv0_qwen_smoke`
- `slurm.node`: `devbox`
- `provenance_id`:
  `d247b27edf638b7a9625954e8bf2aeac745b61247230f8bcf733e991a3ed9d16`
- selection manifest ID:
  `186372487e0a79d64a4895dfa5ad142a5e9b1dd3f675c9a3468205147c38c4ce`
- context ID:
  `e0c5f61d874334a2cab26f82fb3594d9ab88c9e5816be390274a0bb5e98c93bd`

manifest의 selected case는 `18447` 하나이고 summary의 유일한 case result도
`18447 / attempted / pass=true`다. selected request hash도 하나다.
`direct_z_artifact_count=1`은 단일 attempted case와 수적으로 맞는다.

manifest SHA는 이 감사에서 직접 재계산하여 summary 값과 일치함을 확인했다.
반면 summary의 `events_sha256`은 events를 읽지 말라는 감사 경계 때문에
재계산하지 않았다. `provenance_id` 역시 세 문서 사이의 exact equality는
확인했지만, 그 생성식을 독립 재실행한 것은 아니다. 따라서 “ID가
교차일치한다”보다 강한 provenance 주장은 하지 않는다.

manifest의 `ode_edit_git`은 commit
`598c5366bf4e355c4496e1b55b5b14043e5322bc`, tracked worktree clean을
기록한다. submit helper는 실제 제출 전에 untracked를 포함한 clean 상태,
`main`, `HEAD == origin/main`을 강제한다. 그러나 이 post-run artifact가 그
검사 결과 자체나 preflight/helper/sbatch digest를 별도 필드로 봉인하지는
않는다. 이는 현재 결과의 모순은 아니지만 다음 gate의 provenance를 더
강하게 만들 수 있는 지점이다.

## 5. 분모와 실패/rollback

summary의 case accounting은 다음과 같다.

| 지표 | 값 |
| --- | ---: |
| planned | 1 |
| attempted | 1 |
| case_count | 1 |
| pass | 1 |
| failure | 0 |
| not run due to abort | 0 |
| abort failure type | `null` |
| run status | `completed` |
| all pass | `true` |

따라서 case-level 누락, 실패 case 제외, abort 뒤 분모 축소는 보이지 않는다.
compact metric은 다음과 같다.

- `all_rollbacks_exact=true`
- final weight relative L2 error 최대값 `0.0`
- teacher logits relative L2 error 최대값 `0.0`
- teacher NLL absolute error 최대값 `0.0`

이는 한 case에서 보고된 rollback과 teacher/native-adapter fidelity가 exact
기준을 만족했다는 증거다. 다만 compact summary에는 각 내부 trace/check의
개별 시행 횟수나 paired-arm별 분모가 없다. 그러므로 이 결과를 “여러
pair에서 모두 성공” 또는 “trace의 모든 가능한 관측점이 검증됨”으로
확장해서는 안 된다.

## 6. 고정 mom2, projector, direct-z

manifest는 다음 Qwen Wikipedia float32 mom2를 fixed hash/size로 기록하고,
summary는 같은 다섯 파일만 loaded로 기록한다.

- `model.layers.4.mlp.down_proj_float32_mom2_100000.npz`
- `model.layers.5.mlp.down_proj_float32_mom2_100000.npz`
- `model.layers.6.mlp.down_proj_float32_mom2_100000.npz`
- `model.layers.7.mlp.down_proj_float32_mom2_100000.npz`
- `model.layers.8.mlp.down_proj_float32_mom2_100000.npz`

누락, 추가 layer, 다른 model 통계 파일은 summary 목록에 없다.
`covariance_policy`는 verified-read-only와 recompute/download 차단을
명시한다. 이 감사자는 파일 bytes를 다시 읽거나 hash하지 않았으므로,
판정은 producer가 남긴 fixed hash/size와 loaded path의 교차일치에 한정한다.

Qwen projector
`examples/null_space_project_Qwen2.5-7B-Instruct.pt`는 manifest fixed 목록에
hash/size가 있으나 정책은 `sha256-and-size-verify-only; never-deserialized`다.
summary의 `projector_files_loaded=[]`은 그 경계를 정확히 유지한다.
따라서 projector를 사용했다거나 projector 계산 경로가 검증됐다는 주장은
금지된다.

direct-z는 manifest상 이 local run에서 computed-once-and-frozen 정책이고
summary count는 1이다. direct-z artifact 자체를 읽지 않았으므로 내용,
hash, case 결속을 독립 검증했다는 주장은 하지 않는다.

## 7. 자원 검토

Scheduler와 summary wall time은 각각 246초와 `240.812937`초로 약
5.187초 차이다. allocation 시작/종료를 포함하는 scheduler elapsed와 내부
runner wall의 경계 차이로 설명 가능한 크기이며 충돌로 보지 않는다.

| 자원 | 관측값 | 해석 |
| --- | ---: | --- |
| GPU peak allocated | 40.801 GiB | 단일 A6000에서 완료 |
| GPU peak reserved | 43.586 GiB | reported total의 91.946% |
| GPU reported total | 약 47.404 GiB | reserved headroom 약 3.818 GiB |
| 내부 host max RSS | 16.514 GiB | `65000M`의 약 26.02% |
| scheduler MaxRSS | 14.747 GiB | `15463312K`; cap보다 충분히 낮음 |

내부 `host_max_rss_kib=17315736`은 scheduler `MaxRSS=15463312K`보다
`1852424 KiB`, 약 11.98% 높다. 서로 다른 process-tree/측정 시점 정의일
가능성이 있으나 허용된 문서에는 정의가 없다. 둘 다 allocation보다 훨씬
낮으므로 현재 run의 HOLD 사유는 아니다. 다음 gate는 두 RSS 계측의 scope를
명시해 수치 불일치를 숨기지 않아야 한다.

반면 GPU reserved 91.946%는 여유가 넉넉하다는 증거가 아니다. 3-case
확장은 case나 paired arm의 동시 실행, 두 번째 model copy, 누적 cache를
허용하면 안 된다. 한 model을 유지한 순차 실행, case별 exact rollback,
누적 reserved/allocated 감시가 새 gate의 필수 조건이다.

## 8. Artifact boundary와 overclaim

검토한 manifest/summary 자체에는 case ID, request hash, aggregate metric,
provenance/resource metadata만 있고 raw edit/eval field나 raw template는
없다. manifest도
`case IDs/hashes/metrics only; no raw edit or eval fields`,
`raw_templates_persisted=false`를 기록하며 summary는
`git_output_written=false`를 기록한다.

그러나 events/direct-z/나머지 output directory를 읽지 않았으므로 “전체
artifact tree에 raw field가 절대 없다”는 독립 검증은 아니다. 현재 PASS는
compact manifest/summary boundary가 준수됐다는 판정이며, 읽지 않은 artifact
내용에 대한 privacy/security 보증으로 인용하면 과대주장이다.

특히 이 한 case PASS로 허용되지 않는 주장은 다음과 같다.

- calibration 20-case 또는 confirmatory 60-case의 성공률
- Qwen 전체 분포나 다른 seed에 대한 일반화
- 편집 efficacy/locality/portability 품질
- AlphaEdit projector의 correctness
- MV-0 전체 또는 Motivation claim의 승인
- 동일 자원에서 3 case를 병렬로 안전하게 수행할 수 있다는 주장

## 9. 다음 3-case paired calibration gate

**Gate 단계 진입은 허용한다. 현재 결과나 helper로 3-case 실행을 직접
허용하지는 않는다.**

근거는 두 가지다.

1. 현재 one-case smoke에는 HOLD를 요구하는 fidelity/rollback/identity
   모순이 없다.
2. 현재 preflight는 case 확장에 새 gate가 필요하다고 명시하고, submit
   helper의 case 문은 Qwen exact triple
   `qwen2.5-7b-inst|mv0_qwen_smoke_v1|1`만 허용한다. `CASES=3`은 현재
   helper에서 즉시 거부된다.

새 gate는 최소한 다음을 사전에 고정해야 한다.

1. 새 run ID, 새 job name, 새 output 및 one-shot marker, exact 세 case ID와
   순서, paired arm의 정의
2. 같은 model/tokenizer revision과 provenance를 유지할지 여부 및 그 exact
   ID; 바뀌면 새 provenance 전체
3. planned case `3`, planned pair `3`, arm별 denominator, dropped/unpaired
   count를 명시하고 하나라도 누락되면 fail-closed
4. case별 direct-z freeze와 case 결속, 다섯 mom2만 read-only load,
   projector load `[]`
5. 각 pair/case 직후 exact rollback; 실패 시 attempted/failure/not-run
   분모를 보존
6. 단일 GPU·단일 model의 순차 실행, 누적 GPU peak 감시, 동시 model copy와
   병렬 pair 금지
7. scheduler JobID/JobName/node/allocation/**TimeLimit**을 post-run evidence에
   남기고 manifest hash를 다시 결속
8. compact artifact firewall 유지와 calibration-only claim boundary

세 case가 selection manifest의 deterministic prefix를 뜻한다면 후보는
`18447`, `3176`, `15669`이지만, 새 gate가 이를 명시적으로 고정하기 전에는
이 감사가 그 선택을 대신 확정하지 않는다.

## 10. 최종 판정

- 현재 `mv0_qwen_smoke_v1` / job `15503`: **PASS**
- PASS 범위: exact one-case execution, reported native/adapter trace
  equivalence와 exact rollback, compact evidence consistency
- 현재 artifact를 근거로 한 3-case 직접 제출: **NOT AUTHORIZED**
- 새 exact 3-case paired calibration gate 작성 및 독립 preflight 후 실행:
  **CONDITIONAL ALLOW**
