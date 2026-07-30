# MV-0 Llama smoke 독립 post-run red audit

- 작성일: 2026-07-30
- 역할: 실행과 구현에 참여하지 않은 post-run red agent
- 대상 run: `mv0_llama_smoke_v1`
- scheduler fact: job `15500`, `COMPLETED`, `ExitCode 0:0`,
  elapsed `00:02:50`, `MaxRSS 9734028K`
- allocation fact: GPU 1, CPU 8, host memory `65000M`
- 최종 판정: **PASS — exact Llama 1-case implementation
  fidelity/trace-neutrality smoke에만 한정**
- Qwen 판정: **새 Qwen 전용 audit 작성은 허용. Qwen 실행은 그 새 audit이
  PASS하기 전까지 HOLD**

## 1. 검토 범위와 증거 경계

이 판정은 다음 네 파일만 읽어 수행했다.

1. `local/results/raw/session01_motivation/mv0_llama_smoke_v1/manifest.json`
2. `local/results/raw/session01_motivation/mv0_llama_smoke_v1/summary.json`
3. `audits/global/2026-07-30-session01-mv0-execution-preflight.md`
4. `project/run_scripts/submit_session01_mv0_server1.sh`

`events.jsonl`, direct-z artifact, raw prompt/context, Slurm log, 다른
report/code는 읽지 않았다. 따라서 이 audit은 그 내용에 대한 독립 재분석이나
syscall 수준의 부재 증명을 주장하지 않는다. 제공받은 scheduler fact와 세
artifact SHA-256 fact는 외부 사실로 사용했다.

직접 다시 계산한 digest는 다음과 같고 제공된 fact와 일치한다.

- manifest:
  `9fa3b63752a9f398dc2e97e5b7021218136a399c62ecaf179ef999d1ee94b635`
- summary:
  `3609cf679ad0987083ade08b2f11c9f9a28ceec95c26ededb114d998b9156c61`

events digest
`22a6d989e79081b4d837c4913f10d5081c78d4976ac92414c8c9276b8a7e0635`
는 summary의 `artifacts.events_sha256` 및 제공된 fact가 서로 일치한다.
단, events 파일을 금지 범위에 따라 열거나 직접 hash하지 않았으므로 이
일치는 외부 fact와 compact artifact 사이의 일치이지 events 내용의 독립
검증은 아니다.

## 2. exact one-shot 및 denominator

### 판정: PASS

Preflight가 승인한 범위는
`llama3-8b-inst mv0_llama_smoke_v1 1` 한 건뿐이다. 읽은 submit helper도
인자 수, model alias, run ID, case 수를 각각 exact equality로 제한하고,
exclusive output path, same-name active job 부재, durable one-shot marker를
검사한 뒤 `sbatch`를 한 번만 호출하도록 되어 있다. marker는 제출 실패
시에도 자동 삭제되지 않는다.

결과 측 증거도 exact 범위와 일치한다.

- manifest: `run_id=mv0_llama_smoke_v1`,
  `model_alias=llama3-8b-inst`
- manifest: selected case는 `18447` 하나, selected request hash도 하나
- summary: `planned_case_count=1`
- summary: `attempted_case_count=1`
- summary: `case_count=1`
- summary: `pass_count=1`, `failure_count=0`
- summary: `not_run_due_to_abort_count=0`
- summary: case `18447`은 `status=attempted`, `pass=true`
- summary: `run_status=completed`, `all_pass=true`,
  `abort_failure_type=null`
- scheduler: job `15500`은 `COMPLETED`, `ExitCode 0:0`

따라서 실패 case 제외, attempted-only 분모 사용, abort 뒤 미실행 case
은닉은 없다. 이 run의 정확한 분모는 planned 1 / attempted 1 / passed 1 /
failed 0 / abort로 미실행 0이다.

잔여 증거 약점은 manifest와 summary 자체에 Slurm job ID가 없다는 점이다.
job `15500`과 artifact의 연결은 제공된 scheduler fact, exact run path,
helper의 exclusive one-shot contract에 의존한다. 이번 최소 smoke를
HOLD할 정도의 모순은 아니지만, 후속 Qwen audit에서는 job ID 또는
`SLURM_JOB_ID`를 compact metadata에 명시적으로 결속시키는 편이 더 강하다.

## 3. provenance와 hash chain

### 판정: PASS

두 compact artifact의 식별자는 서로 정확히 이어진다.

- run ID: `mv0_llama_smoke_v1`
- model alias: `llama3-8b-inst`
- provenance ID:
  `273d06af367904aac61931331e6aa2ea09204b0db1f6c7557bca16a9e2c84d2b`
- context ID:
  `3020b3f5cea62e6cfbd173f0c99a4348cecf7e84425bb087720ced7f395482e5`
- selection manifest ID:
  `186372487e0a79d64a4895dfa5ad142a5e9b1dd3f675c9a3468205147c38c4ce`
- selected/result case ID: `18447`

summary가 기록한 manifest digest는 직접 계산한 manifest digest와 같고,
summary가 기록한 events digest는 제공된 events digest fact와 같다.
manifest는 ODE-Edit commit
`471ace2591ef325c7f009333715de9d5dc991c5e`, tracked-worktree clean,
model/tokenizer revision
`8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`의 일치를 기록한다.
runtime은 preflight pin과 같은 Python 3.12.0, torch 2.9.1+cu128, CUDA
12.8, transformers 4.57.1, NumPy 2.2.6, accelerate 1.13.0이고,
float32, visible GPU 1, `use_cache=false`, offline을 기록한다.

이는 compact provenance chain에는 충분하다. 다만 이 audit은 fixed-file
각 항목의 hash를 원본 파일에 대해 재계산하거나 commit/worktree 상태를
현재 Git에서 재검사하지 않았으므로, manifest에 기록된 provenance를
넘어선 독립 source attestation을 주장하지 않는다.

## 4. projector load, covariance recompute, direct-z

### Projector: PASS

manifest의 projector 정책은
`sha256-and-size-verify-only; never-deserialized`이고, projector가
`fixed_files`에 존재하는 것은 hash/size pin 대상이라는 뜻이다. summary의
`projector_files_loaded`는 빈 배열이다. 즉 “projector 파일이 검증 대상에
포함됐다”와 “projector를 load했다”를 혼동할 근거가 없으며, compact runtime
evidence는 no projector load와 일치한다.

### Covariance recompute/download: PASS, contract 수준

manifest 정책은
`verified-read-only; recompute-and-download-blocked`이다. summary에는
Llama layer 4–8의 사전 고정된 `mom2` 다섯 파일만
`covariance_files_loaded`로 기록되어 있다. preflight는 covariance miss,
dataset load, force-recompute, download/recompute 요청을 즉시 실패시키고
`all_pass=false`로 중단하도록 요구한다. 실제 결과는
`completed`, `all_pass=true`, `abort_failure_type=null`이다.

따라서 승인된 fail-closed contract의 관찰 단위에서는 recompute/download
요청이 없었다는 판정이 타당하다. 그러나 summary에는 별도의
`recompute_attempt_count=0` 같은 직접 counter가 없고 events/log/code를
이번 audit에서 읽지 않았으므로, 이를 OS I/O trace 수준의 독립적인
“절대 부재” 증명으로 과장해서는 안 된다.

### Direct-z: PASS, 내용 검증 제외

manifest는 direct-z를 이 local run에서 한 번 계산하고 freeze하는 정책을
기록하며 summary의 `direct_z_artifact_count=1`은 1-case 분모와 일치한다.
direct-z 값이나 생성 trace 자체는 금지 범위에 따라 검사하지 않았다.

## 5. rollback 및 fidelity/neutrality

### 판정: PASS

summary의 compact metric은 다음과 같다.

- `all_rollbacks_exact=true`
- `max_final_weight_relative_l2_error=0.0`
- `max_teacher_logits_relative_l2_error=0.0`
- `max_teacher_nll_abs_error=0.0`

planned/attempted case가 모두 하나이고 그 case가 pass했으므로, 위 값에서
실패 case가 분모 밖으로 빠진 흔적은 없다. Preflight의 최소 gate인 exact
rollback, native-adapter fidelity/trace-neutrality failure 시 abort 조건과
모순되는 결과도 없다.

이 PASS는 compact metric이 정의한 비교에 대한 것이다. 전체 모델의 모든
parameter/state에 대한 별도 bytewise scan이나 metric 구현 자체의
재검증을 했다는 뜻은 아니다.

## 6. artifact boundary

### 판정: PASS, inspected/contract 범위

읽은 manifest와 summary에는 raw edit request, target text, evaluation
field, generated context 원문이 없다. case ID, request hash, provenance/hash,
파일 경로, compact metric, denominator, resource 값만 남아 있다.
manifest는 `raw_templates_persisted=false`와
`artifact_firewall=case IDs/hashes/metrics only; no raw edit or eval fields`,
summary는 `git_output_written=false`를 기록한다.

따라서 읽은 compact artifact 자체는 preflight의 artifact firewall을
지킨다. 또한 `all_pass=true`는 forbidden field/raw payload persistence가
fatal이라는 preflight contract와 일치한다. 다만 events, direct-z, raw
directory 전체, Git tree를 의도적으로 검사하지 않았으므로 이 판정을
“모든 미검사 파일에 민감정보가 절대 없다”는 주장으로 확장하지 않는다.

## 7. resource envelope

### 판정: PASS

- allocation은 GPU 1 / CPU 8 / memory `65000M`으로 preflight의 exact
  envelope와 일치한다.
- scheduler elapsed는 170초이고 summary wall time은
  165.68871339131147초다. 약 4.31초 차이는 allocation/job wrapper와
  in-process 측정 범위가 다를 수 있는 크기이며 6시간 limit보다 충분히
  작다.
- summary의 GPU peak allocated는 40,711,586,304 bytes, peak reserved는
  42,689,626,112 bytes, GPU total은 50,899,386,368 bytes다. 각각 total의
  약 80.0%, 83.9%이며 peak reserved 기준 약 8.21 GB의 장치 여유가 남았다.
- scheduler `MaxRSS`는 `9734028K`, summary
  `host_max_rss_kib`는 `11195260`이다. 두 수치는 서로 같지 않으므로 같은
  계측값처럼 합치면 안 된다. step/process/집계 범위 차이는 이번 증거만으로
  확정할 수 없다. 다만 둘 다 `65000M` allocation보다 크게 낮아 cap 위반이나
  memory pressure의 증거는 아니다.
- scheduler 종료 상태와 exit code는 정상이다.

GPU 여유는 Llama 1-case smoke가 성공했다는 사실만 지지한다. 이것을 Qwen,
case 확장, 동시 job의 resource 보장으로 전용하면 안 된다.

## 8. 공격적 overclaim 제한

이번 결과로 허용되는 주장은 다음 하나다.

> 고정된 Llama-3-8B-Instruct revision, seed 17, CounterFact case 18447,
> 고정 context/provenance 아래의 1-case MV-0 smoke가 compact
> fidelity/trace-neutrality, exact rollback, denominator, fail-closed
> artifact contract를 통과했다.

다음 주장은 허용되지 않는다.

- 편집 효과, 일반화, locality 또는 downstream 품질이 입증됐다는 주장
- 여러 case/seed에 대한 통계적 안정성이나 재현성이 입증됐다는 주장
- Qwen에서도 같은 결과나 resource profile이 보장된다는 주장
- projector 또는 covariance I/O의 부재를 syscall 수준으로 독립 증명했다는
  주장
- raw prompt/context/events/direct-z의 내용까지 이 audit이 검증했다는 주장
- 이 PASS가 retry, marker 삭제, 같은 run ID 재사용, 3-case 확장, MV-1을
  허용한다는 주장

## 9. 최종 gate

Preflight의 post-run 필수 조건인 정상 process 종료, `all_pass=true`,
1-case denominator 보존, exact rollback, compact artifact firewall은
제공·검사된 증거에서 모두 충족되고 상호 모순이 없다. 위의 증거 한계는
주장 범위를 좁히는 사유이지 이 최소 smoke를 HOLD할 실패 증거는 아니다.

따라서 `mv0_llama_smoke_v1`은 **PASS**다.

Llama PASS로 인해 **Qwen용 새 audit을 작성하고 심사하는 단계는
허용**한다. 그러나 현재 preflight와 helper는 Qwen을 명시적으로 허용하지
않으므로 **Qwen job 제출 자체는 HOLD**다. Qwen 실행에는 최소한 다음이
고정된 별도 PASS audit이 필요하다.

- Qwen 전용 model/tokenizer revision 및 provenance
- Qwen moments/projector의 hash/size와 projector verify-only 정책
- 새 run ID, 새 job name, 새 durable one-shot marker
- Qwen 전용 allocation/cap 및 fail-closed resource preflight
- exact 1-case denominator와 동일한 abort/artifact/rollback 조건
- scheduler job ID와 compact artifact의 명시적 결속

현재 helper를 수정 우회하거나 Llama run ID/marker를 재사용해 Qwen을
제출해서는 안 된다. Qwen 새 audit이 PASS하기 전에는 3-case 확장과 MV-1도
계속 HOLD다.
