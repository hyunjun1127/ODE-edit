# Server2 GPT-J cold W0 — 제출 인계

Nonce: `USER-GH-SH1-SH2-BASEMODEL-W0-20261007-R1-SERVER2`

## 현재 상태

**SUBMISSION_HANDOFF — 실제 평가 완료 아님.** GPU **60147**, CPU collector **60148**를 등록하고 owner/fullargv/source/config/resources/dependency/batch-script held 검사 후 release했다. Collector는 `afterany:60147`, 다른 job에 대한 dependency는 없다. Release 직후 단1회 snapshot은 두 job 모두 PENDING, reason=None이었다. 이후 scheduler/result를 polling하지 않았다.

W&B는 sealed runner의 모델 로드 전 online startup 및 마지막 bounded readback으로 확인한다. 현재 actual online/run URL/평가 결과는 **NOT_OBSERVED**다. 정책 ACK·CPU 검사는 GPU/remote PASS가 아니다.

## 범위와 source

- 모델: EleutherAI/gpt-j-6B, revision `47e169305d2e8376be1d31e765533382721b2cc1`. 기존 lowercase cache를 읽기전용 사용. 24,207,819,307B 원 weight full SHA `0e183edc2025ecfdba4429ba43c960224103b3c3dc26616503cdc2158a3d6c93`를 현물 검산했다. 다운로드/복제0.
- Cold W0의 fixed first2000 actual 평가 1회, R2000/P4000/N20000=26000 prompt pairs/52000 candidates. 기존 raw로 대체하지 않는다. Fit/edit/solve/H/C0/P/CP는 0.
- 원 GPT-J PRICE scorer `jlz_price_gptj/scores.py`와 native `CounterFactAdapter.evaluation_ids/panels`, 원 reducer/mapping을 읽기전용 재사용한다. GPT-J transformer 최종 norm 후 lm_head, physical readout27, FP32/eager/no-grad/eval/TF32off/autocastoff/use_cachefalse. 수동 left padding 및 implicit positions, 기존 physical microbatch2를 유지한다.
- 공식 prefix loader로 dataset/order와 first2000 token manifest를 CPU 봉인했다. Ordered IDs SHA `0b912d11659eb087ee71a391b7bea1e02ecc9d999a48254eb8559434965640f4`, observer manifest SHA `daddfe1a9c7af63425db251649861be8f0752b0bb79af8682583daed47a50331`. Generated training contexts는 이 RPN 평가에 사용하지 않는다.
- W&B SH1 helper source `930e46531f38d5fa97a6d860275db73978c7daf4`를 archive에 결속했다. model/family=gptj, writer=none, role=scientific, arm=W0_BASE_MODEL, 실제 job ID 및 이름을 기록한다. W0_first2000 9키/RPN와 harmonic, edits/pre_state/post_state=0만 기록하며 fictitious fit/current/milestone은 없다. N desired=true, pct/nats 및 TF/preference 구분.
- 모델의 전체 named parameter/buffer pointer/version/shape/dtype guard와 finite/count/token identity를 actual 평가 안에서 확인한다. 새로운 비교 GPU forward나 numerical pilot을 추가하지 않는다.

- 실행 source: `771b2c048abf148438c11d843b48144cd99fe944`
- Tree: `4da966cc8a63232321079d831b3be176372a5176`
- Config SHA: `4049323daee95278bf5f209008fabdc0298535f8d152a843fad86b763db74b82`
- Lock SHA: `a8676900ea962572f3c47f53815fd21a722d8a47be520c0e8ba1ac71907a5185`
- Held inspection SHA: `055fa3f1cf900d060fa19cfe86789be6cab38e0295691feeb8578fa73b68e93b`

Local root: `/mnt/raid5/janghj/ODE-edit/local/base-model-gptj-w0/attempt-r3/`. 실제 실행은 이 경로의 immutable source.tar/source, config.json, execution.lock.json 및 main.sh/collector.sh를 사용한다. Source publication과 향후 report publication을 혼동하지 않는다.

## CPU 검산·자원 및 보존한 실패

Owner 최소 CPU 6 tests PASS: mapping/axis/count, N desired=true 및 tie failure, job array0, no-edit/no-save, launcher. 별도 independent reviewer=0, GPU 검사 PASS 주장은 없다. Collector는 새 raw chunks를 읽고 identity/count/strict/token numerator·denominator/summary/hash를 재검산하며 부분·실패도 별도로 보고한다.

실제 A6000 49140MiB VRAM inventory를 확인했다. 요청은 GPU1/CPU6/host59392MiB/4h, collector GPU0/CPU6/24576MiB/4h, exportNONE/Requeue0다. 4h는 계획 요청치이지 ETA가 아니다. Project cap2에서 admission 당시 기존 own GPU1(FE)+신규1을 확인했다. 원 FE 및 다른 job 변경0.

원 60416MiB 요청은 scheduler의 실제 58G 제한으로 거부됐다(jobID0/할당0). Source1917fe45와 attempt-r1/config/lock/거부receipt를 보존했다. 메모리만59392로 낮추고 CPU6·방법·평가·microbatch는 유지했다. 이후 resource-repair-r2 config를 JSON 중계하는 과정에서 nanosecond 정수의 IEEE754 변환 정밀도가 손실되어 제출 전 stat 검사가 차단했다(jobID0/할당0). 실제 자산은 불변이다. 원 exact 정수 텍스트를 보존한 resource-repair-r3 config를 다시 대조해 통과했고 이 config로 attempt-r3를 제출했다. 과학 실패나 model 변경이 아니다.

초기 wildcard cap 점검은 모든 사용자 GPU5개를 세어 거부했으므로 project identity 근거로 쓰지 않았다. 실제 own queue와 명시 project patterns를 결속한 cap 점검은1+1<=2 PASS다. Generic memory audit는 역사 S4 sbatch6개와 Python controller의 #SBATCH 형식 비적용 문제를 표시했다. 이를 PASS로 바꾸지 않았으며 실제 memory-request checker 및 held argv의 --mem/CPU/GPU/time/dependency를 직접 검산했다. 공유 helper·역사 source 수정0.

## 명령·산출물·인계

기존 제출 명령(중복 실행 금지):

```text
/mnt/raid5/janghj/EasyEdit/.venv/bin/python -m project.run_scripts.base_model_eval.gptj_server2_control submit --config /mnt/raid5/janghj/ODE-edit/local/base-model-gptj-w0/resource-repair-r3/config.json --attempt-name attempt-r3 --cpu-receipt /mnt/raid5/janghj/ODE-edit/local/base-model-gptj-w0/resource-repair-r2/cpu-tests.json
```

Runner는 raw/chunk-*.json, result.json, runtime.json, terminal.json, tracking/identity.json 및 transport receipt를 local에 기록한다. Collector는 collection.json/report-ko.md/collector-terminal.json을 생성한다. NoCP/exact_resume=NOT_AVAILABLE. 실패 시 새 attempt 자동제출0.

현재 공개 보고에는 미관측 점수를 채우지 않았다. Sealed runner/collector는 자연 진행하며 agent는 bounded 제출 인계 후 STOP. 상세 완료리뷰는 사용자 recall 때 수행한다. Root dirty·기존 모델/source/raw/job 모두 KEEP, raw/token/model/fullstdout Git0, NO_BROADCAST_NOT_REQUIRED.
