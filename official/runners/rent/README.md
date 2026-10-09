# rent 실행 연결

`rent`는 연구실 공용 A100 서버(Kubernetes, 계정 `nlp-lab`, 학생 폴더 `janghj`)다.
Slurm이 없으므로 job 번호는 **제출 시각 KST `DDHHMM`**(예: `092005` = 9일 20:05)으로 정한다.
접속값과 인증정보는 이 파일에 기록하지 않는다.

## 제출

login 서버(Python 3.8)에서 repo 루트 기준으로 실행한다.

```bash
python3 -m official.runners.rent.submit <template.yaml> [--dry-run]
```

템플릿은 `{{RENT_JOB_ID}}`를 Job 이름, label `odeedit-job-id`, env `ODEEDIT_RENT_JOB_ID`
세 곳에 쓴다. `submit.py`는 번호를 채운 manifest를 repo 밖 `~/janghj/jobs/odeedit/<Job 이름>.yaml`로
남기고 다음을 확인한 뒤 서버 dry-run을 거쳐 제출한다.

- 같은 번호의 janghj Job이 이미 있으면 거절한다(같은 분에 두 번 제출 → 다음 분에 다시 제출).
- 끝나지 않은 janghj Job의 GPU 합계가 2장을 넘으면 거절한다.

`official.tracking`은 Pod 안에서 `ODEEDIT_RENT_JOB_ID`를 읽어
`execution_backend=kubernetes`, `job_id=DDHHMM`으로 기록하고 W&B `run.name` 끝에
`job<DDHHMM>`을 붙인다. env 없이 실행하면 `local`이다.

## W&B

- 비밀 아닌 설정: `servers/local/wandb.env`(git ignore). `ODEEDIT_WANDB_PYTHON`은
  `/workspace/envs/wandb-sdk-0.30.0/bin/python`.
- 인증: 사용자가 `/workspace/.netrc`(login 서버의 `~/janghj/.netrc`)에 직접 저장한다.
  공용 계정이므로 이 서버 전용 API key를 쓴다.
- 확인: `python3 -m official.runners.rent.submit official/runners/rent/wandb_smoke.job.yaml`
  → `logs/wandb-setup/smoke-job<DDHHMM>/result.json`의 `READY_ONLINE_VERIFIED`.

## official baseline runner

세 모델(llama3·qwen25·gptj) × 6개 방법의 공통 runner다. 방법 구현·hparams·평가·checkpoint·W&B는
공통 `official` 모듈만 쓰고, server2 GPT-J runner의 흐름을 모델 일반화했다.
FLU/CON은 `DEFERRED_CHECKPOINT_EVALUATION`으로 미루며 CF chain은 W20 checkpoint를 보존한다.

| 파일 | 역할 |
| --- | --- |
| `freeze.py` | login 서버. 커밋의 `official/`을 `~/janghj/checkpoints/odeedit-official/<commit12>/source`에 고정 복사, `on_main` 기록 |
| `prepare.py` | CPU Pod. 모델·C0·P·tokenizer·stream을 전부 해시해 `assets.lock.json`·hparams lock과 대조, `manifest-<model>.json` 작성 |
| `native.py` | 모델 일반화 native 엔진. C0 캐시 miss 시 재계산 대신 실패 |
| `run.py` | `w0`, `qualification`, `chain`, `smoke` 모드 |

실행 단위는 matrix `run_id`(예: `llama3-cf-memit`, `qwen25-cf-alphaedit_blue-l2-10`)다.
Qwen BLUE 본 행은 L2 격자 선택이 필요하므로 격자 행으로 실행한다.

```bash
python3 -m official.runners.rent.freeze                      # login 서버, commit12 확인
python3 -m official.runners.rent.submit official/runners/rent/prepare.job.yaml \
    --set COMMIT12=<commit12> --set MODELS=llama3,qwen25,gptj
python3 -m official.runners.rent.submit official/runners/rent/run.job.yaml \
    --set COMMIT12=<commit12> --set MODE=qualification --set RUN_IDS=llama3-cf-memit,llama3-cf-ft \
    --set SLUG=q-llama3 --set DEADLINE=14400
```

- `qualification`: CF B1–B3 연속 실행과, 실제 저장한 B2 checkpoint에서 복원해 다시 실행한 B3의
  가중치·native history·context·RNG·평가 결과가 모두 같아야 `PASS_ACTUAL_QUALIFICATION`이다.
- `w0`: 모델·데이터셋별 1회. chain은 `w0/<model>-<dataset>/READY.json`을 요구한다.
- `chain`·`smoke`는 `on_main` 커밋에서만 실행된다. `smoke`는 zsRE 1 batch이다.
- 결과: `checkpoints/odeedit-official/<commit12>/runs/<model>/<run_id>/<mode>-job<DDHHMM>/`.
  checkpoint는 최신 1개를 유지하고 이전 것은 runner가 지운다.
