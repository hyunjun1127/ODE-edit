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
