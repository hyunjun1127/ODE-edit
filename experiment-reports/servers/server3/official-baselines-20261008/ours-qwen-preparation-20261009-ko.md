# Qwen ours 실행 전 준비

상태: **PREPARATION_COMPLETE_AWAITING_USER_RUN_SETTINGS**. 사용자의 최신 답변은 “내가 따로 줄테니깐 준비만 해놔.”이다. CAP075/CF first2000은 CPU 호환성 검사용 임시 프로필이며 실제 실행 설정으로 확정하지 않았다. GPU 모델 실행·Slurm 제출·checkpoint 저장은 0이다.

## 준비 및 실측

- Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28` 다운로드 완료. 4개 weight shard의 Hugging Face LFS SHA 일치, 전체 11파일 15,242,788,168 bytes. 기존 다운로드 검증 receipt를 재사용했다.
- C0 L4–L8 5파일의 전체 SHA와 크기를 현물 검산했다. 모델/C0를 생성하거나 통계를 재계산하지 않았다.
- 서버4의 기존 Qwen context 276 bytes만 개별 복사했다. 원본 유지. 파일 SHA `792d32b503fa04b9180caa5281a8869e42e9bb2f12a322585ebc9a0cfd9f34e6`; 내용 정규화 SHA `6688fe84115305610c00b328df931b0f0f4bc6c8eced70386ef975c58d85a9c3`으로 공식 GPU 계획의 기준과 일치한다. context 원문은 Git에 넣지 않았다.
- 기존 CPU runtime torch 2.9.1+cu128 / transformers 4.44.2에서 공식 ours Adapter/fit/entry import 및 Qwen tokenizer 로드 성공. first2000을 B100×20으로 토큰화하고 각 key prefix identity를 확인했다. 모델 forward 없음.
- 단일 noCP run 출력용 8 GiB 예약은 앞선 serializer 상한 추정이며 실측 사용량이 아니다. 이전 CP 포함 24 GiB 추정 역시 저장 권한이나 실행 설정 확정이 아니다.

## W&B

기존 환경파일의 추가 키가 새 공통 `load_env`에 거부되어 기존 파일은 유지하고 다음 전용 비민감 설정을 작성했다:
`/data/janghj/ODE-edit/local/ours-qwen-readiness-20261009/wandb.env`.

공통 `official.tracking`으로 CPU online smoke를 한 번 수행했다. [run d4d0993f0d9d4f10](https://wandb.ai/wkdguswns2256/layer%20allocation/runs/d4d0993f0d9d4f10): **READY_ONLINE_VERIFIED**, 합성 scalar 3점 전송 및 원격 3점 readback, dropped 0. entity `wkdguswns2256`, project `layer allocation`. local run이므로 실제 job ID 없음. 가짜 job 번호·과학 결과·prompt·tensor·코드를 업로드하지 않았다.

공통 logger를 복제하거나 수정하지 않았다. 향후 Slurm 실행 시 원 helper가 실제 job ID를 name/config에 연결한다. ours 전체 JSON은 local에 두고 strict scalar transport에는 config SHA를 연결한다. `bind_fit(..., wandb_run=...)`의 nested `config.ours` connector와 scalar-only transport를 혼용하지 않는다.

## 검증 및 남은 구분

- 공식 CPU 113 tests PASS; 전용 준비/공통 tracking CPU 39 tests PASS.
- 공식 source 157개 SHA PASS, 외부 task imports 0.
- owner audit이며 별도 독립 reviewer는 사용하지 않았다.
- 이것은 GPU qualification이나 과학 runner 제출 완료가 아니다. 사용자가 실행 설정을 전달하면 해당 설정의 runner/transaction/evaluator/collector 연결, 실제 모델 검증, source freeze 및 cap1 admission을 별도 결속해야 한다. 기존 baseline runner를 ours라고 실행하지 않는다.
- 기존 job/source/raw/CP와 공통 환경을 변경하지 않았다. 후속 자동 제출·반복 monitoring 없음.

[상세 준비 receipt](../../../../audits/servers/server3/official-baselines-20261008/ours-qwen-preparation-20261009.json).
