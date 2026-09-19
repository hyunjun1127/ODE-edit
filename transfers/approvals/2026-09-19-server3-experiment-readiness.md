# SH3 공통 실험 자산 선택 수신 승인

Instruction: ODEEDIT-S06-SH3-EXPERIMENT-READY-ASSETS-20260919-V1.
사용자 2026-09-19 “데이터(항상 사용하던10k 순서), 모델, Covariance, Projector 전부 바로 실험에 사용할 수 있도록” 지시에 따른 실제 준비 승인이다.

수신/전송 sole owner: SH3, session01a0b9c8-d12c-7133-b336-cc1b5fc2a6b3.
Source owner는 GH/SH1/SH2/SH4의 해당 project 입력 자산이며 source는 read-only KEEP.

허용 source roots:
- Server1/2: /mnt/raid5/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/
- Server4: /data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/
- 위 각 ODE-edit/local/의 **완료 run**에 이미 저장된 native context/hparams/source/asset manifest 구성요소.
- Server1/2 /mnt/raid5/janghj/EasyEdit/ 및 Server4 /data/janghj/EasyEdit/ 안의 필요한 native package,
  hparams, examples/null_space_project_Meta-Llama-3-8B-Instruct.pt,
  examples/null_space_project_Qwen2.5-7B-Instruct.pt,
  examples/data/stats/ 의 canonical 두 모델 covariance.
- 등록 서버의 janghj HF hub cache 아래 meta-llama/Meta-Llama-3-8B-Instruct@8afb486c1db24fe5011ec46dfbe5b5dccdb575c2,
  Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28 exact snapshot와 그 consumed blobs.

Destination:
- /data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/
- /data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/{easyedit,assets,models,contexts}/
- S3 기존 HF/model/P/C0의 identity가 맞으면 read-only reuse, 기존 mismatch 파일 overwrite0.

실행 전에 SH3가 bounded discovery 결과로 member별 정확 source→destination/size/SHA/owner/모델·layer
allowlist를 작성한다. rsync --files-from 또는 동등한 explicit file copy로 누락만 수신한다.
이 승인 범위 안의 정확 목록 작성은 재승인 조건이 아니다. 범위 밖이면 요청한다.
Directory 전체 mirror, --delete, credentials/privateinventory, 다른사용자자료, live scientificraw,
기존파일덮어쓰기/삭제/이동0. S3만 destination writer로 두고 다른 SH에게 duplicate push 요구0.
Partial 수신은 신규 task partial경로로 두고 filehash/size 확인후 create-once finalize한다.
모델 checkpoint 입력은 reusable asset이며 edited experiment checkpoint 미저장 정책과 별개다.
대용량 stdout/raw Git0. compact list/receipt는 transfers/verifications/2026-09-19-server3-experiment-readiness/에 보존한다.
