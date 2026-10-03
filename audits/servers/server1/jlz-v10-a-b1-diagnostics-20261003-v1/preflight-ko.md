# SH1 단일 B1 진단 사전 검토

권한 nonce: `ODEEDIT-USER-GH-SH1-JLZ-V10-A-B1-DIAGNOSTICS-20261003-R1`.
정본 authority는 `590c73d78d9f15cd5b7c7a9181e0e17723b616fb`다.

## 읽기·경계

정본 5파일 전체, 동결 31-source closure 및 공용 prompts, 배경 보고 3개를 읽고
현재 local 파일과 frozen Git object SHA/size를 결속했다. 기록은 local
`preparation-r2/full-read.json`이다. 배경 원인 주장과 첨부 수정안은 가설이며
이번 실행의 결론·방법·선택 규칙으로 사용하지 않는다.

App root CWD는 `/mnt/raid5/janghj/ODE-edit`, SH1 session은
`01a04939-f93a-7b50-bca0-65438eab2062`다. registry의 29e4 경로는 역사다.
전용 non-main worktree에서 구현했고 공유 root의 929 dirty entry는 보존했다.
동일 nonce의 기존 SH1 receipt 및 exact-prefix job은 착수 때 발견되지 않았다.

## source와 수학

동결 `c2d5fb107a0435491d8b4705b43b75f6177bbb5c`의 실제 과학 함수를 재사용한다.
기존 `optimize.py`에는 기본값 None인 observer callback 세 위치만 추가했다.
원 25후보/24갱신, c2/9/25 성분 경로, 전체-B adjoint, loss, q bridge,
Adam, terminal25, exact materialized commit은 변경하지 않았다.

c13/17/21 추가 성분 측정은 기존 같은 graph의 별도 retained adjoint다.
이를 optimizer gradient로 대체하거나 합산하지 않는다. callback 전후 q,
q.grad, Adam state, RNG와 guard를 비교한다. CPU 원 동결 optimizer 대
관측 optimizer 전 후보 loss/gradient/layer 기록 및 마지막 weight가 exact였다.

Main의 기존 collector/test_operations에는 c2d5fb10 이후 CPU review 수정이 있다.
처음 준비 시 모든 main 파일이 frozen과 같다고 가정한 검사가 이를 감지하여
CPU 준비만 중단했다. 해당 파일을 되돌리지 않았다. 새 freeze는 31개를
frozen Git object에서 가져오고, 자기 optimize callback과 새 namespace만 추가한다.
기존 collector는 호출하지 않는다. 이것은 GPU/과학 실패가 아니다.

## 입력·runtime·자원

Fixed10k 검증 및 first100 packing identity는
`09c2ea06e490b6052cac81f6743bdbd51d3912dddca321a8e94791efd23e1dd2`다.
기존 local 모델/tokenizer/C0/context를 새 SHA/size/stat으로 결속했다. 전송·다운로드0.
S1 runtime은 Torch 2.9.1 계열/Transformers 4.57.1이며 원 S3 4.44.2/H200과
구분한다. actual CUDA version/device는 job runtime.json에 다시 기록한다.
공유 environment는 변경하지 않았다.

Fit/observer MB1을 사용하고 whole-B100 solve와 원 reduction은 유지한다.
실제 GPU qualification은 같은 고정 candidate에서 원 경로와 추가 관측 경로의
forward/gradient 및 RAM snapshot/restore를 비교한다. CPU 검사를 GPU PASS로
표기하지 않는다. ALL/NONE actual 평가 parity는 fit 뒤 같은 subset에서 확인한다.

GPU base estimate는 model+5 FP64 factor+entry/effective selected weight를 합쳐
42,690,691,072 bytes다. 전체 peak 44–48 GiB는 추정이며 A6000 actual PASS가 아니다.
RAM snapshot 5개는 5,872,025,600 bytes 추정, host peak 계획65GiB/요청96GiB다.
실제 OOM이면 typed capacity block으로 보존하고 자동 추가 fit을 하지 않는다.
6h wall은 ETA가 아니다. 신규1GPU/taskcap1, projectcap2 또는 더 엄격한 실제 cap을
제출 직전 active+pending을 포함해 확인한다. 기존 타 job 변경0.

## 관측·저장·검산

D2는 같은 relation의 기존 fixed10k exact template 및 native subject_last로
결과 관측 전에 식별했다. N1000 중874 식별,126 제외다. 모델 score로 고르지 않았다.
네 mask는 같은874문항이고 D1 N은1000 유지한다. mask는 valid token 집합의
subject/non-subject 분할이며 padding은 제외, target prefix는 non-subject에 포함한다.
동일 Wentry/Weff F.linear 결과를 선택하며 writer 재solve0이다.

W/H/R/P/optimizer 및 resume-equivalent disk 저장0. 다섯 snapshot은 CPU RAM only다.
원 source/raw/실패 경로는 보존하며 old task STOP은 그대로다.
원 5-batch/Q1 runner는 실행하지 않는다. 과학 fit1, B2/armB/추가 fit0.

원 production CPU suite와 새 callback/mask/rollback/독립 reducer/collector/held 등록
회귀 28개가 통과했다. 최종 실행 source 검사는 local `cpu-gate-r3/receipt.json`에
정확 source SHA가 있으며 compact 사본은 같은 audit의 `cpu-regression.json`이다.
독립 reviewer agent는 사용하지 않았다. owner source 검토와 별도 구현 CPU reducer다.
사실 수치와 산술 차이만 보고하고 과학 원인 판정은 GH에 남긴다.

Raw는 same-host local에 보존하여 GH가 접근 가능하므로
`NO_BROADCAST_NOT_REQUIRED`다. Git에는 소스·검사·작은 기록만 게시한다.
