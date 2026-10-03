# JLZ v11 구현과 실행 경계

SH4는 `ODEEDIT-USER-GH-SH4-JLZ-V11-INCREMENT-2K-20261004-R1`의 MAIN, NOALLOC, matched native MEMIT-H 세 경로를 새 namespace에서 구현했다. 각 경로는 독립 W0/H0에서 first2000을 BS100×20으로 처리한다. 기존 v9 2k source, 실행과 산출물은 변경하지 않는다.

## 수학과 계산

전체 request-mean J로 `.05` early-stop을 판단한다. 이 판단을 위한 평가 뒤 동일 D에서 microbatch backward를 재계산한다. 이는 새로운 후보나 fit이 아니며 추가 physical forward를 별도 계상한다. SUM native gradient와 전체 B 배 allocation gradient를 합하고 q bridge를 한 번 적용한다. 후보 2, 9, terminal의 성분별 diagnostic backward는 방문한 후보에서만 수행하고 optimizer에 중복 가산하지 않는다. terminal에서는 optimizer step이 없다. 임계값 근처에서는 같은 후보의 full reference loss로 stop을 결정한다.

Entry 전체 B의 native mean key로 SPD proxy를 한 번 고정한다. Cholesky 안정식과 같은 A의 native G+E를 비교하며, jitter나 대칭화로 통과시키지 않는다. fit에는 실제 writer, pulse, 과거 replay가 없다. 마지막 평가 D를 고정한 뒤 실제 하층 write가 적용된 모델에서 상층 key를 다시 얻어 increment를 쓴다. 원 native stack(requests).T의 CPU FP32 layout으로 final-key history를 한 번 누적한다.

matched MEMIT-H는 고정 BLUE 원 `apply_memit_seq_to_model`을 호출한다. singleton z, residual/divisor, native stop/Adam/history를 유지한다. 현재 transformers의 Tensor/tuple container만 process-local adapter로 연결하고 원 파일은 변경하지 않는다. 작은 pilot에서 실제 token 및 초기 native NLL, native key, H append와 state 연결을 검산한다.

## 입력과 평가

정본 17파일과 CSV 2000행의 모든 field를 검산했다. main 20 pack과 학습 밖 dev 2 pack, observer 26000개의 token identity를 결속했다. 기존 W0의 package byte 및 batching 동등성은 확립되지 않았으므로 이번 권한에 따라 first2000 W0를 한 번만 관측하여 세 경로에서 정확한 source/runtime/input/state identity로 공유한다. W0 B1-pre는 같은 raw에서 파생한다. 새로운 full10k W0 평가는 없다.

매 batch pre-current와 post-current, W5/10/15/20 allseen을 기록한다. milestone current는 같은 allseen raw에서 추출한다. Collector는 NLL/TF/분모, paired lost/gained, cohort/prefix/active, source/state/RNG join, candidate/update/H 횟수를 모델 없이 검산한다. 누락은 PARTIAL 또는 NOT_MEASURED로 기록한다.

## 자원과 종료

V11 cap1의 직렬 pilot→MAIN→NOALLOC→MEMIT-H와 CPU collector를 upfront 등록한다. 기존 점유 job의 afterany는 자원 의존성이지 v9의 과학 PASS gate가 아니다. 후속 science는 동일 source/config의 pilot READY가 없으면 즉시 기술 실패로 종료한다. 외부 callback이나 새 submit은 없다.

GPU job은 각각 1GPU/8CPU/59392MiB, qualification 최대4시간, 본선 각각 최대168시간이다. 요청 wall은 ETA가 아니다. 입력 pack의 최대 길이는 자원 검산에 사용하며, 새 B100 실측은 MAIN B1에서만 얻는다. Local output/atomic scratch/여유 reserve는40GiB이고 runtime 시작에도 다시 검사한다. 기존 파일을 삭제하지 않는다.

Checkpoint 및 복원등가 payload는 저장하지 않는다. 이전 v9 tensor 저장 예외도 상속하지 않는다. 원 source/raw/실패 receipt는 보존한다. 초기 actual 연결 또는 정식 RESOURCE_PENDING 인계 뒤 agent monitoring/automatic resume을 끄고 sealed runner만 계속한다.
