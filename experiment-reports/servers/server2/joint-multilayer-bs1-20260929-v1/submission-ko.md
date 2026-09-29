# SH2 다층 joint BS1×100 이관 — 제출·초기 인계

Nonce `ODEEDIT-GH-SH4-SH2-JOINT-BS1-MIGRATION-20260929-R1`. 상태 **MONITORING_PAUSED_AWAITING_USER**, 초기 관측 **JOINT_B1_TO_B2_OBSERVED**. 전체 실험 완료 보고가 아니다.

## 전량 제출 및 범위

GPU array `55116_[0–8]%3`, CPU collector `55117` / `afterany:55116`. 원 cap2에서 전량 held 검사/release 후 최신 사용자 "CAP 3으로 해"에 따라 해당 array throttle만3으로 변경했다. 원 source/config/lock은 역사cap2 그대로이며 별도 user-cap3 receipt로 실제 운영cap을 결속한다.

|Job|Parent|Arm|offered edits|
|---|---|---|---:|
|55116_0|B010|JOINT_STEP|100|
|55116_1|B010|NATIVE|100|
|55116_2|B010|JOINT_CUM|100|
|55116_3|B050|JOINT_STEP|100|
|55116_4|B050|NATIVE|100|
|55116_5|B050|JOINT_CUM|100|
|55116_6|B090|JOINT_STEP|100|
|55116_7|B090|NATIVE|100|
|55116_8|B090|JOINT_CUM|100|

원 metadata 선택500의 앞100을 사용한다. fixed10k first100이 아니다. 원272 control/observer·다섯 L4–L8·FP32/eager·TF32 matmul=false/cuDNN=true·원 native L2=10/blue=false·JOINT 수식/예산/수치 guard 불변. 900과학attempts 및 native300targetfits/joint600solve는 계획이고 actual 완료수치가 아니다. 기술replay S4=0, S2 별도replay=0; 초기검사는 실제 과학 경로의 통합 기록이다.

## 초기 상태와 비용

관측시각 `2026-09-29T10:29:45.343834+00:00`. Outcome `ACCEPTED`, proposals `11`, trials `50`, full guards `5`. Method elapsed `277.3839613618329` sec, geometry `79.23173331469297` sec; 중첩 timer이며 합산하지 않는다. 정상 reject는 과학 실패/기술오류와 구분한다. W/M/anchor/dual/RNG/context 연결 receipt는 submission audit에 결속했다. 미래 step/최종성능/저장완료는 미관측.

각1GPU/8CPU/60416MiB/exportNONE/Requeue0, array/task/project cap3. Collector CPU8/24576MiB/GPU0. RTX A6000 48GiB; 사전peak GPU44GiB/host48GiB는 추정이며 실측보장이 아니다. CPU 토큰 검산 최대길이36, 실제 peak는 runner terminal에 기록하도록 되어 있다. Wall7일은 요청 상한이며 ETA/GPUh cap이 아니다. 다른 allocation 포함 admission했고 타job변경0.

## 저장 및 입력 보존

USER 명시 예외: offered25/50/75/100마다 actual full FP32 다섯 weights. 총36snapshot/180tensor/42,278,584,320B payload(39.375GiB)+metadata 계획, atomic save/reload와 모델출력 검증 유지. Exact editor resume NOT_AVAILABLE. 원 CP/raw는 삭제·이동하지 않았다.

S4 정확10job 취소 receipt SHA `2f934836622526bc5555ede5ea54e185323968a8f6f3cba507dd07f5c54f683d`, elapsed0/AllocTRES없음과 원 submission mapping 결속. 소형49member/2,096,027B fullSHA 수신, 대형전송0. S2 parent3CP 현재 file fullSHA 및 CPU W/M/context/RNG 검산, P file fullSHA 일치. Model10member는 S2 prior fullSHA+현재stat 재사용. S41801 broad closure와 actual native21closure를 구분했다.

## 검증 범위·재현

CPU29회귀+5S2routing/cancellation tests PASS. CPU는 actual Llama PASS가 아니다. S4 frozen runtime/solver/evaluator/원 native 파일 불변; S2 별도 entry/prepare/launch를 추가했다. Owner audit 및 독립 fixture/reducer 검사이며 별도 red agent PASS 주장은 없다. 타 task pause 유지. Raw/weights/prompts/teacher/fullstdout Git0. NO_BROADCAST_NOT_REQUIRED: 승인된 소형수신만, 새결과 원격방송0.

- 실행source `2a7762a4368de0dbe540619ceacefd7760d25698` / tree `f32e4129e46284d0b9bd477f9620b01b2eebc607`
- lock SHA `b0d320edfb8adce9f6dedf6fed9b4790cb545a7b9ef63f4cf651d8004556614b`
- config SHA `016c33011d0be2a38f99b4923ba373021317f508661251539749fe5192a08295`
- 출력 `/mnt/raid5/janghj/ODE-edit/local/joint-multilayer-bs1/20260929-v1/attempt-s2-r1/output/`
- CPU: `/mnt/raid5/janghj/ODE-edit/local/joint-multilayer-bs1/20260929-v1/cpu-audit-r1/owner-cpu-audit.json`
- 재현 명령은 `/mnt/raid5/janghj/ODE-edit/local/joint-multilayer-bs1/20260929-v1/attempt-s2-r1/branch.sbatch`, `/mnt/raid5/janghj/ODE-edit/local/joint-multilayer-bs1/20260929-v1/attempt-s2-r1/collector.sbatch`에 봉인. 승인 없이 같은출력으로 재실행하지 않는다.
- 검사: `/mnt/raid5/janghj/EasyEdit/.venv/bin/python -B -m unittest project.run_scripts.joint_multilayer_bs10.test_server2 -v`; 원29검사는 `server2_entry audit` 경로.

이 초기 인계 후 agent polling/terminal 대기/heartbeat/자동recall은 중지한다. 이미등록 runner/collector는 예정대로 진행하며, 상세 결과 검토는 사용자 recall 이후다.
