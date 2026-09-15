# EP-TW-1 저장 episode FD repair — 준비·제출 인계

Instruction `ODEEDIT-S06-EP-TW1-SAVED-EPISODE-FD-REPAIR-SH4-V1`. **PENDING / WAITING_USER_RESUME**. 기술 GPU PASS, scientific G0 또는 1000요청 완료 보고가 아니다.

## 실제 제출 및 중지

단일 job **47942 / odeedit_ep_tw1_repair_s4**를 held→owner/source/args/memory 검증→release했다.
마지막 관측은 **2026-09-15T04:46:07.004700+00:00 UTC / PENDING / Reason=None / RunTime0**이다.
그 뒤 scheduler/result/log 및 결과파일 존재 여부를 추가 조회하지 않았다.
프로그램은 그대로 두며 agent만 명시 사용자 호출을 기다린다. 별도 후속 submit/callback/heartbeat/자동분석0.

같은 GPU allocation에서 먼저 저장 episode 기술 진단을 수행한다. E/D 각각 direct-weight route와 두 방향 FD가 모두 통과하고
source/model/teacher/scientific-lock identity가 일치해야 다음 독립 process가 fresh W0/coldM0 EP-TW-1 first1000 B100×10을 실행한다.
기술 실패가 나면 shell fail-closed이며 본실험0. 과거 실패 B1을 B2부터 resume하지 않는다.

## 보존한 실패·재사용 범위

기존47884 FAILED1:0/473 allocated GPU초, native100/solve1/commit0을 그대로 보존한다.
옛 E AD0.026770689893859417 대 FD0.25586175077340434/0.22180749523904691의 coarse FAIL은 폐기하지 않는다.
국소성 미확인이 우선 진단 대상이며 AD가 맞다고 확정하거나 EP 편집 효과 실패로 해석하지 않는다.
Saved Vp/A/Z/anchor/radius/K/H 파일 256307027B,
SHA `67b1aaaf765fca753e3399016967f62b2e0cf3f5b2f374874b1eb9d9aae88451`를 새로 fullSHA 검증했다. 기술 과정 native target/solve0.
원 gE/gD·postfit RNG·optimizer/local teacher 실물이 없어 complete continuation checkpoint가 아니다.
새 diagnostic seed2026091503을 봉인하며 old gradient byte equality를 주장하지 않는다.
기존 teacher47592/192문서/98GPU초는 prior fullSHA+현재 stat 및 manifest로 재사용한다. 재생성/전량 중복재해시0.
Native286.5445305초는 기존473초 내부 component라 따로 더하지 않는다.

## 수리 및 검증 수준

|항목|확인 범위|
|---|---|
|FULL_READ|사용자 점검·GH review36 scalar checks·repair envelope/dispatch·v3계약·PROTOCOL·old receipt/source|
|CPU|전체86개 PASS; 독립 red35개는 그 부분집합. 실제 Llama parity 아님|
|Direct 경로|L4 selected weight leaf로 functional_call, custom residual node 우회. gC와 gW Aᵀ 및 bilinear 검산 예정|
|조기 저장|각 완료 E/D gradient·E0/D0·RNG·rows 저장; 매 signed probe C/hash/action/ULP64/rows/receipt를 판정 전에 저장|
|FD|E self+고정독립 방향 뒤 D self+고정독립 방향. 각 h/2^k k0..9 전부, 최대80 signed objective 관측|
|판정|relative .15/absolute1e-7/convergence .15 유지. 분모 max(absAD,1e-7), k>=1 연속2 resolved scale, 첫 valid window|
|해상도|실제 FP32 action·중복/zero·baseline3회 jitter·Taylor를 기록. 8eps loss-scale은 model 오차상한이 아님|
|과학 계약|policy/ledger/native fitter/map/target/teacher 불변. positive quality allowance0, RAW fallback 원규약 유지|
|Actual GPU|Direct/FD E&D/materialization/새 G0 모두 **NOT_OBSERVED**|

한 gradient sweep 내부에서 예외가 나면 미완성 g/rows 전체는 없을 수 있다. 직전 완료 단계와 오류는 보존하며
미완성 값을 완전 gradient로 부르지 않는다. CPU 검사는 모든 neural backward의 독립 증명도 아니다.
새 scientific B1의 Vp/A/gE/gD가 동일해야 저장 기술 증거를 정확 재사용한다. 다르면 새 endpoint에서 같은 bounded 검사만
진행하고 same-state라고 부르지 않는다. 검사 실패/불확정이면 수치 허용오차를 늘리거나 다른 과학 정책을 추가하지 않는다.

## 자원·비용·출력

제출 직전 project active/admitted0 + 신규1 =1≤cap2. 다른 사용자/job변경0.
1GPU/8CPU/60416M/exportNONE/no-requeue, wall12h, GPU-hour hardcap=null.
가용 120,436,858,880B/inodes226,321,194. 신규40GiB reserve+별도20GiB 여유 검사(독점 예약 아님).
기술300–1800초/본실험2–8GPUh는 사전 추정이며 신규 실제 allocated/peak/결과는 아직 미관측이다.
E grid 최대280 microbatch groups, D grid2560 document forwards; 각각 direct backward 및 baseline 반복·I/O/restore는 별도 ledger.
기술과 scientific의 분모·비용은 구분한다. 진단 old native100을 신규1000 성공분모에 더하지 않는다.

Technical `/data/janghj/ODE-edit/local/ep-tw1-c4/20260915-v1/repair-r1/technical-v1`; conditional science `/data/janghj/ODE-edit/local/ep-tw1-c4/20260915-v1/repair-r1/scientific-r2`.
Source `2a0fe8e2033086bd86eb740d9b83eb8a2d5c5972` / tree `cfe6c43024dbc0928877fbc16e9aee138626419e`.
Archive `c18f18e4aa773454bc34f10713090224a4502556f509b11e944dd7a991fef7c2`.
Repair lock `e6501577e55aa509765d454d3c605f12a7d710f1a07cf83bbd18aecc97c2f43c`; science lock `f715685f793508ca35be0cc4ed3a4765e584e40ef3d60a1f578599f9f89e4ff2`.
Resume `/data/janghj/ODE-edit/local/ep-tw1-c4/20260915-v1/repair-r1/resume-manifest.json` / SHA `f8885520d3e1b9570a6409a6a3b0608e628f2556a2c1089f343674ac1643dadf`.

CPU 재현: `PYTHONDONTWRITEBYTECODE=1 /data/janghj/EasyEdit/.venv/bin/python -m unittest discover -s project/run_scripts/bg_tw_reference/ep_tw -t . -p 'test_*.py'`.
제출/conditional 명령 및 경계는 `project/run_scripts/bg_tw_reference/ep_tw/REPAIR.md`, `repair.sbatch`와 sealed submission receipts에 있다.
현재 원실행을 다시 제출하라는 명령이 아니다. 이후 explicit recall에서 exact47942/technical/G0만 한정 확인한다.
Raw/tensor/prompt/teacher/probe/fullstdout local-only, Git에는 코드와 compact provenance만. PNG생성0.
NO_BROADCAST_NOT_REQUIRED. 기존 source/실패/teacher/raw 및 다른 paused task 불변, scientific_promotion=false.
