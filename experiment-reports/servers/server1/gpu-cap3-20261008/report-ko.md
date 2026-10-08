# 서버1 project GPU cap3 — resource-only override

KST 2026-10-08. 최신 직접 사용자 요청 `USER-DIRECT-SERVER1-GPU-CAP3-20261008-R1`으로 **server1 현재·향후 own-project GPU 동시 상한3**을 적용했다. 이전 cap2의 resource admission 제한을 supersede하며 job당 GPU는1로 유지한다. 서버2 새 FLU/CON GPT-J 여섯 baseline 재등록 지시는 공식 전달과 full owner ACK를 확인했으며, 실제 새 job 등록은 아직 미관측이다. 서버2 cap2는 유지한다.

## 과학 identity는 보존

기존 runtime source `adb244e6f9c86b54f73bd6d8fb833b338f470ded`, sealed config·archive/lock·W&B run UUID/config·원 제출 identity는 immutable이다. 옛 cap2/DAG가 적힌 frozen 과학 자료를 cap3로 소급 덮어쓰지 않는다. 추가 resource-control receipt로 새 권한과 실제 scheduler 조정을 결속한다. native editing/hparams/solver/history/precision, BS100×20, first2000, W20-only FLU/CON·기존 R/P/N 일정·eval seed·reference/profile는 변경0이다. job당 GPU1·기존 CPU/메모리/시간 ceiling·NoCP·raw local KEEP·privacy는 그대로다. cap3는 GPU3/model job 요청이나 추가 fit/pilot 승인, GPU 가용/성능/완료 증거가 아니다.

## 대상과 조정 경계

조정 전 root snapshot은 MEMIT `61519`와 stock AlphaEdit `61520` RUNNING, AlphaEdit-BLUE `61521` 미할당 PENDING/`afterany:61519`, own GPU0 collector `61522`가 세 baseline `61519:61520:61521`을 기다리는 상태였다. 문서 worker가 새 Slurm 조회를 수행한 값은 아니다.

root는 fresh owner/source/script/full argv·WorkDir·state·GPU allocation·own-project 전체 admission/DAG·cap/QoS/physical 제한을 대조한 뒤 **미할당 PENDING인 정확한 BLUE61521의 resource-only `afterany:61519`**만 제거했다. 변경 즉시 receipt는 PENDING/Dependency `(null)`을 확인했고, root의 후속 bounded queue snapshot은 `61519/61520/61521` 전부 RUNNING, `61522` PENDING/own-three dependency다. RUNNING61519/61520·source/argv/input/native trajectory와 own collector의 세-target dependency, OURS/PRICE/타 owner/무관한 job은 변경하지 않았다. 향후에도 stricter QoS/물리/메모리 제한이 cap3보다 낮으면 그것을 우선하며 scheduler resource PENDING은 정상 인계 상태다.

## 실제 resource 결과와 서버2 인계 상태

root는 WT와 실제 App의 ignored `servers/local/gpu-caps.tsv`에서 server1 row만3으로 바꾸고 node/memory/다른 row는 보존했다. PENDING의 명시적 `(null)` allocation 표현을 좁게 처리하는 guard를 반영한 resource-only script와 CPU3 tests PASS를 확인했다. missing/모호한 GPU request를0으로 해석하거나 RUNNING allocation guard를 없앤 것이 아니다. [actual resource receipt](/mnt/raid5/janghj/ODE-edit/local/gpt2xl-baselines-fluency-consistency-2k/20261007-v1/native-repo-repair-r1/cap3/receipt.json), SHA `c86a18aaff94761b4ab5fee47217a102f3de5b48386726cdfce8356c2186316e`는 projected width3, QoS `lab_gpu_s1` user GPU ceiling4, partition UP, reserve/inode 검산과 source/config·RUNNING·타 job 변경0을 결속한다. job당 GPU1·source/config/archive/W&B identity와 collector는 그대로다. 이 CPU/resource proof는 모델/GPU 과학 완료·속도·ETA의 증거가 아니다.

향후 own native 제출은 `repo_native_submit --user-cap3-override`가 pinned `repo_native_admission` resolver에 최신 승인 receipt를 결속해 세-root width3을 허용한다. flag 없이는 원 cap2 경로가 그대로다. 관련 cap3/admission/original-control CPU30/30 PASS를 root가 확인했으며 GPU/Slurm tests는 아니다. 일반 legacy checker는 historical canonical2와 기존 job patterns를 계속 읽으므로 그 출력을 cap3 PASS/현재 DAG proof로 표현하지 않는다. 글로벌 정책은 변경하지 않았고 현재·향후 own-native3 범위만 explicit user override로 결속한다. frozen runtime `adb244e6…` bytes는 불변이다.

공식 `REGISTERED_SSH_UNIX_WEBSOCKET`의 `turn/start` directive accepted 후 exact same-turn `thread/resume` [owner readback](/mnt/raid5/janghj/ODE-edit/local/gpt2xl-baselines-fluency-consistency-2k/20261007-v1/native-repo-repair-r1/server2-reregister-owner-readback.json), SHA `8d16cff1f84ccb39bcf7dba35a0ab359dc24504cfe8ab80d4fb721dab966b4fd`에서 full `OWNER_ACK nonce=USER-DIRECT-SH2-GPTJ-NATIVE-FLUCON-REREGISTER-20261008-R1 server=server2 accepted=true`를 확인했다. turn은 `01a11b97-f23f-7411-ba9b-6010c1640d7b`, 상태는 inProgress다. owner는 RUNNING MEMIT61428 보존, 동일 native-profile 신규 등록 없음, 공통 source를 그대로 재사용하는 private adapter 결속 중이라고 보고했다. cap2에서 여섯 baseline+CPU collector를 fresh 등록·검사·release하고 renewed MEMIT는 기존 RUNNING과 중복하지 않게 직렬화하는 방향이다. actual 새 sourcefreeze/job IDs/held inspection/release는 아직 미관측이며 ACK·등록 의향을 재등록 완료로 표현하지 않는다. 기존 frozen running source를 덮어쓰거나 기존 job을 이 지시로 취소하지 않는다.

dirty root 및 타 agent 변경은 보존했다. 문서 worker는 root local receipt를 읽고 이 보고서와 compact audit만 작성했으며 Slurm/model/GPU/network/config/commit/push 작업은0이다. root의 실제 resource action·공식 relay와 문서 worker의 기록을 구분한다. 서버2 실제 재등록/source 결과는 owner 후속 receipt이며 반복 monitor/automatic retry는 만들지 않는다.
