# GPT2-XL PRUNE / RECT 준비·등록 사실보고

실제 GPU2개 및 CPU collector 등록·held 검사·release 완료입니다. 현재 dependency PENDING이며 결과/성능/GPU parity를 주장하지 않습니다.

두 arm은 BLUE `311b076a92e4ed0f14f5c8b4909732da781bc5f7`의 ordinary MEMIT GPT2 JSON(layers13–17, blue=false, LR.5, 20평가/19갱신, 20000C0)을 사용합니다. 각각 독립 cold W0에서 동일 first2000을 BS100×20으로 native apply하며 H/P/추가 fit/pilot이 없습니다. 원 CAKE/BLUE/EasyEdit 및 기존 job은 보존합니다.

PRUNE은 20회 dense native fit 후 단 한 번 spectrum 변환합니다. 명시 승인된 `PRUNE_TERMINAL_BASE_FIX`만 적용하여 `currentW + compressedD` 대신 `saved_coldW0 + compressedD`를 씁니다. 원 spectrum/cutoff/dtype/operator는 그대로이며 upstream bitwise 동일이 아닙니다. selected cold W0 및 SVD workspace는 RAM만 사용합니다. W5/10/15는 dense, W20 post만 repaired PRUNE입니다.

RECT는 native dense provisional lower planning·restore 후 상대 delta/(W+1e-8)의 40% kth `>=` mask를 public apply합니다. 동률 때문에 실제 support가 40%보다 클 수 있어 support/tie 분모를 기록합니다.

최종 owner CPU suite는 71개 실행, 69 PASS/2 SDK 의존 skip/0 FAIL입니다. native parser/private imports, 실제 조립·직렬화·콜백·transaction·noCP·exact input, PRUNE zero/no-compression controls, terminal cold/dense/final hash, RECT 표현식, scalar mapping/axis/job identity/cap DAG를 검사했습니다. 독립 source/CPU reviewer는 별도로 34개 모두 PASS했고 terminal 결속 수리와 hash convention을 재검산하여 남은 source/contract blocker를 발견하지 않았습니다. GPU/model/native fit 및 실제 online 기록 검증은 하지 않았습니다. 초기 CLI json NameError 및 이전 반례·CPU receipt는 local에 보존합니다.

새 admission은 최신 사용자 server1 합산 cap2를 적용하고 W0 추가GPU 예외를 사용하지 않습니다. 현재 선행 CAKE60739·ALPHAEDIT_BLUE60740 source/job을 그대로 두고 정확 resource frontier afterany 뒤에 등록합니다. 계획 자원은 GPU1/CPU8/65536MiB/48h, collector GPU0/CPU8/24576MiB/4h이며 wall은 ETA가 아닙니다. 실제 held owner/source/fullargv/resources/dependency 검사 뒤 release합니다.

NoCP, exact_resume=NOT_AVAILABLE, raw local KEEP, scalar/compact source만 Git·W&B, 큰 전송 없이 NO_BROADCAST_NOT_REQUIRED입니다. 반복 monitor/heartbeat/automatic retry 없이 봉인 runner20batch와 CPU collector가 자연 진행합니다.

| Arm | 실제 job | Dependency | bounded 상태 |
| --- | ---: | --- | --- |
| PRUNE | 60757 | afterany:60739:60740 | PENDING Dependency |
| RECT | 60758 | afterany:60739:60740 | PENDING Dependency |
| CPU collector | 60759 | afterany:60757:60758 | PENDING Dependency |

실행 source `2689024d4c4a88cbe95f98b5e8a6191fad437894`, config SHA `caf21f441ab150484a7d6f3d769883978cdf2d70aded5122b5a9b3b2b3c46924`,
lock SHA `979274eba2ac5114d93b0e0a9a0b5ed4c2e31f46590f9b399ac2bc357712e396`.
Held owner/name/전체 argv/script bytes/source/native closure/node/QoS/GPU·CPU·memory/wall/exportNONE/requeue0/dependency/noCP/privacy 검사와 release가 완료됐습니다.
신규 두 run과 모든 현재 own server1 GPU allocation/admitted PENDING을 합친 DAG 폭은 2이며 실제 할당은 CAKE60739의 1GPU였습니다. 선행/타 job mutation은 없습니다.
W&B startup/실제 method GPU parity/첫 commit은 아직 미관측이며 SDK CPU PASS로 대체하지 않습니다. 정식 pending snapshot 인계 뒤 monitoring을 pause합니다.
실제 로컬 봉인자료 경로는 `/mnt/raid5/janghj/ODE-edit/local/gpt2xl-prune-rect-2k/20261007-v1/attempt-v1/` 입니다.
실제 immutable source archive에서도 다섯 task CPU suite 34개 모두 PASS했습니다. 이는 추가 GPU fit나 online 인증이 아닙니다.
