# GPT2-XL CAKE / AlphaEdit-BLUE 2k

상태: 실제 held 등록·검사·release 완료. CAKE 초기 cold load/context 및 W&B remote identity 검산 완료;
AlphaEdit-BLUE는 자원 PENDING. B1 commit/GPU method PASS·전체 완료는 아직 미관측입니다.

등록後 단발 초기 snapshot에서 CAKE W0 first2000 26000 prompt-pair(R2000/P4000/N20000) 및
B1 PRE 1300 prompt-pair(R100/P200/N1000)의 observer 비변이·feedback0 summary를 확인했습니다.
해당 관측은 W0 771.19초, B1 PRE 40.19초이며 전체 native fit ETA가 아닙니다.
첫 write commit→다음 entry는 아직 미관측으로 남기고 bounded initial partial/resource PENDING 인계 뒤 monitoring을 pause합니다.
봉인 runner·collector는 계속 자연 진행하며 새 recurring monitor/자동 재시도는 없습니다.

직접 사용자 후속 “blue는 alphaedit blue를 사용하자”에 따라 원 MEMIT-BLUE 선택만
AlphaEdit-BLUE로 대체했습니다. 정본 원 bytes/이전 보고는 보존했습니다.
두 arm은 독립 cold W0, 같은 ordered first2000, BS100×20입니다.

| Arm | 원 native | 층 | L2 | H | P |
| --- | --- | --- | ---: | --- | --- |
| CAKE | CAKE 0b378234 | 13–17 | 40 | 5층 native carry | 기존 5층 |
| ALPHAEDIT_BLUE | BLUE 311b076a AlphaEdit | 13/17 | 80 | 2층 native carry | 기존 physical slots0/4 RAM 선택 |

두 원 clone/source를 수정하지 않았으며 private import/path compatibility diff만 채택했습니다.
actual native JSON constructor, dict target_new, original target fit/solve/cast/add를 유지합니다.
Context generator는 각 원 구현의 첫 cold 준비에서 한 번만 사용하며 EasyEdit context를 자동 이식하지 않습니다.
기존 W0 raw를 무조건 이식하지 않고 각 own native 입력이 결속된 cold W0를 한 번 관측합니다.
소스/code/config/SHA만 봉인; model/P/H/edited weight/optimizer checkpoint 저장0.

CPU 66개 실행 중64 PASS, SDK 환경용2 skip, 실패0입니다. Frozen archive의 task CPU29개도 PASS입니다.
독립 bounded source reviewer가 original parser/import 및 조립을 검토했고, asset/revision/P mapping/closure 경계를 보완했습니다.
원/수리 fixture와 로컬 CPU stdout은 별도 보존합니다.

최신 combined cap2는 W0·준비·과학을 모두 포함합니다. 기존 root dirty/다른 source/job/raw는 보존합니다.
등록 이후 GPU runner20batch 및 독립 CPU collector가 자연 진행하며 장기 agent polling/자동 retry는 없습니다.
벽시간48h는 요청 상한이지 ETA가 아닙니다.

| Arm | 실제 job | Dependency | bounded 상태 |
| --- | ---: | --- | --- |
| CAKE | 60739 | 없음 | RUNNING; cold load/context 확인, B1 commit 미관측 |
| ALPHAEDIT_BLUE | 60740 | 없음 | PENDING Resources |
| CPU collector | 60741 | afterany:60739:60740 | PENDING Dependency |

실행 source `d88e51c0138f3504214faea7a566b7493b763cce`, config SHA `aae8d3931c77d0a8726a0f91feecad259faef6800171e18088d5a1d55a66507b`,
lock SHA `b9ceb7fcb7613d11e7b986be095c1022ea43b52d0d447c7949c3f7d259a21c73`.
Owner/full argv/Slurm source bytes/closure/resources/dependency를 held 상태에서 검산한 뒤 release했습니다.
CAKE [실시간 run](https://wandb.ai/wkdguswns2256/layer%20allocation/runs/5bdcffd677dc40e2)은 실제 remote ID/name/config(job60739 포함) startup 대조를 통과했습니다.
이는 metric history readback 또는 science 완료 인증과 다릅니다. AlphaEdit-BLUE 온라인 startup은 아직 미관측입니다.
실제 local receipt: `/mnt/raid5/janghj/ODE-edit/local/gpt2xl-cake-blue-2k/20261007-v1/attempt-v1/`.

Generic access helper가 명시 허용 plans/servers 경로를 거절한 사실을 audit에 별도 기록했고,
공유 policy를 바꾸지 않고 사용자 exact own-scope 승인으로 게시했습니다. Helper PASS로 표시하지 않습니다.

Raw는 local KEEP, Git/W&B에는 scalar·소형 manifest만. `NO_BROADCAST_NOT_REQUIRED`는 same-host 자료보존 예외입니다.
