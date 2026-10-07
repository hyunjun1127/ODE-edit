# GPT2-XL CAKE / AlphaEdit-BLUE 2k

상태: 구현·CPU 검산 완료, 아직 미제출. GPU/online PASS 아님.

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

CPU 66개 실행 중64 PASS, SDK 환경용2 skip, 실패0입니다. 실제 GPU·온라인 검증은 NOT_RUN입니다.
독립 bounded source reviewer가 original parser/import 및 조립을 검토했고, asset/revision/P mapping/closure 경계를 보완했습니다.
원/수리 fixture와 로컬 CPU stdout은 별도 보존합니다.

최신 combined cap2는 W0·준비·과학을 모두 포함합니다. 기존 root dirty/다른 source/job/raw는 보존합니다.
등록 이후 GPU runner20batch 및 독립 CPU collector가 자연 진행하며 장기 agent polling/자동 retry는 없습니다.
벽시간48h는 요청 상한이지 ETA가 아닙니다. 실제 제출 ID/source/config/lock/held/release는 등록 후 별도 기록합니다.

Raw는 local KEEP, Git/W&B에는 scalar·소형 manifest만. `NO_BROADCAST_NOT_REQUIRED`는 same-host 자료보존 예외입니다.
