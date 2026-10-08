# Qwen CF display transport 수리 (2026-10-09)

수락 nonce: `GH-SH4-CF-DISPLAY-REPAIR-KEEP-HEALTHY-20261009-R1`.

## 수리 및 증거

- 원 native NumPy request/cohort mean+around를 실제 CF strict NLL bits에서 계산해 세 display companion을 전달한다. 원 E/G/S/Score/display Score, native math, 평가/생성 일정은 불변이다.
- 반올림 경계 CPU fixture에서 legacy caller 오류를 재현하고 own caller→공통 schema 통과를 확인했다. 모델/forward/fit/온라인 업로드는 추가하지 않았다.
- fresh scontrol + sacct + 원 submission/source/Command/WorkDir 결속: CF61743/45/47/49/51/53 및 archive61744/46/48/50/52/54 모두 PENDING, Start Unknown, elapsed0, allocation없음. 이 task의 실제 raw/evaluation/commit은 아직 없다. 실제 Qwen 오류 관측은 NOT_OBSERVED이며 온라인 상태는 NOT_STARTED다.
- zsRE61755..61766은 원 source/config/IDs 그대로 KEEP. first61755의 resource dependency만 새 CF tail archive로 연결한다. tuning61674 및 다른 task는 변경하지 않는다.
- 원 execution `d614add5e4c650821ed8d2503c071a1e02605ca8`과 `/data/janghj/ODE-edit/local/qwen-baselines-12-20261009/execution-noqual-r2`는 보존한다.

## 자원 및 등록 계획

05:22 KST 새 bounded inventory에서 tuning61674 GPU2와 별도 held-out61776/61777 각GPU1, 합4가 관측됐다. 최신 본 task cap2보다 큰 기존 할당은 취소/수정하지 않는다. 새 CF 첫 job은 이 세 actual frontier 모두 afterany 대기하며 이후 GPU1→archiveGPU0 직렬, 마지막 archive 뒤 기존 zsRE 직렬을 유지한다. 신규 실행이 cap초과를 더하지 않는다.

fresh disk available 54,373,789,696B; batchguard32GiB 및 serial final-only verified archive/reclaim 유지. receiver/adoption/submission-lock은 새 frozen caller와 실제 새 ID로 별도 봉인한다. 원 archive/raw/CP는 변경하지 않는다. per-payload admission/독립 fullSHA 검증/consumer 종료 전 원본 삭제 없음.

최초 source 게시 단계는 IMPLEMENTED_CPU_CHECKED_NOT_SUBMITTED였으며, 05:25 KST 실제 교체 등록·held검사·release까지 완료했다. GPU qualification/resume equivalence는 NOT_RUN_USER_DISABLED. GPU/온라인 PASS 아님. README는 GH 단독 통합이다.

검증: CPU52개, official source157 SHA / external task imports0, diffcheck. 별도 독립 reviewer는 사용하지 않았으며 owner code review를 수행했다. 소형 source/receipt만 Git, 원 raw local KEEP, NO_BROADCAST_NOT_REQUIRED.

## 실제 등록 및 보존 결과

실행 source: `dc80ec529c940019d1bee27a67a4e908eb37cc64` (수리 구현 `aa57a90c`). 새 attempt는 `/data/janghj/ODE-edit/local/qwen-baselines-12-20261009/execution-cf-display-r1`이다. 후속 보고서 게시 commit은 실행 source와 구분한다.

| CF method | 취소한 미시작 GPU / archive | 새 GPU / archive | 새 GPU dependency |
|---|---|---|---|
| FT | 61743 / 61744 | 61783 / 61784 | afterany:61674:61776:61777 |
| MEMIT | 61745 / 61746 | 61785 / 61786 | afterok:61784 |
| AlphaEdit | 61747 / 61748 | 61787 / 61788 | afterok:61786 |
| AlphaEdit-BLUE | 61749 / 61750 | 61789 / 61790 | afterok:61788 |
| MEMIT_FE | 61751 / 61752 | 61791 / 61792 | afterok:61790 |
| SPHERE | 61753 / 61754 | 61793 / 61794 | afterok:61792 |

새 job name은 각각 `qwen-cf-ft`, `qwen-cf-memit`, `qwen-cf-alphaedit`, `qwen-cf-alphaedit_blue`, `qwen-cf-memit_fe`, `qwen-cf-sphere`; archive는 같은 이름의 `-archive`이다. 각 archive는 대응 GPU afterok이며 성공적인 실제 W20/검증된 이관·reclaim 뒤 다음 cell을 허용한다.

zsRE61755..61766의 source/config/IDs는 유지했다. 첫61755만 잠시 hold→dependency `afterok:61794` 재연결→release했다. 원 CF/archive12개는 전부 hold 후 downstream-first 취소했으며 원 자료 삭제는 없다. 기존 source와 새 source의 CF config bytes 동일성, 새6개 archive adoption/실제 job/source binding, 유지 zsRE Command를 검산했다. 취소 oldID는 새/유지 active DAG에 남아 있지 않다.

bounded postrelease snapshot: 새12개와 유지 zsRE12개 모두 PENDING(Dependency), 수동 hold없음. tuning61674와 별도61776/61777은 RUNNING 그대로였다. 새 CF W0/평가/fit/online init은 아직 미시작이며 W&B URL/remote delivery는 NOT_STARTED다. receiver 준비와 실제 checkpoint 이관/삭제를 구분하며 현재 신규 payload 전송0/삭제0이다.

정확 source/config/샘플 lock SHA, ID/name/dep/상태/시각 및 보존 목록: `audits/servers/server4/qwen-baselines-20261009/cf-display-repair/submission.json`. 전체 hold/cancel/inspection/release/재연결 증거는 ignored attempt root에 보존했다. 새 recurring monitor/자동 retry는 없다.
