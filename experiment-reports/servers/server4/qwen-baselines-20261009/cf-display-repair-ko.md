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

현재 이 문서의 최초 게시 시점은 IMPLEMENTED_CPU_CHECKED_NOT_SUBMITTED. 실제 취소/새 등록/연결은 후속 compact receipt로 추가한다. GPU qualification/resume equivalence는 NOT_RUN_USER_DISABLED. GPU/온라인 PASS 아님. README는 GH 단독 통합이다.

검증: CPU52개, official source157 SHA / external task imports0, diffcheck. 별도 독립 reviewer는 사용하지 않았으며 owner code review를 수행했다. 소형 source/receipt만 Git, 원 raw local KEEP, NO_BROADCAST_NOT_REQUIRED.
