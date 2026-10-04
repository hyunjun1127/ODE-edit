# V13 MD/CD sequential 2k 접수·구현 보고

상태: `IMPLEMENTED_CPU_CHECKED_NOT_SUBMITTED`. 이 문서는 제출 전 사실 기록이다.

권한 nonce: `ODEEDIT-USER-GH-SH4-JLZ-V13-MDCD-2K-CAP2-20261005-R1`.
SH4/session `01a04939-b5c7-7a03-ba2d-ef3343d62cfd`, 실제 server4 전용 non-main worktree를 검증했다.
동일 nonce 기존 등록이 없어 신규 task로 시작했으며 기존 B1/V14 및 다른 job은 변경하지 않았다.

MD/CD 각각 독립 cold W0/H0, same first2000/BS100×20. 2,000 unique cohort, 두 arm 합 4,000 request applications.
각20 fit/20 commit/100 layer H append, final R2000/P4000/N20000; B21/추가 fit/새 baseline/CP 없음.
원 B1 실행 `08230095` planner/writer/evaluator와 actual qualification을 정확 closure로 재사용한다.
새 sequential source는 `project/run_scripts/jlz_realized_writer_sequential/`이다.

| 검산/단계 | 현재 사실 |
|---|---|
| authority envelope/contract | SHA 일치, 정본 및 sequential override 정독 |
| 원 source/runtime/native/assets | actual import closure34개 및 asset prior full SHA+fresh stat 재결속 |
| 입력 | CSV 전2000행/field 검산,20 native pack,26,000 평가 identity/token 행 봉인 |
| 신규 CPU fixture | tiny two-batch MD/CD/state/transaction/observer/collector/cap,12 PASS |
| 실제 신규 GPU/main | NOT_RUN/NOT_SUBMITTED |
| 신규 독립 reviewer | 미사용, owner source audit/red checklist |
| W20 수치/paired/비용 | NOT_MEASURED |

두 독립 GPU1 job을 held 등록·fullargv/source/resources/dependencies 검사 후 release하도록 구현했다.
각 GPU1/CPU8/59392MiB/24h, exportNONE/Requeue0. CPU collector GPU0/CPU8/24576MiB/4h/afterany 두 arm.
현재 물리 자원이 없으면 정식 PENDING이 정상이며 기존 job은 선점/취소하지 않는다.
24h는 wall 상한이지 신규 2k ETA가 아니다. Aggregate output/atomic/scratch/margin reserve12GiB를 계획했다.

Observer 매 batch pre/postcurrent와 W5/10/15/20 allseen을 기록한다. CPU collector는 exact raw identity/denominator와
20commit/19join/100H perarm 및 후보/paired/cost를 검산하며 보고+inventory 후 terminal을 쓴다.
낮은 quality/rank deficiency/finite rangeLS/집중/미수렴은 record-only. 기술 mismatch는 사실을 보존하고 해당 stage를 차단한다.
보고 수치는 scientific promotion 없이 한국어 factual-only로 남긴다.

원자료 root: `/data/janghj/ODE-edit/local/jlz-v13-mdcd-sequential-2k/20261005-v1/`.
Raw/metric 전체행/model/tensor/prompt/fullstdout는 Git 미게시·local KEEP. NoCP/exact_resume=NOT_AVAILABLE.
NO_BROADCAST_NOT_REQUIRED: 같은 서버 실행이며 source/compact 보고/SHA inventory만 공유한다.
등록 initial/resource pending 인계 후 monitoring_active=false/automatic_resume=false, sealed20batch/collector만 자연 진행한다.
