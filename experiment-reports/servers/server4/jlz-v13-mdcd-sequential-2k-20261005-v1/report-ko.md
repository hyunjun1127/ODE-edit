# V13 MD/CD sequential 2k 구현·제출 인계 보고

상태: `SUBMITTED_RELEASED_MD_RUNNING_CD_RESOURCE_PENDING`.
두 독립 main과 CPU collector를 모두 held 검사 후 release했다. 아래 상태는
2026-10-05 05:46:08 KST(2026-10-04 20:46:08 UTC)의 마지막 bounded snapshot이며 완료 보고가 아니다.

| 단계 | job ID | 마지막 관측 | dependency |
|---|---|---|---|
| MD, BS100×20 | 58442 | RUNNING, GPU1 할당, runtime36초 | 없음 |
| CD, BS100×20 | 58443 | PENDING(Resources), 미할당 | 없음 |
| CPU collector | 58444 | PENDING(Dependency), GPU0 | afterany:58442:58443 |

두 arm 사이 과학 dependency/직렬화는 없다. CD의 자원 대기는 정상 scheduler pending이다.
신규 main B1 commit→B2 ownentry 및 GPU 기술 receipt는 `NOT_OBSERVED`, W20 결과는 `NOT_MEASURED`다.
제출·CPU fixture·기존 B1 qualification 재사용을 신규 trajectory 완료/PASS로 표기하지 않는다.

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
| 신규 CPU fixture | tiny two-batch MD/CD/state/transaction/observer/collector/cap, 실행 source12 PASS |
| 등록 최소 수리 CPU fixture | dependency None 회귀 포함13 PASS; 실제8B GPU 검증은 아님 |
| 실제 신규 GPU/main | 두 arm 제출·검사·release 완료, MD RUNNING/CD Resources pending |
| 신규 독립 reviewer | 미사용, owner source audit/red checklist |
| W20 수치/paired/비용 | NOT_MEASURED |

두 독립 GPU1 job을 held 등록·fullargv/source/resources/dependencies 검사 후 실제 release했다.
각 GPU1/CPU8/59392MiB/24h, exportNONE/Requeue0. CPU collector GPU0/CPU8/24576MiB/4h/afterany 두 arm.
현재 물리 자원이 없으면 정식 PENDING이 정상이며 기존 job은 선점/취소하지 않는다.
24h는 wall 상한이지 신규 2k ETA가 아니다. Aggregate output/atomic/scratch/margin reserve12GiB를 계획했다.
마지막 자원 snapshot의 disk available은35,667,652,608 bytes이며 기존 자료 삭제는0이다.

## 봉인 source와 등록 수리

실제 GPU 실행 source는 `2ff0ecc63dda2c830fbd68e6a2d76f3317a446f7`이다.
원 B1 source/math/runtime는 읽기전용 그대로이며 신규 sequential controller/collector만 추가했다.

| 봉인 대상 | SHA256 |
|---|---|
| configuration | b2745b667e011f2a27e81b5f5f275cb731b9f418342b20cf741855c2149fd429 |
| execution lock | a399d80257c38182cb67498d4aef8be12729f0e8fffe118f8d7f1e8b2701351c |
| source archive, 1,525,760 bytes | 86a1a21bbb969e6a83b1fac8a00d2aa3456e98a4b244664e1d9f64e68af735c8 |
| full held inspection | 4b596eea43e444feae994ef505563c2c2bf78666501cdcf0f01753607d1a4802 |
| local submission receipt | 392df1b8aea13ba1a33c492cbb0b2b90660459bded9c1ab3f6648e2c6d9f072d |

최초 등록 후 release 직전 `NO_ARM_SERIALIZATION` 검사에서 dependency 없는 `None`을
문자열 membership으로 다뤄 TypeError가 발생했다. 전3건 held/GPU실행0/release0 단계였다.
등록 로직만 source `21024d4b1760e017c4f49af8acfd0709be9fac1d`에서 최소 수리하고,
13건 CPU 회귀와 같은 held 등록 재검사 후 release했다. 추가sbatch0, cancel0, science변경0.
원 failure/source/archive/lock/config/launcher를 보존했으며 GPU 실행 source는 `2ff0ecc6` 그대로다.

공유 receipt는 `runs/server4/jlz-v13-mdcd-sequential-2k-20261005-v1/submission.json`,
compact source/held/CPU/initial snapshot inventory는 자기 audit 디렉터리에 게시한다.

Observer 매 batch pre/postcurrent와 W5/10/15/20 allseen을 기록한다. CPU collector는 exact raw identity/denominator와
20commit/19join/100H perarm 및 후보/paired/cost를 검산하며 보고+inventory 후 terminal을 쓴다.
낮은 quality/rank deficiency/finite rangeLS/집중/미수렴은 record-only. 기술 mismatch는 사실을 보존하고 해당 stage를 차단한다.
보고 수치는 scientific promotion 없이 한국어 factual-only로 남긴다.

원자료 root: `/data/janghj/ODE-edit/local/jlz-v13-mdcd-sequential-2k/20261005-v1/`.
Raw/metric 전체행/model/tensor/prompt/fullstdout는 Git 미게시·local KEEP. NoCP/exact_resume=NOT_AVAILABLE.
NO_BROADCAST_NOT_REQUIRED: 같은 서버 실행이며 source/compact 보고/SHA inventory만 공유한다.
일반 access helper는 명시 승인된 `runs/server4/.../submission.json` 한 경로에서 exit7/NOT_PASS였다.
이 task envelope의 exact prefix 허용을 좁은 게시 예외 receipt로 기록하며 공용 helper를 수정하지 않는다.
등록 initial/resource pending 인계 후 monitoring_active=false/automatic_resume=false, sealed20batch/collector만 자연 진행한다.
