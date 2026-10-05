# SH3 V14 budget1.5 R2 / fixed first2000

상태: **RELEASED / MONITORING_PAUSED_AWAITING_USER**.
2026-10-05 17:50:33 KST의 한정 snapshot: GPU58786 `PENDING(Resources)`, CPU58787 `PENDING(Dependency)`.
Actual GPU qualification/B1/W20은 NOT_OBSERVED이며 제출을 실행 성공으로 표시하지 않는다.

| stage | job | dependency | resource | wall |
|---|---:|---|---|---|
| minimal qualification → cold B1..B20 | 58786 | 없음 | ubuntu/gpu,1GPU,8CPU,121856MiB | 48h |
| CPU raw reducer/partial coverage | 58787 | afterany:58786 | ubuntu/gpu,0GPU,8CPU,24576MiB | 4h |

Owner janghj, `exportNONE`, `Requeue0`. 두 job의 held owner/full argv/source/script/resource/dependency 검사를 완료하고 collector→GPU 순서로 release했다. 수동 hold 없음.
실행 source `ab7c910695377bef81ee35280502ecf176cb7d38`; tree `cc60cbf764903b288cda2cc2684c0afaddc44cbb`.
Lock SHA256 `7bdb736a6676cbbc61634a51e3304f1ee05796bb59658f802c4ed1dc7799cddc`, config SHA256 `8cf9374f9b4cc24a6fc823671986fcceda470b15089953a6458c520aaeac9896`.
한정 snapshot 뒤 scheduler/log/result polling 중단. Sealed runner는 기술 qualification→W20→collector를 자동 수행한다.


- 원 S4 source `2ab04d0b`의 actual writer/subject/optimizer 출발점을 유지하고 별도 SH3 source로 봉인한다.
- shared1.5/local.75 true capped Euclidean projection, whole-B cached transpose VJP,
  첫 site causal adjoint pruning, complete-owner R4a 및 same-method fallback을 연결했다.
- 원 MB2 contiguous schedule을 유지한다. 더 큰 MB/R4b/OOM retry 최적화는 미검증·비활성이다.
- cold W0/H0→20 fits/20 commits/100 H appends/19 joins. B1은 중간 관측이며 quality gate가 아니다.
- CPU 신규 6검사 PASS. 기존 15 NumPy algebra 증거 exact reuse. Owner audit이며 독립 reviewer는 사용하지 않았다.
- GPU qualification은 B2 fixed candidate 1회, fit/update0. Actual native/full gradient·projection·state checks 후 같은 cold W0/H0에서 main을 시작한다.
- S3 torch2.9.1+cu128/transformers4.57.1/readiness Python 및 모델/C0/contexts/fixed2k/native inputs 결속.
- 기존 S3 W0 first2000 evaluator/token/runtime exact reuse, actual cold W/H hash는 runner에서 재확인한다.
- host 계산 peak 약40.35GiB, 요청119GiB/8CPU/1GPU/cap1. 디스크 현재 약40.5GiB, 예약16GiB.
- wall48h는 최대 요청값이며 ETA가 아니다. 과거 S4 fit20배=11.33h는 역사 외삽이며 새 S3 실측이 아니다.
- noCP / exact_resume=NOT_AVAILABLE. Raw/prompt/tensor/fullstdout은 local만, 기존 자료 KEEP.

누적 매batch 평가와 W0/B1/W5/W10/W15/W20, RSPSNS·조화평균·TF strict/token-micro/prompt-macro(Ntrue),
NLL/tail/paired/cohort 및 CPU independent reducer를 연결했다. 제출과 완료는 별도 receipt로 갱신한다.

재현: frozen `attempt-r1/source`와 `config.json`, `execution.lock.json`을 원본으로 사용한다. 중복 제출 금지.
Local attempt: `/data/janghj/ODE-edit/local/jlz-v14-budget15-r2/20261005-v1/attempt-r1`.
