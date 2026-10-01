# JLZ kernel reporting 오류 수리 / 재사용 preflight

Instruction ODEEDIT-GH-SH1-JLZ-EFFICIENCY-KERNEL-REPAIR-20261001-R1 FULL_READ.
Authority61966a4db583139b09c568fb561ea3af703086b1,
instruction SHAe4ef2c730e7f30cfb58f871d7d78032b3209e42749872a4cadf2c0dda2cdf871.
Parent contract/수치식/budget은 불변이며 PROTOCOL은 bbe19548과 diff0이다.
Shared root dirty를 보존하고 전용 clean repair branch에서 수행했다.

## 실제 원인과 비용

Exact accounting: 56704 / odeedit_jlz_efficiency_s1 / janghj / FAILED /1:0 /
401 GPU-sec. Collector56705 COMPLETED0:0 /1 CPU-sec.
Program393.51547831110656초; 이 값과 allocation을 더하지 않는다.
원 source bbe19548352e1cf543d54fdbb46ead3c6fb40436, lock56e9995639f56380adb9cb97d33e7875c68378b2b14e48f6a3ed1363e5c17757.
Kernel 비교 dict의 reference/candidate label과 timing() 통계 dict의 동일 키가 충돌했다.
실제 traceback 및 같은 timing()을 호출한 CPU 반례로 TypeError를 재현했다.
이는 OOM/수치 qualification 실패가 아니다. 원 logs/output/collector/source는 변경하지 않는다.

## Narrow 수정

Labels는 reference_kernel/candidate_kernel, 통계는 reference/candidate dict로 분리했다.
Production assemble_summary와 filter_direct_routes, run 소비자 및 collector schema를 연결했다.
정상12case/6comparison/60measurement JSON 직렬화, 통계 보존, direct 선택과 제외,
불완전 inventory fail-close를 실제 production helper를 호출하는 회귀검사에 추가했다.
Owner CPU21회 실행 PASS(상속 fixture 중복 포함); GPU kernel PASS라는 뜻이 아니다.
Core/native/reference/solver/budget/geometry/measurement/evaluation source SHA는 원 lock과 동일해야 한다.

## 재사용 및 필요한 RAM 재계산

| 항목 | 처리 |
|---|---|
| Fixed32 / short84 scalar qualification | 원21JSON allowlist SHA/size를 봉인해 REUSE |
| E123_MB8 UNQUALIFIED | 그대로 제외, 재시험 없음 |
| Native original/cached/batched | 저장 결과 REUSE, 새 native fit0 |
| Native-batched UNQUALIFIED | 그대로 유지 |
| Model/W0/H0·entry·teacher/key·adj | 새 cold process에서 준비, 기존 state/input identity 확인 |
| Reference 반환 R | noCP로 RAM 소실; 원 short12만 재계산, 모든 point/gradient/objective trace와 branch/status 일치 필수 |
| Kernel | 이전 일부 timing은 RAM에서 소실; 미완료12case 재측정 |
| RAM probe/B1008/observer | 미완료 범위만 실행 |

새 small 사용량은 REF12 reconstruction, native0이다. 재사용116을 새 호출에 더하지 않는다.
새1200/1000chain 없음. 원401 GPU-sec는 historical 비용으로 따로 보존한다.
재구성 불일치면 기술 HOLD이며 threshold완화나 repeat-to-PASS하지 않는다.
Same-candidate B100 reference/candidate는 새 job 동일 GPU에서 평가한다.
원 small timing은 기존 paired 측정의 선택자료 재사용임을 GPU UUID와 함께 명시한다.

새 job1GPU8CPU131072MiB12h, collector4CPU16384MiB2h.
기존56684와 별도 dependency=null, 실제 admission cap2 준수 후 held-inspect-release.
이전56684의 pause 유지, admission resource-only 외 조회0. 기존 job cancel/requeue0.
이번 초기 인계는 kernels.json 저장 및 B100 paired warmup2 PASS 증거까지다.
NoCP / exact_resume=NOT_AVAILABLE / NO_BROADCAST_NOT_REQUIRED.

추가 CPU binder preflight: 기존 hardcoded contract SHA 검사가 제출 전에 중단했다.
git diff로 1390b3b9의 resources.admission/parallel_override 두 필드만 바뀌었음을 확인했다.
최신 사용자 병행 정책 hash4485c11ffc9b0dbdf04dc5d672e284254fc15a37bc469271aa0166e8ffb48772로
결속했다. 수식/숫자예산/허용치 변경0, GPU cost0, 아직 output/lock 생성 전 실패였다.
독립 red focused production regression3 PASS 및 PRE_SUBMIT_CPU_SOURCE_PASS_WITH_WARNINGS:
과거 asset/accounting은 owner receipt 재사용 범위, 신규 비용과 재사용116을 중복 합산하지 않는다.
