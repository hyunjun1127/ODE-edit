# SH4 EN adaptive-nullspace: T0 / B1 native 중간 기록

Instruction: ODEEDIT-S06-EN-ADAPTIVE-NULLSPACE-B300-SH4-MIGRATION-V1

Job `51260`, execution `b6e86234640ca127546aee094f2a67bbe684a490`.
본 기록은 T0와 B1 native 저장 receipt를 읽은 중간 관측이다. B1 네 endpoint 및 B300 완료 보고가 아니다.

## Actual T0

- finite/identity PASS; precision `NOT_ESTABLISHED`.
- T0 553.2383925765753초. Entry weight와 RNG 복원 true.
- 고정 4 reference에서 cached/physical objective 차이 0, gradient bitwise 동일.
- 사전 고정 단일 FD relative error 0.006228139038. 광범위 FD grid나 전체 수치 검증 PASS가 아니다.
- z 비교 native/cache+head1/cache+head4: 14.8104/11.9597/7.8544초.
- cache1 max NLL 차이 0.0002870559692, max gradient relative 0.00338629606.
- cache4 max NLL 차이 0.0002806186676, max gradient relative 0.00646168480.
- 정지 iteration은 일치했으나 위 차이는 보존한다. 실제 검사 요청은 4개이며 production chunk16 전체의 별도 parity PASS를 주장하지 않는다.
- Exact weighted rank267, numerical released2, ADAPT released267은 T0 panel만의 결과다.

원 evidence: `attempt-v1/output/T0/` 및 `T0-result.json`. 과거 T skip/FD waiver를 상속하지 않았다.

## B1 fresh shared native

- 100 requests, compute_z100/key1/solve1, candidate 중 history append0.
- Outer wall194.83989013079554초, actual delta norm7.611672532332007.
- Entry SHA `4300365feb31fec8b3ea07a9228d2245814b1bccb04f9ff5d9201a64cfd3e2e6`.
- Native endpoint SHA `41909e02c8a0bee4354f9652ce105f6a5791317f73f046c129ca75aee1d12f6a`.
- Current keys14336×5346, logical prefix5042/byte variants304를 그대로 유지, 전체 가중합1.
- 이 동일 native에서 N4/EN_EXACT/EN_NUM/EN_ADAPT를 계산한다. 이후 B2/B3는 각자의 trajectory다.

NoCP 유지, 다른 job 변경0, teacher 역전송0. 본 승인 범위의 실행·관찰·보고를 계속한다.
