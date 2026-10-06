# Scientific W&B metric schema — SH4

정본·envelope·원첨부 SHA를 확인했고 담당 수락했다. GPT-J 미봉인 caller에 대한 CPU 통합 작업만 수행했다. 새 Slurm/GPU/모델 계산, 기존 job/run 수정, 역사 snapshot 업로드는 없다.

## 현재 상태

- 기존 source `8f554981`의 매핑과 B4/B5/W0 검산을 재사용했다. current/pre·current/post는 R100/P200/N1000, B5 all-seen만 R500/P1000/N5000이다. W0_first2000은 R2000/P4000/N20000이며 다른 horizon으로 대체하지 않는다.
- R/P/N별 TF/NLL/선호 성공률을 분리하고 pct/nats/true-new margin과 성공률 harmonic을 유지한다. W0의 N 비교는 같은 cohort raw subset이다. 과학 평가 정의를 바꾸지 않았다.
- helper/caller 통합 검사에 실제 raw에서 얻은 scalar payload를 fake SDK worker에 전달하고 eval 축 edits/step_sync=false 및 fit/global_candidate 축을 확인하는 검사를 추가했다. whitelist 통과만으로 축 검증을 대신하지 않는다.
- preflight에 helper·정본 SHA를 포함했다. 향후 freeze는 CPU 통합 receipt와 동일한 caller/helper/policy bytes를 확인해야 한다. 아직 지원되지 않는 helper로 봉인하거나 제출할 수 없다.
- 로컬 metadata/fake-SDK 검사 **11 PASS**, AST/import/config **31파일 PASS**. 과학 toy/model/GPU PASS가 아니며 owner 검사다. 별도 reviewer 없음.

현재 확인 main의 SH1 helper는 아직 `COMPARISON_SCHEMA=price-first2k-scalar-v1`과 비교 whitelist를 제공하지 않는다. 따라서 **caller 준비 완료 / shared helper 통합 미완료 / 실제 online NOT_OBSERVED / 미봉인·미제출**이다. SH1 소유 helper는 수정하지 않았다.

SH1 배포 후 `review_tracking` → `preflight` → `prepare --tracking-review`에서 동일 소스 결속을 확인한다. 실제 online identity/metric readback은 별도로 승인된 다음 과학 run의 기존 startup/finish 범위에서만 수행한다. 이번 정책 task로 job을 만들거나 과거 run을 수정하지 않는다.

CPU 상세 receipt는 audits의 integration.json에 경로·SHA를 기록했다. 원 raw·prompt·tensor·credential은 로컬 보존. NO_BROADCAST_NOT_REQUIRED. 반복 모니터링·자동 재개 없음.

