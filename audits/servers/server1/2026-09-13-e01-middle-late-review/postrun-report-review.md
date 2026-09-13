# E01 상세 보고서 postrun 내용 점검

상태: `CONTENT_CHECK_PASS_FINAL_PACKAGE_REHASH_PENDING`.

점검 source는 `089dfe7eb807a904752272cb3d0697d06b94eca8`, tree `df5ae9f68cc2b160a54c3e709934c20c44ec563b`이다. 이번에는 새 raw/tensor 해시나 모델 실행을 반복하지 않고, 실제 생성된 한국어 보고서·CSV·source·기존 검산 receipt를 대조했다.

다음 표시값이 CSV와 일치했다.

- 핵심 원본/replay RS/PS/NS 표12행: counts/denominator/Δpp/lost/gained.
- Strict/token 표6행: 원본 strict → replay strict → 원본 token → replay token 순서. NS는 target-true, RS/PS는 target-new다.
- At-write 표12행: 원본 at-write와 원본/replay final 구분, numerator·분모·loss/recovery.
- Fullseen NLL/margin 분포36행: mean/median/p90/p95/p99 표시 반올림.
- Weight/history 차이4행: reference norm을 분모로 한 relative F, maxabs, changed count.
- Signed derivative 요약12행: 부호별 count와 실제 mean. Clipped progress_slope를 대신 사용하지 않았다.
- 전이 CSV522행 모두 lost/gained/unchanged-success/unchanged-failure 합이 원분모와 일치했다.
- Markdown14개 표의 열 수가 모두 일치했다.

Replay-own at-write는 각 endpoint200 requests만 관측되었다. 원본 at-write를 replay 자체의 과거 관측으로 바꾸지 않았으며 missing prefix5800/9800 및 신규 window800 requests를 명시했다. B060/B100의 metadata exact 문구는 기존 `tensor_metadata_comparison` receipt와 일치한다. 성능이 비슷하다는 이유로 tensor/trajectory fidelity를 PASS로 승격하지 않았다.

경미한 표시 결함 하나는 부모 담당자가 수정했다. 비용표의 과거45719/45805 상태가 빈 칸이었고, 이제 `PRIOR_COMPLETED_RECEIPT_REUSED`로 표시된다. 원래 비용 CSV 수치는 변하지 않았다.

`validation.json`은 focused17개 PASS 및 reducer 산출물7개의 실제 byte 재현을 기록한다. 이 점검에서 tests/raw reducer를 다시 실행하지 않았다. Plot receipt의 source SHA는 현재 plotting/package source와 일치한다.

현재 blocking 내용 불일치는 없다. 다만 최종 package manifest/rooted receipt의 전체 재해시·main 통합은 점검 시점에 아직 부모 담당자의 후속 작업이었다. 이 문서는 final package rehash/main push PASS를 먼저 주장하지 않는다. 검토한 report SHA와17개 소형 member identity는 `postrun-report-review.json`에 보존했다.

GPU/model/Slurm/remote0, native/runtime 변경0. 이번 Middle/Late 관측 보완 완료와 전체 E01 미완료(첫 native cell4/20)를 구별한다. 과학적 원인 종합은 GH 소유다.

마감 통지: 부모 담당자가 이 점검 뒤21-member package 재해시 PASS를 전달했다. 별도로 소형 manifest SHA `bf975b9d8bce02002bea2d55843016c3d9fdd0e1f83cd7714037146e781861b7` 및 rooted receipt SHA `b29b719eae31e2a47ad89cb7e3ad23dbdf56b856a1dbeaf8aab694d6d298d9b8`가 실제 파일과 일치함을 확인했다. Member root는 부모 검증값 `3ac18e8057b142a886c8d747509fa8e7b8d2668bf2dd474cd7db3495e24576a8`이다. 이번 postrun worker는 전체 package 검사를 중복 수행하지 않았고 main 통합은 부모 담당이다.
