# SH2 중간 결과의 README 반영

Parent `USER-GH-S1-S2-BASELINE-REFRESH-S2-FLUCON-20261010-R1`, owner accepted turn
`01a124db-95bd-7231-981d-429bbc98305c`. Owner e2537348, main 통합417a12e7.
결과 관측 **2026-10-10 17:18:09 KST**. 완료23행/불일치0 중 새3개 결과를 반영한다.

| Qwen2.5 방법/dataset/job | Score | Eff | Gen | Loc |
| --- | ---: | ---: | ---: | ---: |
| AlphaEdit CF62075 | 83.03 | 99.05 | 97.55 | 63.36 |
| AlphaEdit zsRE62083 | 해당없음 | 85.05 | 78.18 | 30.79 |
| MEMIT-FE zsRE62085 | 해당없음 | 0.00 | 0.00 | 0.03 |

62085의 Eff/Gen0은 실제2,000요청 저장 정오값을 재집계한 측정값이다. 미측정 대체0이 아니다.
원 Loc은 `0.033333333333333326`이며 표에는0.03. 두 zsRE의 E/G/Loc token 분모는
6691/6691/11476, Loc은 loc_ans 요청별평균이다. 진단 token-micro Loc과 다르다.
공개 query24858개의 입력/target mismatch0, 원 frozen source/query SHA 결속을 owner가 확인했다.
CPU query 및 raw 집계 검산은 pretrained 수치 parity 주장이 아니다.

세 chain source `7b5097aa447946e35de42229c22b0c0feabd11ae`, 각20commit/W20/2000 및 원 terminal/raw 확인.
table-rows SHA256 `ea0b0c6df054428565e0a489d8df88fd04f118442235a992a9d6fdefcfeb5549`.
원 raw SHA는 순서대로:

- CF62075: `a951d9d4dc8296ce0f8666d127e73eb948d7c08d281de25b6cc4e71426975439`.
- zsRE62083: `158d862a6af68e39266b3cbff27c6607e0f2ad31696d04c4e183b95ea41d9e19`.
- zsRE62085: `d86b34e16dc09f247c22fa0a28daa72f53e9d1d4e1ef0d3a7bf728f13eb159e4`.

GH 검산은 owner compact자료 SHA/실측 eligibility/원미반올림값→표시/README 변경범위에 한정한다.
README10수치셀 및 Qwen author62531 RUNNING 상태1셀만 변경, 기타factual/생성/역사예외를 보존한다.
FEauthor62532 및 resumed SPHERE62538 PENDING은 유지한다. native FE와 history variant를 합치지 않는다.

CF FLU/CON은 DEFERRED 유지. 신규평가는 구현CPU15/source166 및 CP복원입력 검산 단계로
보고됐으나 **실제 신규ID는 아직 미수신**이다. 기존 SH1평가62581–62583/62259–62261은 중복금지.
역사Llama3개는 별도 legacy consumer binding이며 새로운 official편집완료로 재명명하지 않는다.
cap3 및 기존62531→62532→62538 보존. GH job변경/새GPU/대용량전송/삭제/monitor0.

[원 report](../../servers/server2/baseline-refresh-s2-flucon-20261010/report-ko.md) ·
[정확 source/config/분모/CP/raw](../../../audits/servers/server2/baseline-refresh-s2-flucon-20261010/table-rows.json).
