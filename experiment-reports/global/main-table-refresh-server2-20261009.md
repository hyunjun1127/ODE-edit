# SH2 완료 결과 main 표 반영

nonce: USER-GH-ALL-SH-COMPLETED-MAIN-TABLE-REFRESH-20261009-R1-SERVER2.

SH2 publication100c6f30의 table-rows.json SHA256
56f45b8f9148156f400c94df857b2d18b037bc8376299d22a7e3962a7021cdb9,
CSV/report 지정 SHA를 대조했다. 2026-10-09 23:05:48 KST snapshot이다.
GH 추가 scheduler 조회/모델 forward/job 변경은 없다.

Qwen zsRE MEMIT61956: Eff37.606352813852816/Gen36.68829906204906/Loc30.025379311583595.
AlphaEdit61960: Eff85.14393217893218/Gen78.60837662337663/Loc30.03645365808373.
공개-query 최종 W20/2000, source55039358, request-macro, 토큰분모6691/6691/11476.
README에 각각37.61/36.69/30.03 및85.14/78.61/30.04로 표시했다.
새 수치2행/6칸, 기존 PENDING→측정값이며 이전 실제점수와의 delta를 만들지 않았다.

Qwen FT zsRE61900은 구 token-prefix W20 완료이므로 재평가 필요/미등록으로 표시했다.
CF FT61898/BLUE61962, zsRE BLUE61964/FE61968은 ING,
나머지 Qwen 미완료 행은 PENDING. 소유자 검산 기준 snapshot이며 실시간 상태가 아니다.
GPTJ CF6 기존 성능은 SH2 재검산 delta0 보고를 수신하고 그대로 보존;
GPTJ zsRE61942–61947은 여전히 PENDING, 과거 수치 대입 없음.
Llama/W0/PRICE 및 historical 예외, CF FLUCON 정책은 수정하지 않았다.

[SH2 결과 검산 보고](../servers/server2/main-table-refresh-20261009/report-ko.md)
