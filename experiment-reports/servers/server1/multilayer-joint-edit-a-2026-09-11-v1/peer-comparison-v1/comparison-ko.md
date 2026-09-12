# Middle A0 / N4 / B-OS — 완료분 비교

A-OS는 아직 이 표에 terminal이 없다. 전체32 endpoints/네 chain 완료가 아니다. 동일 common READY/member root 및 평가 panel SHA를 확인했다. SH1 A0는 local raw 재검산, SH2 N4/B-OS는 owner의 raw 검산을 결속한 Git publication을 독립 재해시한 수준이다. SH2 private raw를 여기서 다시 읽거나 cross-hardware inference parity를 증명한 것은 아니다.

| panel | metric | arm | n/d | % | new NLL | true NLL |
|---|---|---|---:|---:|---:|---:|
|Current100|NS|N4|711/1000|71.100|8.478977|5.960385|
|Current100|NS|A0|682/1000|68.200|8.350999|6.044962|
|Current100|NS|B-OS|706/1000|70.600|8.191680|5.640650|
|Current100|PS|N4|197/200|98.500|1.194420|10.573924|
|Current100|PS|A0|188/200|94.000|2.042551|9.602004|
|Current100|PS|B-OS|197/200|98.500|0.721487|10.213814|
|Current100|RS|N4|100/100|100.000|0.026669|14.254081|
|Current100|RS|A0|97/100|97.000|0.289325|12.006464|
|Current100|RS|B-OS|100/100|100.000|0.032181|13.860775|
|Fixed100|NS|N4|696/1000|69.600|8.672807|6.515947|
|Fixed100|NS|A0|687/1000|68.700|8.741071|6.569171|
|Fixed100|NS|B-OS|695/1000|69.500|8.416586|6.262721|
|Fixed100|PS|N4|194/200|97.000|1.502077|9.628624|
|Fixed100|PS|A0|193/200|96.500|1.448443|9.710147|
|Fixed100|PS|B-OS|192/200|96.000|1.227015|9.262562|
|Fixed100|RS|N4|100/100|100.000|0.227358|12.724216|
|Fixed100|RS|A0|100/100|100.000|0.216743|12.854098|
|Fixed100|RS|B-OS|100/100|100.000|0.213433|12.511553|
|Past100|NS|N4|686/1000|68.600|7.743244|5.907326|
|Past100|NS|A0|674/1000|67.400|7.833631|5.992606|
|Past100|NS|B-OS|697/1000|69.700|7.432674|5.504968|
|Past100|PS|N4|193/200|96.500|1.319750|9.972620|
|Past100|PS|A0|195/200|97.500|1.236449|10.450047|
|Past100|PS|B-OS|191/200|95.500|1.081097|9.682976|
|Past100|RS|N4|100/100|100.000|0.027743|13.628889|
|Past100|RS|A0|100/100|100.000|0.033141|13.862134|
|Past100|RS|B-OS|100/100|100.000|0.022506|13.197546|

RS/PS는 new NLL < true NLL, NS는 반대이며 tie는 실패다. A0 Current RS/PS가 N4보다 낮고, 같은-bank Base/Past 위험도 native보다 높았다. B-OS도 Base/Past 위험이 증가했으며 PCG 두 RHS의 finite 미수렴 상태를 함께 읽어야 한다. 이 차이만으로 공동 편집이나 보정 공간의 불가능성을 결론내리지 않는다. 동일 cohort aggregate 표이지 request별 원자료를 다시 join한 paired-delta 추론표는 아니다.

## 비용과 미완료

A0 allocation1561초 + 공통준비916초. SH2 B-OS allocation 14284초, 이전 실패 98초는 별도다. 서로 다른 host의 합계 wall speedup이나 forward 비용 동등성으로 해석하지 않는다. A0에는 PCG가 없고 B-OS는 a/u 각20회, relative residual0.1563698513/1.7954058935다. B independent Audit/attribution 미기록은 보간하지 않는다. A-OS 및 나머지 지정범위는 pending이다.

Peer source publication `47e3bd81bba5bc671780db126bd07323f856cfcb`, report SHA `1d3681c70b165ef15ce6200db7cd28213cee8d8f65f7b6defd616a3174f5f5b9`. Shared functional kernel은 a1fc35fc이고 execution/analysis source는 각 원패키지에서 구분했다. scientific_promotion=false.
