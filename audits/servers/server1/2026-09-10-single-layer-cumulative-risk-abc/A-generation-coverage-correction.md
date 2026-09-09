# Generation scope clarification — 추가 A 관측 실행0

Authority: `ODEEDIT-GH-SH1-CUMRISK-GENERATION-SCOPE-20260910-R1`.
설계의 “Middle C step8”은 상위 C의5개 trajectory endpoint이며,
A selected Direct-C snapshot008을 기본 mandatory로 추가하는 계약이 아니다.
이전 provisional commit90f0707a의 누락 해석을 이 명확화로 철회한다.

실제 요구량은 A Native3+선택Direct-B/C6=9endpoints×60prompt와,
상위 C의5endpoints×60prompt이다. 기존 A/B report/manifest/receipt 및 raw bytes는 그대로다.
준비했던 별도 A8 observer/test/launcher는 production tree에서 제거했다.
Provisional source는 commit90f0707a에서 복구/확인 가능하며 model/GPU/Slurm 실행0이다.
현재 C job43274/source/설정은 변경하지 않는다. 최종 완비 판정은 원래 A/B/C 요구량에 따른다.
Scientific promotion=false.
