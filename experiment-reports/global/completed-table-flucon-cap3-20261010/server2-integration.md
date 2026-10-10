# SH2 완료 결과·cap3 main table 반영

Nonce `USER-GH-S1-S2-COMPLETED-TABLE-S1-FLUCON-CAP3-20261010-R1`.
Owner accepted turn `01a1236d-9585-7183-88d1-d29261b81fcc`, publication `437f3f21900c6e2ca634804fb36b94cff612b83f`.
관측은 **2026-10-10 10:31:11 KST**, GH가 새 scheduler/GPU 조회를 추가하지 않았다.

| 모델 / zsRE method | job | Eff | Gen | Loc | E/G/Loc target token 분모 |
| --- | --- | ---: | ---: | ---: | --- |
| GPT-J / AlphaEdit+SPHERE | eval61947, 원 edit61735 | 99.67080586080587 | 96.28967490842491 | 27.99720049195529 | 5557/5557/9694 |
| Qwen2.5 / MEMIT | 수정 cold62081 | 41.49242604617605 | 39.44256493506493 | 26.392390042363424 | 6691/6691/11476 |

각2,000 requests이며 저장 predicted/target ID와 correctness를 독립 대조한 요청별 평균이다.
Loc은 loc_ans 정확도이며 W0agreement/token micro로 바꾸지 않았다. 공개-query 입력·target CPU 검산과
frozen evaluator/source/ordered cohort/terminal/최종CP provenance를 owner가 결속했다.
GH는 소형 결과 파일 SHA, 적격성·분모·query proof·source/config·원정밀도→half-up2 변환을 대조했다.
README 두 zsRE 행의6개 셀을 각각 **99.67/96.29/28.00**, **41.49/39.44/26.39**로 반영했다.
Qwen CF AlphaEdit62075의4개 상태 셀은 PENDING→ING으로 갱신했다. 기존 CF 수치 및 생성 셀은 그대로다.

## 출처와 한계

- GPT-J source `ce8d536fa7eb4dfdd38f7024381e68d212f44ea0`, config
  `cd0e8288c05e79d1ae4dc42cbceee638a2d82b1b083e8392d379ac63110835bc`, raw SHA
  `616f1808fbc3161745b9903138bef37fe134ecd2c9247074769c2f00b0ecd4cd`.
- Qwen source `7b5097aa447946e35de42229c22b0c0feabd11ae`, config
  `93e6f98a2a1c06e7da4320c6b5cd7837396781c6a377c8199cd8fb86f018a5c0`, raw SHA
  `246a1c74b664741b0d8978d9e7d65ff399384c52658d52b6017e3cebb239941b`.
- Owner 26행 inventory 중20행 적격/6행 미완료/issues0, CPU controls5 PASS. 기존 적격 결과는 SHA 재대조,
  이번 두 결과는 새 독립 request-macro 검산. frozen query source27members 검산.
- 최종CP는 기록된 SHA/provenance와 fresh stat를 대조했으며, 이번 audit에서 payload 전체 재hash/deserialize는 하지 않았다.
  원본 pretrained forward 수치 parity를 입증한 것으로 표시하지 않는다. 새 forward/GPU/W&B 과거 이력 변경 없음.

## 실제 cap3 적용 및 남은 항목

SH2 root/live admission 및 준비 WT의 local cap4→3, node/memory60416MiB/pattern 유지.
실제 allocated2, 기존6job DAGwidth2라 dependency 변경/hold/cancel/newGPU 모두0이다.
62075/62085 RUNNING, 62083·FE author62531/62532·B9 resume62538 PENDING을 유지한다.
62538은 parent62087 B9 이후 재개 예외이며 W20 수치 없음. 과거 깨진 context 결과는 제외한다.
SH1의 별도 완료 결과·추가 FLU/CON 제출 영수증은 아직 이 SH2 회신에 포함되지 않았으며 별도로 통합한다.

[SH2 보고서](../../servers/server2/completed-table-flucon-cap3-20261010/report-ko.md) ·
[정밀값·raw/source/config/CP](../../../audits/servers/server2/completed-table-flucon-cap3-20261010/table-rows.json) ·
[자원 적용](../../../audits/servers/server2/completed-table-flucon-cap3-20261010/cap-receipt.json) ·
[CPU 검산](../../../audits/servers/server2/completed-table-flucon-cap3-20261010/validation.json).
table-rows SHA `e30daac6d3397afbde9759294ba7b24ccbcd7382a68e159227b974141b60ba0a`.
원 raw/CP/frozen source KEEP, NO_BROADCAST_NOT_REQUIRED. 새 monitor/자동 retry 없음.
