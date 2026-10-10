# SH1 history FLU/CON 완료 결과 통합

Parent nonce `USER-GH-S1-S2-BASELINE-REFRESH-S2-FLUCON-20261010-R1`.
Owner publication `d6733abb1afe9022584699b36410bd7e685b9d47`, accepted turn
`01a124dc-4d21-70f0-9fa7-94452f671f11`.

| 별도 MEMIT_FE_HISTORY | 원 편집 job | 평가 job | FLU raw | CON raw | 표시 FLU×100 / CON×100 |
| --- | --- | --- | ---: | ---: | --- |
| GPT-J | 61927 | 62581 | 5.334287172759865 | 0.01054845862757061 | 533.43 / 1.05 |
| Llama | 61928 | 62582 | 5.273934747386175 | 0.07580516374408931 | 527.39 / 7.58 |

SH1은 각2,000 case/20,000 prompt, 유효분모2,000/missing0 및 reference-bound CPU 전량재채점,
원W20 CP fullSHA/20commit/cohort/source/config를 검산했다. GH는 compact자료 exactSHA,
raw평균→표시변환 및 README의 변경셀 한정을 확인했다. GH의 독립 대형raw재채점/새GPU평가가 아니다.
generation 실행 source는 `37094d92d70d0499304da9f7b82ab59340c57232`이다.

- table-rows SHA256: `aafa6d5e9d4cb85948a4d21c712bd0753635ec6adeedc44ad00123a620222024`.
- GPT-J endpoint SHA256: `90fcb991278a39b83cfcd24fa2c99ab7571b1896e038689878a85cfe9e1b6dce`.
- Llama endpoint SHA256: `6a4fd0118cd7c59ae0da9141b70b93af1a3c2ee95ba8c25b2ccb71184108d528`.

2026-10-10 **17:10:08 KST** snapshot: corrected Qwen history62583 RUNNING, collector62584 PENDING.
Author62529 RUNNING/18commit/W20 없음, 62530 PENDING. 미관측 generation을 수치로 채우지 않는다.
SH1 신규GPU/submit/cancel/전송/삭제0, cap3 유지.

zsRE6개 공개-query 및 저장 predicted/target 요청별평균 E/G/loc_ans는 기존표와 같아 유지했다.
각2,000 requests, token 분모6035/6035/12465, missing0. pretrained forward parity 주장은 없다.
Native FT/SPHERE/MEMIT-FE generation과 모든 factual 수치는 불변이다.

SH2 같은 accepted turn `01a124db-95bd-7231-981d-429bbc98305c`의 실제 agentMessage에서
SH1 최종 인계 수신과62581/82/83 dedup 채택을 확인했다. 재전달·중복제출 없음.
SH1 local 유효미등록CP0이며 remote historical3개는 SH2의 현물/원identity/loader 검산 후보일 뿐이다.
SH2 새평가 job등록 및 최종 결과검산 보고는 별도 미수신 상태로 남긴다.

[Owner 보고](../../servers/server1/baseline-refresh-s2-flucon-20261010/report-ko.md) ·
[정확 rows/원자료 provenance](../../../audits/servers/server1/baseline-refresh-s2-flucon-20261010/table-rows.json).
