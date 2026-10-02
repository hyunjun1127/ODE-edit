# JLZ v7 착수 보고

상태: SUBMITTED_RELEASED. Pilot 실행 중이며 main 초기 연결/500 완료는 아직 관측하지 않았다.

- Nonce: `ODEEDIT-USER-GH-SH4-JLZ-V7-CAUSAL-500-20261002-R1`
- 정본15개 SHA/size 및 전체 문서·CSV 결속 완료.
- CPU 설계107+4 및 생산/adapter/pipeline/memory27검사 PASS. 실제 모델 PASS 아님.
- A/B 각각 cold BS100×5, W5 평가 후 종료. noB6/noCP/신규baseline0.
- v6 중단 유지, 기존 자료 보존. 13:55:49Z owned queue0, 취소0.
- 입력·W0재사용 및 상세 한계: [preflight](../../../../audits/servers/server4/jlz-causal-writer-v7-bs100x5-20261002-v1/preflight-r1.md).

## 실제 등록

| 구간 | Job | dependency |
|---|---:|---|
| prep A: BS2 pilot/qualification | 57511 | 없음 |
| prep B: BS2 pilot/B100 5후보 | 57512 | 없음 |
| cold main A, 500 | 57513 | afterok:57511:57512 |
| cold main B, 500 | 57514 | afterok:57511:57512 |
| CPU collector | 57515 | afterany:57511:57512:57513:57514 |

실행 source `fe6758512eba37595e99244d150d115268553f01`, lock `97a03ae24674f0e7fbe115e96f0b0d930166aaa4fa80a5af29c30206b1e9f09e`.
5개 모두 held 검산 후 정상 release. 최대 GPU 동시2, 각1GPU/8CPU/60416MiB, collector GPU0/8CPU/24576MiB. 다른 job 변경0.
소형 [제출 receipt](../../../../audits/servers/server4/jlz-causal-writer-v7-bs100x5-20261002-v1/submission-r1.json)에 archive/config/launcher 결속.
원 raw와 로그는 `/data/janghj/ODE-edit/local/jlz-causal-writer-v7/20261002-v1/attempt-r1/`에 보존. Git에는 원 raw/텐서/prompt/전체 로그를 넣지 않았다.

실제 초기 관측은 별도 receipt로 추가한다. NO_BROADCAST_NOT_REQUIRED.
