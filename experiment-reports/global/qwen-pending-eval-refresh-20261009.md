# Qwen 미시작 평가 코드 교체: README 통합

수신 nonce: `SH2-GH-QWEN-PENDING-EVAL-REFRESH-RELEASED-20261009-R1`.
근거는 SH2가 게시한 `c95e5d05`의
`audits/servers/server2/qwen-pending-eval-refresh-20261009/submission.json` 및 보고서다.
GH는 추가 scheduler 조회·취소·제출·모델 forward 없이 이 영수증을 대조했다.

2026-10-09 09:14:51 UTC 단발 snapshot: 기존 FT CF61898/zsRE61900 RUNNING KEEP.
미시작 61899 및 61901–61922의 23개 취소 이력은 원 보고서에 보존한다.
새 MEMIT CF/zsRE61954/61956, AlphaEdit61958/61960, BLUE61962/61964,
FE61966/61968, SPHERE61970/61972는 held 검사·release 후 Dependency PENDING이다.
README Qwen 6행/54개 metric 칸을 실제 job name/ID로 갱신했다.
10개 신규 chain의 결과값은 아직 없으며 성능 완료로 표시하지 않았다.
W0 행, 다른 모델 성능/상태, PRICE 행은 변경하지 않았다.

새 과학 source `5503935821b0ececb4aef09a5bccb5308879a6b5`,
등록 control `1d9fa5cf35dfff469ccccdafab03543f9f3b09a5`.
CF 생성/native fit/hparams/checkpoint 정책 유지, zsRE 새 공개-query/Loc 경로 결속.
원 FT 및 W0는 과거 평가 source/provenance를 보존한다.
CPU query 일치는 실제 모델 점수/온라인 W&B 완료를 뜻하지 않는다.

GPT-J eval-only61942–61948의 ID/source는 보존했다. 최신 자원 의존성은
61942 afterany61970, 61943 afterany61972, 61944 afterany61966,
61945 afterany61968이며 나머지 edge는 불변이다.
`audits/global/zsre-2k-reeval-20261009/integration.json`의 최초 등록 dependency는
이 교체 이전 snapshot으로 보존하고, 현재 resource binding은 이번 SH2 영수증을 따른다.
추가 성능 재평가나 기존 checkpoint 변경/이전/삭제는 수행하지 않았다.

[SH2 보고](../servers/server2/qwen-pending-eval-refresh-20261009/report-ko.md)

검산 범위: 영수증의 GPU10개 이름/ID와 CF6·zsRE3 칸 매핑,
FT 두 RUNNING 표시, 취소 ID가 현재 Qwen 표에 없는지,
Qwen W0 및 Qwen 밖 README bytes 보존. GPU/온라인 검증은 미실행.
