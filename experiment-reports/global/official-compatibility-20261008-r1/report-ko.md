# Official 공통 호환성 및 server4 후속 진행

2026-10-09 KST. Parent `USER-OFFICIAL-BASELINES-20261008-R1`.

SH1의 `SH1-GH-OFFICIAL-COMPAT-REVIEW-20261008-R1`과 SH2의 `SH2-GH-OFFICIAL-TRACKING-BINDING-20261009-R1`에 따라 공통 코드를 보완했다. 사용자 추가 지시 “server4도 다른 서버들이 현재 작업중인 것 진행하라고 지시해”는 기존 SH4 배정 Llama3 AlphaEdit/BLUE/SPHERE × CF/zsRE 6행의 구현·기술검증·실행 진행으로 결속한다. 새 arm이나 무관 job 변경은 없다.

SPHERE/BLUE hidden container 및 누적 KV mask, 공식 Llama/Qwen generation cache/position 호환을 수정했다. FE native generator, 입력·loss·hparams·baseline matrix는 유지했다. CAKE generation의 global RNG/top-k5/total100/noEOS/decode 조건을 유지한다. source provenance는 원 upstream SHA를 보존해 `official/SOURCES.json` 148개를 재봉인했다.

W&B는 SH1 기존 logger의 정확한 원본 7모듈을 기반으로 `official/tracking` 공통 배포 한 개를 만들었다. API는 `init`/`log`/`finish` 및 `official_generation_progress`다. `official-baselines-scalar-v1` request-macro 점수와 실제 R/P/N prompt/token 진단을 구분한다. CF W0 모델당1회/W20 chain당1회와 zsRE no-generation을 지원한다. actual job ID/name, 불변 source/config/URL, scalar privacy 및 bounded readback 조건을 유지한다. 기존 helper/frozen jobs는 그대로다.

CPU 검산 104개 PASS(60 generation, 7 hidden/KV, 24 fake SDK, 13 preparation/runtime), independent generation 및 tracking 검토 blocker0. source verifier PASS/external task imports0. tiny 무작위 모델 및 fake SDK 검사이며 실제 pretrained GPU/온라인 인증·전송 PASS가 아니다.

SH1 factual evaluator는 담당자의 별도 구현·게시 source에 결속한다. GH가 duplicate evaluator를 만들지 않는다. 각 담당은 최신 공통 main source를 own runner와 연결하고 실제 parity/resume/source binding 후 승인된 chain을 제출한다. 본 문서 작성만을 실제 direct owner 수락 또는 제출로 표시하지 않으며, 전달 영수증을 별도로 게시한다. 기존 job·raw·checkpoint·dirty root 보존, GH GPU/Slurm/온라인/삭제0.
