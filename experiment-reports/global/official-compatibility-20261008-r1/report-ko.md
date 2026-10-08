# Official 공통 호환성 및 server4 후속 진행

2026-10-09 KST. Parent `USER-OFFICIAL-BASELINES-20261008-R1`.

SH1의 `SH1-GH-OFFICIAL-COMPAT-REVIEW-20261008-R1`과 SH2의 `SH2-GH-OFFICIAL-TRACKING-BINDING-20261009-R1`에 따라 공통 코드를 보완했다. 사용자 추가 지시 “server4도 다른 서버들이 현재 작업중인 것 진행하라고 지시해”는 기존 SH4 배정 Llama3 AlphaEdit/BLUE/SPHERE × CF/zsRE 6행의 구현·기술검증·실행 진행으로 결속한다. 새 arm이나 무관 job 변경은 없다.

SPHERE/BLUE hidden container 및 누적 KV mask, 공식 Llama/Qwen generation cache/position 호환을 수정했다. FE native generator, 입력·loss·hparams·baseline matrix는 유지했다. CAKE generation의 global RNG/top-k5/total100/noEOS/decode 조건을 유지한다. source provenance는 원 upstream SHA를 보존해 `official/SOURCES.json` 148개를 재봉인했다.

W&B는 SH1 기존 logger의 정확한 원본 7모듈을 기반으로 `official/tracking` 공통 배포 한 개를 만들었다. API는 `init`/`log`/`finish` 및 `official_generation_progress`다. `official-baselines-scalar-v1` request-macro 점수와 실제 R/P/N prompt/token 진단을 구분한다. CF W0 모델당1회/W20 chain당1회와 zsRE no-generation을 지원한다. actual job ID/name, 불변 source/config/URL, scalar privacy 및 bounded readback 조건을 유지한다. 기존 helper/frozen jobs는 그대로다.

CPU 검산 104개 PASS(60 generation, 7 hidden/KV, 24 fake SDK, 13 preparation/runtime), independent generation 및 tracking 검토 blocker0. source verifier PASS/external task imports0. tiny 무작위 모델 및 fake SDK 검사이며 실제 pretrained GPU/온라인 인증·전송 PASS가 아니다.

SH1 factual evaluator는 담당자의 별도 구현·게시 source에 결속한다. GH가 duplicate evaluator를 만들지 않는다. 각 담당은 최신 공통 main source를 own runner와 연결하고 실제 parity/resume/source binding 후 승인된 chain을 제출한다. 본 문서 작성만을 실제 direct owner 수락 또는 제출로 표시하지 않으며, 전달 영수증을 별도로 게시한다. 기존 job·raw·checkpoint·dirty root 보존, GH GPU/Slurm/온라인/삭제0.

## 실제 전달과 담당 수락

공통 source는 `b10a87dfdbb2b12d6e0393bf55a391738f5adb54`/official tree `85f2cb7bbdbe8eeed27c4e24b923b45d6615cab6`에 게시했다. SH1·SH2에 active 동일 task steer로 전달했고 직접 owner ACK를 회수했다. SH4는 기존 idle 세션을 대상으로 `01a11c1c-513e-72d2-895a-f94ca37f7809` turn을 실제 시작했고, nonce `USER-GH-SH4-OFFICIAL-BASELINES-CONTINUE-20261009-R1`의 명시 OWNER_ACK를 app recent-turn에서 확인했다. 긴 read/resume 응답 timeout은 turn write 전이었으며 중복 turn을 쓰지 않았다.

SH4 ACK 시점 상태는 구현 중·신규 job 미제출이다. 기존 Llama3 AlphaEdit/BLUE/SPHERE × CF/zsRE 6행만 진행하며 Qwen/OURS 및 과거 취소 이력은 보존한다. 실제 pretrained/GPU qualification·Slurm 제출·완료를 수락으로 대신 주장하지 않는다.

SH1 factual API도 source `596896ff82a0c0aab8f64e4920f6f09d73072c58`/main `55afa07d2555718c4ad8b2e483db8a260aa94e35`에 게시되어 exact bytes로 통합했다. `evaluate`/`evaluate_counterfact`/`evaluate_zsre`/`build_zsre_w0_reference`와 SHA·실제 signature를 SH4의 같은 accepted turn에 추가 steer했고 수락 응답을 확인했다. 따라서 해당 공유 API 미게시 대기는 해제됐다. 최신 통합 verifier는 source148/python199/imports0 PASS다. 자세한 ACK·turn·SHA 영수증은 `messages/head/2026-10-09-official-shared-compatibility-delivery.json`에 있다. 장기 실험·scheduler polling은 하지 않았다.
