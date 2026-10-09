# FE author hparams: SH1 실제 제출

부모 승인 `USER-SH-FE-AUTHOR-HPARAMS-2K-CF-ZSRE-20261010-R1`의 Llama CF/zsRE 두 cold chain만 등록했다.
실행 source `9c3fe23282bcaf2e353b1494913a323e2f037d39`, official tree `17467b484b1d558e7b83a015c375aca81d705215`.
이 보고서 게시 commit은 실행 source와 별도다. GH가 README 네 행을 통합하며 SH1은 README를 수정하지 않았다.

| dataset | 실제 job | 초기 상태 | afterany |
|---|---|---|---|
| CF | official-s1-cf-llama3-memit-fe-author-history (62529) | PENDING | 62259, 62260, 62261 |
| zsRE | official-s1-zsre-llama3-memit-fe-author-history (62530) | PENDING | 62529 |

전량 held owner/argv/script/source/config/input/CPU/memory/GPU/QoS/dependency 검사 후 release했다.
각 GPU1/CPU8/98304MiB/48h(ETA 아님). 현재 project cap4, 기존 allocation3 및 DAG 최대폭3.
기존 FluCon 실행을 보존하고 종료 뒤 직렬 진행한다. 취소된62262/62263은 되살리지 않았다.
W&B는 실제 job에서 기존 scalar transport로 시작하며 현재 SDK/remote/GPU 완료를 주장하지 않는다.

실제 registry.hparams overrides에 author profile을 전달한다. Llama clamp.75/steps35와 Qwen clamp1/steps35만 비교 변경이며 기존 MEMIT_FE JSON/기본값은 불변이다.
parser가 추가하는 max_length40/batch_size1도 유지한다. 기존 FP64 system-buffer/CPU rollback/native H once append는 변경하지 않았다.
CF는 factual 및 FLUCON DEFERRED, zsRE는 public-query zsre_paper request-macro loc_ans를 사용한다.
W0 및 매 committed batch latest checkpoint, W20 보존. 새로운 두 chain에 옛 edited W/H를 복원하지 않는다.

author/transport CPU44 PASS, source166 SHA/import0 PASS. Llama 전체zsRE2000 actual tokenizer CPU parity input/target mismatch0.
이는 pretrained forward 또는 resume GPU 동등성 입증이 아니다. 별도 GPU qualification/smoke는 NOT_RUN_USER_DISABLED.

SH2 공통 입력: job62101의 contexts/READY/config-provenance/context-token-ids를 그대로 검산하는 optional native_context API를 게시했다.
전달 canonical SHA `5c01bc1a91c0890af2897f18b7a5f95de011badc1eca5e27f44199acc0cb6515` 및 actual local tokenizer 재토큰화 일치.
생산자 모델/tokenizer leaf SHA와 runtime/generator/module/coldseed/profile 결속을 검산하며 provenance를 새 source로 바꾸지 않는다.
MEMIT/FE native context builder AST 동등 CPU 검산. SH2 자체 fresh asset 일치는 SH2의 별도 책임이며 GPU hardware bitwise를 주장하지 않는다.
SH1 Llama는 이 Qwen context나 다른 SPHERE producer context를 이식하지 않고 기존 FE의 native first-apply context 경로를 유지한다.

실제 config/lock/held/source/admission 상세는 local `official-baselines/server1/fe-author-hparams-2k-20261010/registration-r1/`에 보존한다.
각 W20 checkpoint는 `preparation-r2/runs/{cf,zsre}/checkpoint/`에 생성될 예정이며 아직 생성/완료로 보고하지 않는다.
raw/CP/model 전송 없음: NO_BROADCAST_NOT_REQUIRED. GPU 완료 장기대기/반복monitor/자동재시도 없음.
