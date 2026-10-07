# Llama3 native baseline: fluency / consistency 2k

## 현재 단계

`RESOURCE_BLOCKED_SLURM_CONTROLLER_IO_NOT_SUBMITTED`. 수락 nonce는 `USER-GH-ALL-SH-BASELINE-FLUENCY-CONSISTENCY-RERUN-20261007-R1`이다. 소스 구현/CPU61검사/실행 archive 봉인은 완료했지만 **첫 held 제출이 Slurm controller I/O 오류로 거부되어 실제 job IDs는 `[]`**이다. actual GPU qualification/B1/W20/W&B remote delivery 모두 `NOT_OBSERVED`. 이는 Slurm PENDING으로 등록된 상태가 아니다.

정확 소유 baseline 여섯 종의 활성 실행은 초기 한정 inventory에서 없었다. 취소 IDs는 `[]` / 취소 0. PRICE OURS, 명시 KEEP 60001, 기존 GPT-J 및 W0/자산 준비는 변경하지 않았다.

## 실제 제출 시도와 blocker

실행 source `e3019677e5b17edf98401e381972c272a711ecc7`, tree `54cdbb68c41a6fb7502e86d9375501db0c58b52c`, config SHA `bd07c80a94ae978c3ab071d0ed5f54d0a5a945f9350ad6141ff4e1173e7fb676`, archive SHA `f0339e37f0dfa5672347fd1cd286fe69787d25aa7add1fd7c0920ec7ddaa8034`를 frozen `attempt-r1`에 보존했다. runtime source와 후속 보고서 publication commit은 구별한다.

첫 MEMIT `sbatch --hold` 1회가 `Batch job submission failed: I/O error writing script/environment to file`로 실패했다. 성공 등록/held inspection/release 모두 0, 나머지 다섯 arm/collector의 제출 호출도 0이다. 실패 뒤 한정 exact own queue에 신규 task는 없었고 기존 OURS 세 allocation은 그대로였다. resource barrier는 기존 admitted GPU IDs `60621,60620,60618,60107,60106,60619,60617,60105`의 afterany로 계획했다. 이 ID들에 취소/hold/dependency 수정은 하지 않았다.

읽기전용 확인: controller `devbox` UP, `StateSaveLocation=/var/spool/slurmctld`. 승인된 서버 연결의 `df`에서 controller state/log filesystem `/dev/nvme0n1p2` available 0 / use100%, inode use5%; server4 `/data`는 약47GiB, root/tmp는 약130GiB available였다. controller 저장공간 부족은 관측 사실이며 제출 오류와 부합하는 원인 **추론**이다. 정확 errno와 controller log는 읽기 권한이 없어 `NOT_VERIFIED`. Slurm의 해당 오류는 controller에서 job script/environment 파일 저장이 실패했을 때 반환되는 코드이므로, server4 입력 파일 오류나 과학 실패로 단정하지 않는다. [SchedMD 원 소스](https://raw.githubusercontent.com/SchedMD/slurm/slurm-24.05/src/slurmctld/job_mgr.c)

| 방법/role | 실제 새 ID | 단계 |
| --- | --- | --- |
| MEMIT | 없음 | 첫 held-submit 거부 |
| PRUNE | 없음 | NOT_SUBMITTED |
| RECT | 없음 | NOT_SUBMITTED |
| AlphaEdit | 없음 | NOT_SUBMITTED |
| AlphaEdit-BLUE | 없음 | NOT_SUBMITTED |
| CAKE | 없음 | NOT_SUBMITTED |
| CPU collector | 없음 | NOT_SUBMITTED |

관리 디스크 삭제/이동/권한 변경/daemon restart는 현재 승인 scope 밖이라 0이다. controller 가용공간 복구 필요를 GH에 직접 전달했고, GH는 exact accepted turn `01a11666-fe5c-7d82-9eea-fe76c57b6d28`에서 장애 보고 수신을 회신했다. 이후 GH 작업 완료는 기다리지 않는다. 자동 retry/반복 sbatch/agent polling 없이 blocked 인계한다. 옛 source/raw/frozen 실패 attempt는 KEEP한다.

## 실험과 원 구현

Llama3-8B-Instruct revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`, 동일 first2000 occurrence / BS100×20 / seed 20261002. 각 방법은 독립 cold W0 / native H에서 시작한다. PRICE planner가 아니라 각 원 native editor와 hparams를 task-private Python/YAML source 사본으로 결속한다. 원 EasyEdit/CAKE/BLUE source, 기존 실행 archive/raw는 불변이다.

| 방법 | native 편집 층 | native H append 예상/20batch | 출처 |
| --- | --- | ---: | --- |
| MEMIT | L4–L8 | 0 | EasyEdit native |
| PRUNE | L4–L8 | 0 | BLUE native MEMIT + 원 PRUNE terminal compression |
| RECT | L4–L8 | 0 | BLUE native RECT |
| AlphaEdit | L4–L8 | 100 | EasyEdit native AlphaEdit |
| AlphaEdit-BLUE | L4, L8 | 40 | BLUE native two-site editor |
| CAKE | L4–L8 | 100 | CAKE native editor |

PRUNE의 최종 compression은 원 수식의 `coldW0 + compress(denseW20−coldW0)`를 한 번 적용한 실제 W20를 평가한다. BLUE의 원 two-site compute_z 호출 수는 다른 방법과 다르며 비용을 동일 fit 수로 숨기지 않는다. 물리 site별 H·native solve·compute_z·forward/update를 별도로 계상한다. 매 batch 성공 시 RAM W/H를 유지하고, 기술/IO 실패 시 해당 entry로 복원한다. 실패 파일과 완료 prefix는 구별하며 noCP / exact resume `NOT_AVAILABLE`이다.

## SH1 generation source와 평가

공통 evaluator/reference sole owner는 SH1이며, 재사용 source는 `83535c6a47c552cc4e5c6385f3a587d752820150` (게시 main `2828ab0938184ae863831c6e5e93fcf7a021eca4`)이다. SH4는 wrapper/producer/collector만 소유한다. profile은 `cf-cake-prompt-inclusive-total100-eos-corrected-v1`, schema는 `counterfact-cake-generation-metrics-v1`, 평가 seed는 20261007이다.

원 CounterFact generation prompts를 unpadded full-prefix/no-cache로 관측한다. top-k 5, temperature 1, top-p 1, prompt+continuation 총 100 token, native EOS를 유지한다. 프롬프트가 cap 이상이면 입력을 자르지 않고 typed zero-continuation 사유를 기록한다. 전용 RNG로 관측 후 Python/NumPy/Torch/CUDA RNG와 hooks/context/native 상태 비변이를 검산한다. 신규 text/token/raw는 ignored local 전용이며 Git/W&B에 보내지 않는다.

Fluency는 NLTK word-token H2/3 + 2H3/3, 단위 bits; consistency는 고정 TF-IDF reference cosine이다. full decoded prompt+continuation, NFKD/정본 newline 정규화를 적용한다. case 안 text 평균 후 원 occurrence macro 평균을 사용하며, missing을 0으로 채우지 않는다. cosine의 native FP64 4-ULP endpoint 정책은 SH1과 producer/collector에서 같고 clamp는 없다.

참조 identity SHA는 `75e595c7f26ec334830e9bb9ca6028098c19ea84a9509a5713985847683f8ea6`. snippets/idf/vocab 및 NLTK 영어 자원을 원본 KEEP으로 정확 1회 task-private 수신/해시 결속했다. NLTK PY3 locator에 필요한 public parent 파일 누락은 작은 정확 경로 보완으로 해결했다. tokenizer upgrade/regex fallback이나 과학환경 pin 변경은 없다. 이 CPU reference 준비는 실제 Llama generation PASS가 아니다.

R/P/N W0 first2000은 동일 cold weight/runtime/token/evaluator SHA가 확인된 기존 scalar raw를 재사용한다. 기존 PRICE zero-H와 native empty/two/five-site H는 별도 기록하며 editor resume로 해석하지 않는다. generation W0는 새 profile에서 아직 관측하지 않았으므로 MEMIT 첫 job에서 한 번 관측하고, 다른 방법은 정확 identity READY의 읽기전용 subset을 사용한다.

각 batch current pre/post R100/P200/N1000 및 generation current100, W5/10/15/20 실제 all-seen과 fixed first500/cohort를 관측한다. W20 R2000/P4000/N20000. current와 all-seen을 섞지 않는다. RS/PS/NS/harmonic과 TF strict/token micro/prompt macro, N desired=true/NLL margin을 기존 evaluator 의미 그대로 기록한다. 성적은 학습/진행 gate가 아니다.

## 검토 수준과 자원

최종 CPU software/native import/producer/controller/observer/collector/helper-axis·privacy 검사 61개가 PASS했고 최종 소스 SHA receipt에 결속했다. toy scientific fit, GPU/model load, 별도 pilot/새 평가/online smoke는 0이다. owner 검토와 별도 agent의 한정 정적 검토를 구별했다. 독립 reviewer는 commit/rollback, W0 evaluator 결속, 제출 receipt/저장공간 경로를 검토했으며 실제 모델 replay나 전체 independent scientific PASS를 주장하지 않는다.

검토로 발견된 logical ledger rollback, W0 fresh evaluator 결속, immutable held/final receipt 구분, W20 batch reserve 및 commit-file 검증 위치를 최소 수리했다. collector의 cosine 정책은 SH1 고정 정책과 일치시켰다. 기존 native 수식/허용오차/성적 기준은 변경하지 않았다.

server4 combined project cap 3, task cap 3. GPU job별 1GPU/8CPU/59392MiB, hard60416MiB, `lab_gpu_s4`, export NONE, Requeue 0. 48h는 요청 ceiling이지 실측 ETA가 아니다. GPU0 collector 8CPU/24576MiB/4h. SDK sidecar와 CPU native H/projector 포함 모델별 host/VRAM 계획을 세웠지만 실제 peak는 `NOT_OBSERVED`이다.

입력/tokenizer로 산출한 6-run raw upper + reserve는 24,259,973,120 bytes (약 22.59 GiB), 준비 시 free 50,377,134,080 bytes. serializer 상한은 prompt 32,096B/case 337,344B, 보수적 56,000 case. 다음 batch guard는 최대 2,100 case + error reserve = 1,245,293,312B이다. 신규 durable W/H/z/optimizer/payload 저장 0; 단일 native event stream + hash summary. 부족하면 typed storage block이며 무관 파일 삭제/평가 축소/자동 retry는 하지 않는다.

새 DAG는 MEMIT에서 공통 generation W0를 준비한 뒤 최대 3개 native lane, 후속 BLUE/CAKE 및 afterany GPU0 collector로 끝난다. 현재 OURS 할당/이미 admitted pending 전체 frontier를 먼저 기다리는 보수적 resource dependency를 결속한다. generic job-prefix helper가 jlz를 누락한 수치는 `NOT_PASS_PROJECT_PREFIX`로 기록하고 정확 owner/argv/GPU/DAG metadata를 사용한다. 기존 jobs 수정/선점/취소 0.

## 추적·산출물·종료 경계

W&B `wkdguswns2256` / `layer allocation`: 새 실제 startup에서 online init, 실제 job 번호/name/config, source/config/model/method/profile/immutable run ID를 기록한다. current/pre,current/post,all_seen/post,W0_first2000,w0 subsets의 scalar만 허용하고 edits/state axes와 fit/global_candidate를 분리한다. SDK 접수는 remote ACK가 아니다. 기존 startup/finish의 bounded readback 외 새 monitor는 없다.

현재 원자료/참조는 local KEEP; 소형 source/report/SHA만 Git 게시하므로 `NO_BROADCAST_NOT_REQUIRED` (대형 생성 raw/W&B spool/credential은 전송 금지). 자동 생성될 CPU collector 결과는 `local/llama3-baselines-fluency-consistency-2k/attempt-r1/collector/`에 report/compact CSV/manifest로 남는다. 파일명이나 Slurm COMPLETED만으로 W20를 추정하지 않고 exact rows/20commit/19join/각 native H 수를 reducer가 검산한다. 등록 뒤 단 1회 initial resource snapshot을 인계하고 agent monitoring/automatic resume/retry를 끈다. 봉인 runner/collector는 자연 진행한다.

일반 agent access helper는 `runs/llama3-baselines-fluency-consistency-2k/submission.json` 패턴을 지원하지 않아 그 검사만 `NOT_PASS_SCOPE_PATTERN`이다. 현재 envelope의 exact 허용 prefix를 게시 예외 근거로 기록했다. 다른 staged 파일은 ownscope이고 공용 helper는 수정하지 않았다. 현재는 실제 runner 미등록이므로 자연 진행 중이라는 주장은 없다.

Codex app-server 경로 조정은 OpenAI Docs 및 로컬 client를 참고한 전달 경로 보완이며 과학 method 변경이 아니다.
