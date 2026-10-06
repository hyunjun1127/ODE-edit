# Llama/GPT-J PRICE 제출 및 W&B 정합

현재 단계는 **11개 새 GPU 작업과 3개 CPU collector 등록·held 검사·release 완료**다. 초기 한정 snapshot에서 새 작업은 모두 PENDING, 60001은 RUNNING이었다. 이는 GPU 기술 검증·W20 완료를 뜻하지 않는다. 이후 상태는 추가 polling하지 않는다.

60001은 사용자 명시 KEEP으로 source/config/W&B/RAM 진행을 보존했다. 기존 60002/60003/60011/60012/60013 및 collector 60007/60017만 정확 owner/source/argv와 elapsed=0, StartTime 없음, 할당 없음인 PENDING으로 확인하여 downstream 먼저 hold한 뒤 collector→역위상 순으로 취소했다. 전부 CANCELLED/elapsed0/할당 없음 accounting을 기록했다. Slurm의 terminal Start=None 및 취소 timestamp를 실제 실행 시작으로 오인하지 않는다. 취소 이력과 원 bytes는 보존했다.

| 모델 | writer | CAP075 | CAP100 | FREE100 |
|---|---|---:|---:|---:|
| Llama3 | MEMIT | **60001 KEEP** | 60102 | 60103 |
| Llama3 | AlphaEdit | 60105 | 60106 | 60107 |
| GPT-J | MEMIT | 60112 | 60113 | 60114 |
| GPT-J | AlphaEdit | 60115 | 60116 | 60117 |

CPU collector는 Llama MEMIT 60104(afterany 60001/60102/60103), Llama Alpha 60108(afterany 60105/60106/60107), GPT-J 60118(afterany 여섯 새 arm)다. [전체 cell ledger](cell-ledger.csv)는 old/new/KEEP, dependency, config SHA, W&B 가용성을 구분한다. 서버1 GPT2-XL 여섯 개는 별도 담당이며 이 보고서의 새 11개에 포함하지 않는다.

새 Llama lane은 60001 → MEMIT CAP100 → FREE100 → Alpha CAP075 → CAP100 → FREE100의 afterany 자원 순서다. GPT-J는 MEMIT/Alpha × CAP075/CAP100/FREE100 여섯 cold first2000 BS100×20이다. 전체 서버 cap은 최신 사용자 3과 현재 더 엄격한 정책의 최솟값이며, Qwen 재개·baseline/pilot/추가 fit은 없다. 60001은 취소·hold·hotpatch·재실행·중복 제출하지 않는다.

W&B는 current/pre와 current/post를 항상 R100/P200/N1000으로 유지하고, all_seen/post는 실제 W5/10/15/20만 기록한다. 기존 B4/B5/W0 원 row로 재집계한 CPU 검산과 fake SDK 검사에서 pct/nats, N desired=true, harmonic, edits/state/fit 단조 축, 실제 job 식별자의 전달, immutable identity 및 privacy가 통과했다. fake SDK는 실제 온라인/GPU PASS가 아니다. 새 job의 startup/finish에서 bounded readback을 별도로 남긴다. 기존 60001에 새 W&B writer나 backfill을 붙이지 않는다.

원 Llama config/input/runtime/model/C0/projector 및 정확 W0 재사용 자격을 유지한다. GPT-J는 기존 준비 자산과 native adapter를 결속하며 첫 승인 main에서 native context 입력을 한 번 준비한다. 실행 중 데이터를 저장한 checkpoint에서 재개하지 않으며 모든 새 arm은 cold W0/H0다.

Collector는 새 MEMIT 두 cell과 보존한 60001을 **각자의 실행 source/config/lock**으로 검산한다. Alpha는 새 Llama 세 cell만 집계한다. 제외 Qwen을 실패로 세지 않는다. 기존 nested native NLL의 context→owner 집계 결함을 CPU reducer에서 수정했다. 대응하는 새 MEMIT/Alpha 결과 비교는 정확 endpoint 검산 전 PENDING_COMPARISON이다.

소스 검토는 owner와 별도 위임 worker/explorer의 한정 검토이며 독립적인 전체 과학 감사나 새 실제 모델 검증을 주장하지 않는다. 모든 원 archive/raw 및 실패·취소 이력은 KEEP이다. Git에는 소형 source/report/manifest만 게시한다. `NO_BROADCAST_NOT_REQUIRED`: 대형 raw와 W&B credential/spool을 옮길 필요가 없다. 일반 Git access helper는 새 `runs/price-model-runs-tracking/server4/submission.json` prefix를 지원하지 않아 **NOT_PASS(exit7)**였다. 현재 USER envelope가 그 정확 prefix를 명시 승인하므로 [범위 예외](../../../../audits/servers/server4/price-model-runs-tracking/publication-scope-exception.json)를 기록하고 소형 receipt만 게시한다. shared helper를 편집하거나 PASS로 위장하지 않았다.

실행 source는 `5226337121fd2c90c595f26297c9927400f8f0af`; 60001은 원 `2440e548be39e55a99747d7847d21a88df419b93`다. 이 보고서·소형 게시 도구는 이후 분석 source이며 frozen 실행 bytes를 수정하지 않았다. [제출 및 SHA receipt](../../../../audits/servers/server4/price-model-runs-tracking/submission-receipt.json)와 [artifact manifest](../../../../audits/servers/server4/price-model-runs-tracking/artifact-manifest.json)에 config/lock/archive/source와 held 검사 경로를 기록했다.

fresh cap은 min(user3, local3, tracked3)=3이었다. 기존 Llama lane 폭1 + GPT-J 폭2가 합산 최대3이라는 dependency 검산을 남겼다. GPU job은 각 1GPU/8CPU/59392MiB/48h, collector 0GPU/8CPU/24576MiB/4h, export NONE/requeue0다. 48h는 ETA가 아니다. 모델별 memory estimate에는 SDK sidecar를 포함한다. 신규 actual peak/속도는 아직 미관측이다. 출력 여유 계획은 36,590,583,808B, 최종 config 시 free 41,496,006,656B였다. 이는 독점 예약/미래 보장이 아니며 각 batch 전 storage reserve guard가 유지된다. 삭제는 하지 않았다.

실제 새 W&B UUID/URL/online readback과 B1/W20은 **NOT_OBSERVED**다. 60001의 기존 run ID `c5b6f44631f44f22`는 변경하지 않았다. sealed runner/collector만 자연 진행하고 agent의 monitoring_active=false/automatic_resume=false/automatic_retry=false로 인계한다. 장기 GPU 대기·recurring monitoring·heartbeat·자동 Slurm retry는 없다.
