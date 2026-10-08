# 실제 등록·초기 인계 결과

사용자 요청에 따라 기존 baseline `61439→61438→61437→61436`을 후속부터 취소했고 원 source/raw/부분 편집·비용을 보존했다. 새 제출은 MEMIT, stock AlphaEdit, AlphaEdit-BLUE 세 개뿐이다. OURS·PRICE·W0·FE 및 무관한 job은 변경하지 않았다.

| 방법 | 새 job | dependency | 초기 상태 | W&B |
| --- | --- | --- | --- | --- |
| MEMIT | 61519 | 없음 | RUNNING | [run153079dd35b24266](https://wandb.ai/wkdguswns2256/layer%20allocation/runs/153079dd35b24266) |
| AlphaEdit | 61520 | 없음 | RUNNING | [run083dadcb82eb488e](https://wandb.ai/wkdguswns2256/layer%20allocation/runs/083dadcb82eb488e) |
| AlphaEdit-BLUE | 61521 | afterany61519 | PENDING/Dependency | 아직 시작 전 |
| own CPU collector | 61522 | afterany61519:61520:61521 | PENDING/Dependency | GPU0 |

실제 Slurm owner/source/script/fullargv/input/자원/dependency 검사 뒤 전량 release했다. 1GPU/8CPU/65536MiB/48h ceiling, collectorGPU0/8CPU/24576MiB/4h이며 요청시간은 ETA가 아니다. 프로젝트 cap2와 admitted DAG width2를 검산했다. cancelled old ID는 새 resource dependency에 없다.

runtime source `adb244e6f9c86b54f73bd6d8fb833b338f470ded`, 구현 commit `762d62fd`, 최초 main 게시 `469aee286dca5d4497ee9b55467064c77939cd07`을 구별한다. config SHA `5c1f8b84237e684cd1ff1a4f532ca56a9eeb3ad22e6b18d9ca8bbafc727ecb36`, execution lock SHA `54356709bf766e9dc53917c6f47d39b42581fe85012ce0106fd0f579b354bfd0`이다. 큰 lock/config/raw는 ignored local만 보존한다.

MEMIT·AlphaEdit의 remote startup에서 run ID/name, 실제 job_id61519/61520 및 signed step_id-5, source/config/model/새 generation profile/schedule 일치를 확인했다. W0 RPN log는 SDK가 접수했지만 아직 metric-row remote readback PASS를 주장하지 않는다. 두 모델의 FP32/eager/TF32off/autocastoff/독립 cold native history도 runtime receipt에 기록됐다. W20 F/C와 과학 완료·속도는 미관측이다.

F/C는 CAKE의 case별 padded KV batch/noEOS/topk5/prompt-inclusive total100 방식으로 **편집 commit20 뒤 전체 first2000 한 번만** 계산한다. 기존 native 편집/hparams/solver/history와 RPN 중간 일정은 그대로다. generation 오류가 이미 봉인된 native20commit을 rollback하지 않는다. 세부 변경은 subagent 작성 [report-ko.md](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-native-flucon-repair-20261008/experiment-reports/servers/server1/gpt2xl-baselines-native-generation-repair/report-ko.md), 정확 compact receipt는 [execution-receipt.json](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-native-flucon-repair-20261008/audits/servers/server1/gpt2xl-baselines-native-generation-repair/execution-receipt.json)을 따른다.

CPU178 PASS와 독립 source 검토를 완료했다. GPU native generation qualification/PASS·가속률 인증이 아니다. 실제 endpoint의 immutable native execution/RNG/state/work receipt와 CPU collector가 완료 시 검산한다.

server2 담당자는 full ACK 뒤 `61429–61434`의 미시작 PENDING 취소와 RUNNING MEMIT `61428` 보존을 확인했다. 새 구현/API/source/report는 공식 SSH→Unix WebSocket `turn/start`로 전달했고 nonce ACK를 회수했다. server2 신규 제출은 이번 전달에서 지시하지 않았다. source와 report 전달 완료는 server2의 실제 runtime 적용 완료와 구별한다.

NoCP/rawlocal KEEP/secret·raw·token·model upload0. 큰 자료 전송은 NO_BROADCAST_NOT_REQUIRED이며 compact Git 게시만 수행했다. 초기 bounded snapshot 뒤 agent monitoring을 중지한다. sealed20batch runner와 own collector는 자연 진행하고 recurring monitor/heartbeat/automatic retry는 만들지 않는다.
