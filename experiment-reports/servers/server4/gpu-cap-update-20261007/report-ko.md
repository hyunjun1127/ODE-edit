# server4 GPU 합산 cap3 — 최신 사용자 정정 적용

현재 지시는 `USER-GH-SH3-SH4-GPU-CAPS-1-3-20261007-R1-SERVER4`다. 최신 사용자 원문 “server3과 4는 gpu cap 1과 3으로 맞추자.”에 따라 **server4 combined project cap3**을 적용했다. 직전 server4 cap2는 아래 역사 이력일 뿐이며 현재 한계가 아니다. server1/2=2, server3=1은 GH 정본을 확인했지만 다른 서버 local 설정/job은 수정하지 않았다. 새 과학 실행 허가나 STOP 재개가 아니며 task-specific cap1/2는 그대로 더 엄격하게 적용한다.

root `/data/janghj/ODE-edit/servers/local/gpu-caps.tsv`, 현재 GPT-J admission worktree 및 이 정책 전용 worktree의 server4 GPU 열만 3으로 맞췄다. node/server4·memory60416MiB·job patterns·다른 행은 유지했다. 원 root dirty는 건드리지 않았으며 현재 GPT-J/정책 WT는 official latest main 정책에 결속했다. frozen 실행 source/config/archive/launcher/raw/W&B identity는 바꾸지 않았다. 현재 canonical SHA는 `d7e10c0a743df6dda934fee16883c769364e6b5b0394c113cd422e79c66c71b3`, envelope SHA는 `84253cae5486034f3a3c412cfe373d691ce74f48ba24a8c76b3514205c791728`다.

## 자원 snapshot과 최소 조정

bounded snapshot에서 server4 own project 실제 할당은 **60105·60616 각 GPU1, 합계2**였다. 9개 GPU parent job의 owner/task/source/submission/launcher를 확인했고 GPU0 collector60108·60622는 GPU 합산에서 제외했다. 다른 node의 60656–60658(server2),60739–60741(devbox)는 scope에서 제외하고 변경하지 않았다. 이름이 `jlz-*`여도 누락하지 않았고 step/parent를 중복계상하지 않았다.

| 자원 lane | 보존 job 순서 | 최신 조건 |
|---|---|---|
| Llama Alpha | 60105 → 60106 → 60107 | 자체 afterany 직렬 |
| GPT-J MEMIT | 60616 → 60617 → 60618 | 자체 afterany 직렬 |
| GPT-J Alpha | 60619 → 60620 → 60621 | 60619는 afterany:60616 |

직전 cap2 지시로 추가했던 60619→60107 대기 edge만 해제했다. 변경 직전에 60619가 exact own PENDING/elapsed0/startUnknown/AllocTRES=null임을 확인하고 `scontrol update JobId=60619 Dependency=afterany:60616`을 수행했다. 변경 후에도 PENDING/할당0이며 owner/Command/WorkDir/요청 GPU·CPU·memory·wall/requeue와 기존 ID가 같았다. **RUNNING60105·60616, 명시 KEEP60001 변경0**, hold/cancel/requeue/hotpatch/newjob0이다. 기존 science/input·W0·NoCP·W&B는 그대로다.

전체 released PENDING+RUNNING DAG의 보수적 최대 antichain 폭은 **3**, GPT-J 자체는 **2**, Llama Alpha는 **1**이다. project3 및 기존 task min을 모두 만족한다. 현재 물리 메모리/Slurm 제한을 우회하지 않으며 슬롯 부족은 정상 PENDING이다. 과학성능은 dependency/gate가 아니다. generic helper의 root job-name patterns는 `jlz-*`를 누락할 수 있고 RUNNING 합산만 하므로 helper ALLOW를 이 증명의 근거로 사용하지 않았다. 또 `squeue -w server4`는 PENDING을 누락해 전체 own queue와 exact ReqNodeList/등록 receipt를 대조했다.

## 검토·보존·종료 경계

기존 pure-CPU `scripts/tests/test_gpu_concurrency_policy.sh`를 그대로 실행해 **19 PASS**(real Slurm calls/model/GPU0)했다. 정상 경계·더 엄격 local·disabled·queue/할당 미확인 오류 처리를 확인한다. 별도 source explorer의 read-only admission 경로 리뷰, owner의 canonical 네 행/중복0·local 열 보존·동일 source/config/launcher 등 19개 immutable 소형 참조 및 DAG 검산을 수행했다. 정책/자원 검토이며 모델/GPU 수치·과학 결과 인증이 아니다. 결과/진행률을 폴링하거나 실험을 추가하지 않았다.

[최신 resource receipt](../../../../audits/servers/server4/gpu-cap-update-20261007/scheduling-receipt.json), [정책·DAG 검산](../../../../audits/servers/server4/gpu-cap-update-20261007/concurrency-proof.json), [현재 상태](../../../../tasks/status/server3-cap1-server4-cap3/server4.json), [직전 cap2 역사](../gpu-cap2-20261007/report-ko.md). 원 raw/아카이브 KEEP, noCP, NO_BROADCAST_NOT_REQUIRED: 같은 host의 ignored cap override와 Git 소형 정책 receipt만으로 충분하며 raw/model/secret/환경/SDK 전송0이다. bounded resource reconciliation 뒤 STOP; recurring monitor/heartbeat/automatic retry/resume 및 장기 완료 wait0이다.
