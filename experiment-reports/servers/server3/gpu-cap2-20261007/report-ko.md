# SH3 합산 project GPU cap2 적용

**APPLIED_NO_SCHEDULER_CHANGE_REQUIRED**. Nonce `USER-GH-ALL-SH-GPU-CAP2-20261007-SERVER3`를 수락하고 root의 ignored local GPU cap 행을1에서2로 변경했다. 이번 전용 non-main worktree도 cap2로 설정했다. node ubuntu, 메모리121856MiB, 기존 job-name patterns와 다른 행은 보존했다. task별 cap1, 더 엄격한 물리·메모리·Slurm 제한과 기존 STOP은 그대로다.

정본 main `6c8d36dec6e1ce4a7e92831b2619807f5c9b38f1`의 envelope와 canonical policy를 FULL_READ했다. 전달된 두 SHA와 일치하며, 중복 없는4개 서버 행 모두2다. 실제 host ubuntu/owner janghj/session 01a0b9c8-d12c-7133-b336-cc1b5fc2a6b3 및 전용WT boundary PASS. 기존 동일nonce ACK/branch는 없었다.

## 한정 admission 관측

2026-10-07 09:37:04 UTC(18:37:04 KST), 이름 필터 없이 owner janghj의 queue metadata를1회 읽고 실제/요청 node 및 pending dependency를 구분했다. 반환14행은 모두 다른 node로 명시되어 server3 범위에서 제외했다. node 미지정 PENDING도0이다. 다른 서버 source/결과를 추가 조회하거나 변경하지 않았다.

| server3 범위 | 관측 |
|---|---:|
| RUNNING/COMPLETING/CONFIGURING 할당 GPU | 0 |
| released/admitted PENDING GPU jobs | 0 |
| pending array/frontier 최대 동시 GPU | 0 |
| own CPU collector | 0 |
| scheduler 변경/취소/신규 제출 | 0 |

빈 own queue이므로 owner/source를 확인할 개별 job이나 조정할 pending dependency/array throttle이 없었다. 기존 job-name 패턴에 jlz/causal/GPT/FE/W0 이름이 빠질 수 있어 패턴 helper의0만으로 결론내리지 않았다. 전체 owner queue에서 node별로 확인한0이다. 이 snapshot은 슬롯 예약이 아니며 다음 실제 admission에 fresh 확인이 필요하다.

## 적용과 검사

[application.json](../../../../audits/servers/server3/gpu-cap2-20261007/application.json)에 exact config before/after SHA, authority SHA, 한정 queue 증거와 검산을 보존했다. 기존 CPU mock19검사가 PASS했고, 현재 worktree helper에 이번 빈 own-node snapshot을 재현한 CPU 검사는 effective cap2를 반환했다. 이는 실제 submit/새 GPU 허가가 아니다. taskcap1이면 min(2,1)=1을 유지한다.

실제 합산에는 ours/baseline/W0/GPU자산준비/pilot/평가를 모두 포함하고 parent/array/step 중복은 제외한다. Helper의 RUNNING 계수와 pending DAG·array frontier 검사는 별개다. 앞으로 job 이름만으로 프로젝트를 식별하지 않고 owner/task/source/submission receipt를 함께 확인한다. 실제 overcap이면 running을 변경하지 않고 추가 admission을 차단한다.

Root checkout은 오래된 tracked policy와 dirty를 보존한다. root ignored cap만2로 바뀌었으며, 그곳의 오래된 helper/canonical policy가 최신이라고 주장하지 않는다. **신규 source freeze/admission은 최신 canonical cap2가 결속된 전용 worktree에서만 수행한다.** 이번 활성 admission worktree는 `/data/janghj/ODE-edit/local/gpu-cap2-20261007/worktree`다. 정지된 과거 task worktree와 frozen source/config는 갱신하지 않았다.

Owner CPU/document 검산이며 독립 reviewer는 사용하지 않았다. GPU/model/W&B run/과학평가·재개·heartbeat·자동retry는0. 보고 게시 뒤 추가 queue polling 없이 종료한다.
