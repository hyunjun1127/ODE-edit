# Server2 공통 GPU cap2 적용 기록

Nonce: `USER-GH-ALL-SH-GPU-CAP2-20261007-SERVER2`. 사용자 “각 서버의 gpu cap은 2이다.”를 수락했다. 이 기록은 자원 정책 적용이며 새 실험·과거 STOP 재개·결과 모니터링이 아니다.

## 적용과 한정 관측

실제 hostname은 server2, 원 CWD는 /mnt/raid5/janghj/ODE-edit, session은 01a0493a-074c-7f91-9a13-769116326fef, origin은 hyunjun1127/ODE-edit이다. 원 dirty 상태를 보존하고 전용 clean non-main worktree에서 처리했다. Authority 6c8d36dec6e1ce4a7e92831b2619807f5c9b38f1의 envelope·canonical policy SHA를 전달값과 대조했고 PROTOCOL 전체를 읽었다. 동일 nonce의 기존 own ACK는 발견하지 않았다.

원 root local cap은 이미 2였고 bytes를 변경하지 않았다. 현재 W0/native admission 및 이번 policy 전용 WT에는 동일한 ignored cap row를 추가했다. node=server2, memory ceiling=60416MiB, job patterns와 다른 서버 행은 변경하지 않았다. task의 더 엄격한 1GPU 제한은 그대로이며 native 두 경로의 combined task cap은 2다.

2026-10-07 18:38 KST 부근의 한정 scheduler 자원 snapshot:

| Job | 소유 경로 | 상태 | 할당 GPU | CPU / host MiB | 대기 의존성 |
| --- | --- | --- | ---: | --- | --- |
| 60656 | BASE_MEMIT | RUNNING | 1 | 6 / 59392 | 없음; 원 afterany:60652 |
| 60657 | BASE_ALPHAEDIT | RUNNING | 1 | 6 / 59392 | 없음; 원 afterany:60652 |
| 60658 | CPU collector | PENDING | 0 | 6 / 24576 | afterany:60656:60657 |

실제 owner janghj(1025), node 및 project Command/WorkDir를 봉인 submission/lock과 연결했다. **Server2 합계 2GPU, GPU pending 0, 현재 DAG 최대 GPU 폭 2**다. 동일 owner 전체 queue의 나머지 pending 9건은 요청 node=server4로 확인하여 제외했고 변경하지 않았다. parent/step 중복 합산은 하지 않았다. 추가 즉시 GPU headroom은 이 snapshot에서 0이며 pending 일정 조정은 필요 없었다. 이 값은 이후 자동 갱신되는 현재 상태 주장이 아니다.

## 패턴 누락 및 검산

기존 local patterns `odeedit_*,odealloc_*`는 위 native 이름 두 개와 일치하지 않는다. 따라서 helper의 이름 패턴만으로 0GPU라고 계산하지 않고 owner/source/submission 증거로 2GPU를 합산했다. shared helper/patterns는 수정하지 않았다. 다음 admission에서도 실제 합산 확인이 필요하며 이번 snapshot은 미래 admission 허가가 아니다.

기존 CPU mock test source를 그대로 재사용 실행하여 19 checks PASS를 확인했다. 실제 scheduler mutation 없는 mock이며 모델/GPU 계산 또는 과학 결과 PASS가 아니다. 검토는 owner scoped audit이고 explorer는 source/evidence 읽기만 수행했다. 별도 independent red audit는 수행하지 않았다.

## provenance 및 보존

실행 source 56d3a445553b60bf1a5e33f0e820e0699364ba0b, config SHA a9ff4078e0839adcee07ce3ece0c26ed1d6f0d9ff71fadfd69dc203f1866decf, lock SHA 9c5bc040deb42f0e6dc66136edf49c5ab37e576cc25ca4feb71e2a36da7d8839는 기존 값 그대로다. 과학 source/config/W&B identity/raw와 RUNNING job을 수정·취소·재시작하지 않았다. 새 job/model/GPU/W&B run/checkpoint는 0이다.

상세 경로·SHA·scheduler 요약은 `audits/servers/server2/gpu-cap2-20261007/application.json`에 기록했다. 새 cap row 세 개는 ignored local 설정이며 Git에는 소형 보고/receipt만 게시한다. `NO_BROADCAST_NOT_REQUIRED`; 원 자료 KEEP. 정책 작업 종료 후 monitoring_active=false, automatic_resume=false, TASK_COMPLETE_STOP.

게시 직전 최신 main c1fc2879f5cf71d00935c5b59e7d87d8678f5389을 충돌 없이 보존·병합했다. 그 main의 별도 USER 정정은 server3=1/server4=3이며 **server2=2는 그대로**다. 최초 all-cap2 검산 receipt는 당시 authority/SHA에 결속된 역사 기록으로 보존하고, 최신 정본의 mock 19 checks도 PASS를 확인했다. 다른 서버 정책·source를 되돌리지 않았다. 새 통합 근거는 같은 audit 폴더의 `integration.json`에 기록한다.
