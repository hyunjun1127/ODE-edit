# Server2 cap4 적용

직접 USER 지시: “gpu cap 4로 해. 전부 올려”.
root 및 준비 WT의 ignored local cap 해당 server2 행만 3→4로 변경했다.
QoS lab_gpu_s2도 GPU4 상한이며 기타 물리/메모리 제한은 유지한다.

6개 기존 arm 61650–61655와 CPU collector61656은 모두 등록/release된 상태다.
exact owner/server/Command/WorkDir/source/state 확인 후 PENDING FE61654만
afterany:61651 → afterany:61650으로 변경하고 postcheck했다.
FT61650 RUNNING, 나머지 5개와 collector PENDING이다.

FT의 공통 factual W0 준비·검증 선행 조건은 유지한다. FT 종료 뒤
MEMIT61651/AlphaEdit61652/BLUE61653/FE61654 최대4 lane,
SPHERE61655는 AlphaEdit61652 종료 후 그 lane을 사용한다.
실제 실행 시작은 scheduler 물리 자원에 따른다. CPU collector는 기존 전체6 afterany 유지.

기존 source6d35c65d/manifest/lock의 cap3은 역사적 제출 기록으로 보존하며 이 별도
resource override가 현재 scheduling cap4를 명시한다. frozen source/hparams/argv/scientific
config를 hotpatch하지 않았다. 새 제출0/취소0, 타 job 변경0.
FLU/CON 연기 및 W20 checkpoint 후속 평가용 보존은 불변이다.
장기 polling/monitor/retry 없음.
