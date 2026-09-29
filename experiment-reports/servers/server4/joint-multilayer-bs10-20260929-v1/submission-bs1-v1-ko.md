# 다층 공동 편집 USER BS1 제출 인계

상태: MAIN_GPU_RESOURCE_PENDING_HANDOFF / INITIAL_NOT_OBSERVED.

사용자 변경을 실행 설정에 반영했고 GH에 USER 명령임을 직접 전달하여 ACK를 받았다. 3CP × 3방법의 9경로, 모든 L4–L8 편집은 유지한다. 각 경로는 **BS1 × 100 edit**, **25·50·75·100 step마다 다섯 actual full weight 저장**이다. 총36snapshot/180tensor/39.375GiB payload 계획이다. 기존 단층 task는 STOP 유지.

## 실제 등록

모든9경로는 array **55090_[0–8]%2**, CPU collector **55091 / afterany:55090**로 held 검사 후 release했다. 추가 chat 호출이 후속 경로의 제출조건이 아니다.

|Job|부모 CP|방법|offered edit|
|---|---|---|---:|
|55090_0|B010|JOINT_STEP|100|
|55090_1|B010|NATIVE|100|
|55090_2|B010|JOINT_CUM|100|
|55090_3|B050|JOINT_STEP|100|
|55090_4|B050|NATIVE|100|
|55090_5|B050|JOINT_CUM|100|
|55090_6|B090|JOINT_STEP|100|
|55090_7|B090|NATIVE|100|
|55090_8|B090|JOINT_CUM|100|

각 GPU job1GPU/8CPU/60416MiB/server4/gpu/exportNONE/Requeue0, CPU collector8CPU/24576MiB/GPU0. Project/task cap2, 이 task는 최대2동시. Wall7일은 요청값이며 실측 ETA가 아니다. Admission 시 자기 owner S4 queue0 및 프로젝트 active0을 확인했다. 다른 task/job 변경0.

제출 직후 한정 snapshot **2026-09-29T09:22:26.962454+00:00**: 모든array PENDING(ReqNodeNotAvail, May be reserved for other job), collector PENDING(Dependency). 노드8GPU/할당8GPU. 이것은 이후 현재상태나 scientific completion의 보증이 아니다. 수동 hold/PENDING(None)를 자원부족으로 해석한 것이 아니다. 이 snapshot 이후 job/scheduler/log/result polling·heartbeat·자동recall을 중단했다. 등록 프로그램만 자연 진행하며 결과 회수는 사용자 호출 시 수행한다.

## 검증·미관측

정본13member/fullSHA 및 봉인CSV CRLF 보존, 별도 USER BS1 order/저장 override 검산, CPU29 PASS. 원500 중앞100,원272 panel 유지. 부모3CP는 prior fullSHA/W/M payload receipt + 현재stat 재사용이며 새 payload검사라고 쓰지 않는다. 재전송0. Raw/model/tensor/prompt/stdout Git0.

CPU toy/import/회귀는 actual pinned Llama PASS가 아니다. 대표 joint B1 accept/정상reject→B2연결, native history5, actual hook/materialization/snapshot reconstruction, scientific900attempt/36저장은 모두 **NOT_OBSERVED**다. 새 task의 정상거절도 offered분모에 남긴다. source/config/guard는 원 과학식 유지, 타task numerical waiver 상속0. exact_editor_resume=NOT_AVAILABLE.

Owner audit 및 별도 CPU reducer/fixtures 검산이며 독립 red agent 사용0. Markdown 상대링크/표열/source/memberSHA/raw-free는 검사했으며 실제 Markdown 렌더는 renderer미설치로 NOT_TESTED다.

## 고정 source / local evidence

- execution source: `6b07104954920bb4c5478a88ebed478dcb9527df` / tree `9ea717d43e9b693285ed58b6885b607a7dc71f6d`
- lock SHA256: `4a64f85fffed70a901e22613f8b52a3e9730ccf362238d84246f188296032f95`
- config SHA256: `d903f36882086482ddceee410e00eb6f4ec98f935a37292b5ee3592ee115ec47`
- archive SHA256: `8561bcb00ed10bf650afbbcfab503e2d7e5bf5a26d9e8a5f0d100c845d3a8a85`
- submission SHA256: `defe2e3c156a208a48d69c9b900394b022cc52dc26048c4c983a29c6a48a71ab`
- release SHA256: `2812e1c334e3005423d93fad0c7e5c0095f596d302c634fad181067898de0ce6`
- local attempt: `/data/janghj/ODE-edit/local/joint-multilayer-bs10/20260929-v1/attempt-bs1-v1`

[제출 감사/receipt index](../../../../audits/servers/server4/joint-multilayer-bs10-20260929-v1/submission-bs1-v1.json), [CPU·자원 preflight](user-bs1-freeze-ko.md). 실행source와 이번 publication source를 구분한다. 원input/source/raw/다른task는 보존했고 NO_BROADCAST_NOT_REQUIRED: same-host CP/model/P 재사용, 원격 대형전송0.

종료: monitoring_active=false, automatic_resume=false. 다음 사용자 recall 전 결과회수/자동수리/추가실험0.
