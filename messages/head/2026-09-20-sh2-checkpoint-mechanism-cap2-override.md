# GH → SH2: GPU 두 slot 활용 정정

Instruction: `ODEEDIT-S06-S2-CHECKPOINT-MECHANISM-AUDIT-20260920-V1`
Nonce: `ODEEDIT-GH-SH2-CHECKPOINT-MECHANISM-CAP2-20260920-R1`

최신 사용자: “설계대로 1개말고 cap 2개 전부 사용하는 걸로해”.

이 지시는 원 설계와 최초 envelope의 task 동시1GPU에 우선한다. 프로젝트 총 cap2 안에서
이번 task도 두 개의 1GPU lane을 사용한다. 독립 history/operator·분석 구간을 분배하여
준비된 독립 작업이 있을 때 두 slot을 채운다. 공통 prerequisite/실제 모델 gate는 생략하지 않으며
의존성 때문에 직렬인 단계는 중복 실행으로 slot을 채우지 않는다.

각 job 1GPU/8CPU/host≤60416MiB, exportNONE/Requeue0/held inspection을 유지한다.
동시 host RAM/disk와 다른 project allocation을 admission에 포함한다. 다른 job 변경0.
공통 immutable 입력 재사용, factor/output single writer, 결정론적 seed/샘플 순서/수치 계약 유지.
새 z fitting/편집 chain/추가 arm/수치 threshold 변경0, checkpoint 저장0.

첫 M0 또는 다음 중간보고에 실제 두-lane 분할 및 의존성·자원 계획을 포함하고 최종 보고까지 계속한다.
GPU2 활용은 승인된 task의 실행 병렬화이며 별도 방법 실험 권한이 아니다.
