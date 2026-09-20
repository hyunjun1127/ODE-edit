# 검증 gate 제거 실행·최종 owner audit

Instruction: ODEEDIT-S06-S2-CHECKPOINT-MECHANISM-AUDIT-20260920-V1.
Override: ODEEDIT-GH-SH2-CHECKPOINT-MECHANISM-REMOVE-GATES-20260920-R1.
실행 source3427b49728bd671085fb1a091f295f9d438de007, analysis source1c915647.
최신 main17727706의 GH 전달 상태 변경은 통합 시 그대로 보존한다.

- C00/C01 재실행0, 원 C01 FAILED 및 기존269 GPU-sec 불변. 수치 threshold 확대0.
- C01/pilot numerical PASS 의존·physical/dense parity 차단·추가 evaluator parity forward 제거.
- 입력 identity/shape/dtype/order/finite, 실제 LU 실패, I/O, W0 복원·비선택 parameter pointer/version 및 Slurm admission 유지.
- 새 계산 PASS는 `COMPUTATION_NOT_NUMERICAL_CERTIFICATION`. `numerical_validation=NOT_ESTABLISHED`.
- Job51137/51138/51139 모두 COMPLETED0:0, 신규71+262+174=507 allocated GPU-sec, 누적776.
- 13×512 operator rows, 7×100 modes, 168 controls, 64 selected NS observations. 31cell 중30 계산/기존 구성요소 완료, C01 역사FAILED1. 새 science cell의 C01 자동BLOCKED0.
- GPU cap2의 두 독립 lane을 제출했으나11개 free CPU 때문에 activation은 Resources pending 후 operator 종료 뒤 시작. 다른 job 변경0; 실제 동시2GPU 실행으로 과장하지 않는다.
- 새 selected W/M/RNG/delta checkpoint0, z fitting0, history append0. 기존12CP/source/실패/CPUraw 읽기전용 재사용.
- B1/B2 W0 key 재사용. Geometry512는 전체 고정 packing을 사용하며 예전32개 부분 packing 결과를 그대로512개로 확대하지 않았다.
- CPU routing/small tensor22건(수입 fixture 중복 포함), 변경 activation routing1건, Python AST/diff 검사. 기존114건 검증은 계보이며 실제 numerical PASS로 사용0.
- 최종 report member36개 fullSHA, 기존 scientific CSV exact equality, 6656/700/168/64 rows의 키·분모·산술 및 원 에너지합 일치 확인. 이 CPU 보고 검산은 새 GPU 수치 gate가 아니다.
- CSV 기반4종PNG/PDF 각각2회 byte-exact, 새 operator/activation PNG 시각검토. Imagegen0.
- H1/H2 SUPPORTED, H3/H4 MIXED는 단일 stream/고정key/사후선정panel 범위. H5 후속질문만, scientific_promotion=false.
- Owner audit와 별도 CPU reducer를 사용했다. 독립 red agent/GPU red PASS 주장은 없다.

정본 보고: `experiment-reports/servers/server2/checkpoint-mechanism-audit-2026-09-20-v1/no-gates-resume-r1/report-ko.md`
SHA256: f7b967aa44ab0495bae177ae1901bb19d2eeaea9786d451163616b5576e2ec88.
manifest SHA256: 2fab8d760a9bcb37199e1caea9019376d41a3d61964a74630fc12cd0295a05b9.
원자료: `/mnt/raid5/janghj/ODE-edit/local/checkpoint-mechanism-audit/20260920-v1/resume-no-gates-r1/`.
수집 당시 새 local footprint2,074,913,825bytes(원 assets 별도). NO_BROADCAST_NOT_REQUIRED.
기존 paused task/원 raw/source/environment에 대한 변경·추가 실험은 없다.
