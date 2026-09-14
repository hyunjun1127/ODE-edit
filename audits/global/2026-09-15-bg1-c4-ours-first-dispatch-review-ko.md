# GH BG-1 C4 dispatch 사전 점검

- 기준 main c2710f3e8fddf2a8df230f076e4519cc6d0d77e5. 새 GH worktree만 작성, root ddc17858 및 README 등 사용자 변경 보존.
- 사용자 지시문/dispatch·reference·method 계약과 survey 전체 정독. PDF 해석 audit 및 수식/산술 checks 읽음; 이번 턴에서 논문/PDF 원문 재감사나 GPU 검산한 것으로 주장하지 않음.
- 최신 main PROTOCOL의 fixed10k/cap2/explicit mem 정책 차이를 읽고 그대로 적용. 오래된 shared checkout PROTOCOL/서버 기록의 cap4·메모리 수치를 재반입하지 않음.
- 기존 fitter의 source-exact history split, select_projector, materialize_alpha 범위를 읽음. 기존 materialize_alpha는 .5/.25를 받지 않으므로 신규 package adapter 필요; 기존 package 수정은 미승인.
- Native singleton source의 canonical residual/context repeat/RHS solve 순서 확인. cached A의 FP32 차이 및 all-token actual screen 기술 검증은 SH4 소유 미실행.
- full-D64 nonlinear barrier와 microbatch별 barrier 평균은 다른 objective다. 전체 scalar slope를 고정하여 원 weighting의 gradient를 누적하도록 envelope에 명시.
- N4 B1–B10 일부 checkpoint만 있을 가능성은 미해결 실물 의존성. 그 경우 재실행/부분max/임의 b를 금지하고 독립 준비 뒤 CALIBRATION_MISSING/G0_BLOCKED. GH가 자산 존재를 추정하지 않음.
- 신규 scientific BG-1 한 chain과 준비·teacher/calibration/technical job을 구분. cap2를 채우려고 baseline 또는 추가 BG 복제 금지.
- 기본 설계의 7-policy와 후속 S128/Pile/old/2stage/ODE/full10k 내용은 historical/후속 계획이며 최신 dispatch에서 실행 제외.
- G0는 reference/teacher/calibration 및 첫 B100 상태유효성 gate; 정상 zero-write/나쁜 성능은 technical failure 아님.
- G0확정 뒤 SH·worker·감사 포함 WAITING_USER_RESUME, 정상 persistent job은 유지, 자동 polling/분석/main/후속submit 중단.
- 사전 envelope review: scope/resource/data split/authority/stop 조건 기계적 정합 PASS. 실제 source red/GPU gate는 SH4에게 요구하며 여기서 대신 PASS 선언하지 않음.
- 사용자 제공 문서14개는 바이트 동일 사본. CSV CRLF는 보존. 구 검증JSON 내 README 등 historical hashes는 생성 당시 기록이며 현재 전체tree 검증으로 승격하지 않음.
- 원 PDF binary 및 원문 raw/teacher/model/cache는 Git 추가0. Git에는 supplied compact source audits/design/candidate metadata와 GH instruction/handoff만.
