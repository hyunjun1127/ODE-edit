# SH4 CPU postrun 검증

범위: ODEEDIT-S06-LOW-COST-WRITE-DONOR-CORE-DETAILED-REVIEW-SH4-V1, job46451만.

- Source: 실행7ece056c33fbb4246245c15f5f7c2a678315c05c 그대로 own-scope 게시. Runtime/fitting/evaluation 및 b51 reference bytes 변경0.
- 신규 결과34 members/14,853,827,915 bytes 전체 SHA/size/stable-stat PASS. 7 saved-state snapshots의 CPU weights_only/mmap FP32/finite/tensor/state 결속 PASS.
- 독립 원시 NLL reducer: 8states×2964promptpairs=23712. 48 state/panel/category checks, duplicate/nonfinite/imputation0.
- Red 별도 reducer: 6core×6groups 및 aggregate96rows/paired252rows 수치 검산 PASS.
- Current100R/200P/1000N, Historical128R/256P/1280N; historical124active+4superseded.
- Wiki128/24999 next-token positions, MMLUdev32 unique max alternative correct counts/invalid0.
- Actual stdout+source: 400request-z/3376loss evaluations/2976Adam updates. Native4fits/4solves; fitting history0, endpoint append8.
- M8 공통entry50×100 재인코딩1회. Audit128/N1280와 MMLU68는 identity만 검사, 성능 미실행. 선택/suffix0.
- 신규 focused CPU tests12 PASS(9 reducer+3 package/plot). 기존preGPU12 CPU검사는 별도 재사용으로 계수한다.
- Python Agg PNG5개 동일명령2회 byte-identical, panel ordering/no-weight-reference/title/output fixture PASS.
- Source syntax compile 및 diff-whitespace 검사 PASS. Git에는 code/aggregate/PNG/compact metadata만 포함하며 model/checkpoint/prompt/full logs는 local-only다.
- Claim-decision=PENDING_GH_REVIEW; 참고선 초과는 자동 실패가 아니다. GPU/model-level parity 및 continuation replay 미실행, pure writer cost NOT_SEPARATED.

원본 raw 보존, 새 GPU/evaluator/Slurm action0. Broadcast NO_BROADCAST_NOT_REQUIRED. 상세 scoped red/state 감사는 같은 디렉터리의 별도 문서를 참조한다.
