# GH → SH3: reference set server4 → server3 선택 전송 재확인

Instruction: ODEEDIT-S06-EN-ADAPTIVE-NULLSPACE-B300-SH3-V1
Nonce: ODEEDIT-GH-SH3-REFERENCE-S4-TO-S3-20260920-R1

사용자 추가 지시: “reference set 이 server3에 없다면 server4에서 전달하는 것도 진행바란다”.

현재 진행 중인 EN adaptive-nullspace task의 명시적 추가 승인이다.
기존 transfers/approvals/2026-09-20-en-adaptive-nullspace-server3.md 범위에서
SH3가 sole destination writer/puller로 직접 수행한다. GH의 별도 raw 재검사나 재승인을 기다리지 않는다.

1. S3 기존 R512/Dev128 입력과 generated256 W0 teacher/key/residual의 manifest, model,
   tokenizer, generation 및 valid-position identity를 먼저 확인한다.
2. 정확한 local 구성원은 REUSE; 없거나 불완전한 필요한 구성원만 S4 완료 자산에서 수신한다.
   R512=S64+Reserve320+AdditionalTrain128, Dev128 observer 구분과 원 문서 순서를 유지한다.
   동일한 teacher가 이미 완결되어 있으면 불필요하게 재생성하지 않는다.
3. S4 /data/janghj/ODE-edit/local/en-execution-reuse/20260919-v1/PREP/attempt-v1/ 및
   기존 approval의 완료 manifest가 결속한 공통 입력을 사용한다.
   Native capsule/다른 teacher를 reference 정의가 같다는 증거 없이 대체하지 않는다.
4. 전송 전에 source→destination 정확 file allowlist/size/SHA/용도와 실제 S3
   free/inode/partial-finalize peak/후속 실행 여유를 계산한다.
   S3 최근 M0 free120960335872B는 과거 관측이며 독점 예약이 아니다.
5. destination은 /data/janghj/ODE-edit/local/en-adaptive-nullspace/20260920-v1/inputs/ 아래
   새 versioned 경로. 기존 자료와 S4 원본은 KEEP, 삭제·이동·--delete·mismatch 덮어쓰기0.
   신규 수신은 size/SHA와 manifest의 문서/position coverage를 검산해 create-once finalize.
6. SH4의 도움이 필요하면 동일 reference 전송 범위의 exact 경로/완료 manifest만 요청한다.
   live scientific raw나 무관한 결과를 조사하거나 다른 SH의 task를 재개하지 않는다.
7. reuse/전송 파일·bytes·검산·남은 공간·불일치/누락을 중간보고와 transfer receipt에 남긴다.
   실제 공간이 부족하면 임의 삭제 없이 구체적 부족량·대체안을 보고한다.

cap1/save_checkpoints=false/B300/추가 accuracy·NLL 지표/기존 수치계약은 그대로다.
Reference teacher는 입력 자산이며 edited checkpoint 저장 예외를 뜻하지 않는다.
