# Owner 사전검산

실행 사양: GH-SH4-HISTORICAL-UPDATE-TIMEAXIS-20260924-V1. 독립 subagent/red 미사용. 직접 source 검토와 CPU 회귀검사이며 GPU PASS가 아니다.

- 정본 envelope, 전문 instruction, README/design/runner-contract/experiment-contract/DAG/schema/source-evidence4개 및 PROTOCOL 정독. CSV 전행 parse·hash·383 task panel/order·156 cell state식·검열 active9966/9784를 검사했다. sealed build_plan/validate_plan을 실행해 원본을 덮지 않았다.
- Whole U 다섯 weight를 층별 FP64 계산하고 한 번 FP32로 변환한다. 모든 endpoint/단일·이중 제거에 full forward. 과거 E3 patch·activation cache·writer import를 사용하지 않는다.
- CPU12 PASS: 실제 모델 검증 대체 아님. evaluator 원본 bytes dynamic load, tokenizer target 정규화/BOS·UNK 처리/수동 left padding/MB16 유지. Scientific tolerance는 NLL2.5e-4/margin5e-4/분해1e-10 불변.
- 원본 CP24 약77.5GB 새 fullSHA 및 tensor120 shape/hash/finite 검사 완료. 공통 receipt를 worker가 재사용하며 반복 원파일 해시/원격전송0. Model4shard 새 fullSHA와 W0 FP32 tensor hash 결속.
- CP history는 mmap payload에 있으나 계산·복원 대상이 아니다. 필요한 selected weight mapping만 참조하고 한 층 FP64 scratch를 해제한다. CPU model load32GiB와 scratch/객체를 포함한 사전 peak 추정44GiB, 요청59GiB. 실제 GPU peak/worker host peak는 아직 NOT_MEASURED.
- 새 score 및 receipt 저장 reserve20GiB, 현 free 약234GiB. 신규 full weight/delta/resume checkpoint 미저장. 기존 inputs/source/raw 삭제·변경0.
- 두 family matching atomic PASS join을 내부에서 검사한다. Peer technical failure/종료를 확인하면 진행하지 않는다. CPU afterany collector는 두 science terminal PASS 없으면 TECHNICAL_BLOCKED를 기록하며 science를 우회하지 않는다.
- JSON Schema 외부 라이브러리 `jsonschema`는 현재 실행 venv 미설치다. source 자체 required-field/finite/identity/uniqueness/schema shape 검사를 사용하며 외부 Draft7 validation PASS로 표시하지 않는다. 공용 환경 설치0.
- Runtime 원본/과학조건/Git raw-free source scope를 구분. 명시 envelope의 runs/status 경로가 generic helper에 거부되면 narrow limitation을 그대로 기록하며 공용 정책 수정0.

계획 wall7일은 partition30일 상한 안의 보수적 scheduler 등록값이며 실측 소요시간이나 사용자 GPUh budget이 아니다. Pilot padded-token/sec 후 예측 범위를 기록한다. 본인 project queue는 사전 snapshot에서 비어 있었고, 타사용자 resource-only node snapshot은 8GPU 중7allocated였다. 제출 직전에 다시 admission한다.
