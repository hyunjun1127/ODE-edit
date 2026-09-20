# EN adaptive-nullspace SH3 선택 수신 승인

Instruction ODEEDIT-S06-EN-ADAPTIVE-NULLSPACE-B300-SH3-V1; 사용자 지정 실험 구현·실행에 필요한 완료 입력만 승인한다.
SH3/session01a0b9c8-d12c-7133-b336-cc1b5fc2a6b3가 sole destination pull/writer.
Source KEEP, 삭제/이동/--delete/기존 mismatch 덮어쓰기0; 다른 사용자/credentials/live scientific raw 접근0.

## 승인 source와 범위

- S2 /mnt/raid5/janghj/ODE-edit Git cab4a59fb4476c133b7a7bb967fa557f028c8e1a의
  project/run_scripts/single_layer_mechanism_first/z_hook.py 및 필요한 tracked import/config/test closure.
  같은 Git source는 main publication으로 수신하는 것을 우선한다. S2 z 실행이력 미확인과 구분.
- S4 /data/janghj/tmp/dnm/hooking.py: exact read-only 참조
  SHA feb3509940a40e8b8027ee7daae4c7486fc43ec80394383627699be4f34f072b.
  import 확인에 꼭 필요한 같은 dnm의 source/config만 bounded metadata→정확 목록에 추가 가능.
- S4 /data/janghj/ODE-edit/local/en-execution-reuse/20260919-v1/PREP/attempt-v1/
  완료 R512+Dev128 generated256 reference teacher/key/residual 및 manifest/source/input closure.
- S4 /data/janghj/ODE-edit/local/single-layer-mechanism-first/20260919-v1/PROGRAM/b1-fd-waiver-r4/
  완료 B1의 native capsule/keys/Pstar/context/geometry/source/입력 manifest 중 본설계에 필요한 것.
  기존 correction endpoint를 새 비교 결과로 대체하지 않는다.
- 위 완료 manifest가 가리키는 S4 /data/janghj/ODE-edit/local/bpcw512/20260918-v2/ 및
  /data/janghj/ODE-edit/local/en-execution-reuse/20260919-v1/ 내부 공통 R512/Dev128
  입력/source·완료 reference capsule 구성원. 모델/관측 정의가 다른 teacher는 대체 금지.
- S3 readiness의 기존 모델/P/C0/context/fixed10k는 local REUSE 우선. 무관한 모델 재전송0.

## 승인 destination과 통제

/data/janghj/ODE-edit/local/en-adaptive-nullspace/20260920-v1/inputs/ 또는 source-imports/ 아래
새 versioned member 경로. 원 root/다른 runtime를 덮지 않는다.
Manifest만 먼저 읽어 각 source→dest/owner/size/SHA/reuse-purpose를 allowlist에 봉인하고,
실제 available disk/inodes/temporary peak/여유 검토 후 누락분만 exactpull.
Reference teacher의 95GiB 수준 과거 상한을 새 실제 최소치로 오기하지 않는다.
동일 local byte는 재수신0; 신규파일은 full size/SHA 확인 후 create-once finalize.
Directory 전체 mirror나 다른 task checkpoint 수집0. 기존 source 측 삭제/이동0.
용량 부족시 임의 정리 없이 exact blocker 보고. 이 범위내 세부목록 확정은 별도승인 조건 아님.
결과 raw는 자동 전서버 배포하지 않는다. NO_BROADCAST_NOT_REQUIRED.
