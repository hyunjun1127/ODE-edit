# Qwen FE current-z L2=100 rescue 소스 인계

동일 USER-FE-QWEN-WORKING-CURRENT-Z-20261012-R1 최신 중단/수리 지시 적용.
기존 4-arm 완주/선택 계약은 새 rescue에서만 SUPERSEDED_USER_EARLY_STOP으로
대체한다. 종료된 L2=10 B9는 W20/수치실패/복구 성공으로 표기하지 않는다.
기존 ac05/10843 frozen 파일, 실행/체크포인트/README 변경 없음.

- 새 소스: official/diagnostics/fe_qwen_current/{rescue.py,rescue_observer.py,test_rescue.py,RESCUE.md}.
- L2=100 한 조건, clamp1 및 저자 native BF16/writer/history/current-entry z 불변.
- 기존 준비/launcher/평가/저장 구현을 명시적 신규 entrypoint의 process-local 계약으로 재사용.
  원 entrypoint의 SPEC/동작은 바꾸지 않는다. 신규 준비 경로만 생성하며 제출 기능 없음.
- current100 commit 뒤 R<80/P<70/N<65이면 QUALITY_EARLY_STOP. 로그/원자료와 마지막
  scheduled CP 보존. 수치실패가 아니며 COMPLETE/선택 허용 없음.
- 실제 W20/2k/전체 current 및 누적 raw/query/commit/CP를 CPU 재검산한 후
  R95/P85/N70 통과 때만 SELECTED. 원본 4-arm 완료 대기는 제거.
- 같은 rescue entrypoint의 final prepare/launch/collector로 CF/zsRE cold 평가 연결.
  본평가를 이용한 재튜닝 없음. 두 최종 W20 전에는 복구 완료 주장 없음.
- K/RHS/raw delta/BF16-cast delta/pre-post W 및 L4/L5 개별 target norm 관측.
  solve 상대잔차는 실제 A/B/X의 고정 RHS 4열만 측정하며 전체 잔차로 주장하지 않는다.
  추가 solve/고유값 스캔/LM forward/전체 H clone 없음. native stdout fit/lookup 유지.

CPU: test_rescue + 기존 test_contract/test_benchmark **30 PASS**, pinned author
실제 batch_edit 함수 실행 포함. 진단 유무 BF16 W/FP32 H/RNG exact, 작은 실제
solve 잔차/스칼라 일치, 품질중단·gate·신규 tracking group/schema 검산.
official.tools.verify: source171/Python438/source_integrity PASS/external_task_imports0.
실제 pretrained/GPU/메모리 피크/성능 검증은 NOT_RUN. SH1/SH2 신규 제출0/HOLD.

조건부 Flu/Con 미완성 초안은 ignored local/fe-qwen-working-20261012/
deferred-flucon-draft-r1에 보존했고 이번 실행 소스 배포에는 포함하지 않는다.
root가 rent 자산·runtime·스토리지·전체 GPU<=2 및 실행/모니터링을 소유한다.
작은 Git 소스/보고만 전달: NO_BROADCAST_NOT_REQUIRED, 모델/CP/raw 이동0.
