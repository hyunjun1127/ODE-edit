# Qwen FE 단일 clamp 0.5 개선 소스

동일 USER-FE-QWEN-WORKING-CURRENT-Z-20261012-R1의 새 prospective arm이다.
기존 L2=100/clamp1은 B9 QUALITY_EARLY_STOP이며 W20/선택 성공이 아니다.
원 소스/CP/raw/동결 attempt는 변경하지 않는다.

- 새 계약: rescue-L2-100-clamp-0p5-v1. 정확히 한 cold heldout arm.
- L2=100 유지, author llms.clamp_norm_factor만 1에서 0.5로 변경.
- 현재 batch-entry z, BF16/native solver/history, 동일 heldout 및 평가/저장 일정 유지.
- prepare/launch/observer의 작은 opt-in plumbing으로 config, 실제 Hydra argv,
  native resolved config, raw/CP identity, tracking에 두 설정을 결속한다.
- 독립 collector는 모든 기존 W20 raw/commit 재집계와 함께 native runtime/CP의
  clamp=0.5 및 L2=100을 확인한다. 이전 clamp1 proof/quality-stop 승격은 거절한다.
- stop R80/P70/N65와 W20 선택 R95/P85/N70은 불변. exit1을 수치실패로 바꾸지 않는다.
- API/CLI: official/diagnostics/fe_qwen_current/RESCUE_CLAMP.md.
  동일 entrypoint에 조건부 final CF/zsRE가 포함된다. 기존 Flu/Con adapter는
  새 계약을 아직 거절하므로 별도 후속 연결 전 사용하지 않는다.

검사: CPU unittest 38 PASS(신규 7 및 기존 회귀 31), 실제 pinned YAML의
clamp-only override/실제 launch argv/native guard/old-proof 거절/quality 분류 포함.
official source verify: 171 source SHA PASS, Python444, external task imports0.
GPU/model forward/제출/실제 W20 성능은 NOT_RUN. SH1/SH2 HOLD 유지.
실제 rent admission/실행/모니터는 요청자 소유이며 이 source 게시와 구분한다.
NO_BROADCAST_NOT_REQUIRED: 작은 소스/보고만 Git, 원 raw/CP는 현 위치 유지.
