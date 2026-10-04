# V14 B1 구현

원권한: ODEEDIT-USER-GH-SH4-JLZ-V14-NATIVE-WRITER-B1-20261005-R1.
cold W0/H0, first100, 요청별 공유 반경 .75, fit 1회·commit 1회만 허용한다.

- builder: 모든 rewrite/KL row의 actual lower materialized weight 경로.
  rewrite-only FP32 nested mean을 전체 요청으로 모은 뒤 native same-A ridge solve.
  CPU RAM stage boundary와 row-group reverse replay로 전체 K/P/direct/input adjoint를 합한다.
- subject: entry model에 context별 actual v를 주입하고 native SUM F gradient를 모은다.
  전체 요청의 공통 종료 판단 후 동일 candidate forward를 backward용으로 재계산한다.
  이 재계산은 별도 candidate/fit가 아니며 replay 비용을 별도로 기록한다.
- optimize: V12 EfficiencyAdam/projection만 exact 재사용. analytic requested-u norm은 1회.
  요청별 영구 freeze 없음. terminal은 backward 없이 마지막 공통 evaluated payload.
- telemetry: 마지막 payload exact copy, actual final native rewrite keys로 H 1회.
  ideal/cast/effective/local-action, masked/actual 차이와 signed realization을 분리한다.
- qualification: B1 부분 입력의 고정 zero/nonzero 후보만 검사한다. 별도 sequential pilot,
  full-B 재fit, P/N 품질 gate, tolerance 완화, CP 저장, 자동 retry 없음.
- collector: 원 NLL/TF row와 token identity, denominator, candidate/업데이트,
  공통 terminal, H/commit 및 parent 비용을 CPU에서 별도로 검산한다.

소스의 CPU toy PASS는 actual 8B GPU PASS가 아니다. 실제 모델 검사는 봉인한 GPU job의
첫 단계에서 수행하며 실패하면 B1 commit을 차단한다. V13은 resource afterany 대상일 뿐
과학적 성공을 V14 선행조건으로 요구하지 않는다.
