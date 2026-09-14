# I1 native exact parity 기술 수정 r2

첫 기술 array 46990은 native와 I1 각 B100을 완료했지만 exact parity는 실패했다. 최초 attempt 및 비교 receipt는 불변 보존한다. Target max absolute 차이 0.00012734532356262207, stored weight 차이 3.423541784286499e-06, .75 partial 차이 2.5676563382148743e-06이다. Entry W/M, 최종 history 및 fixed-W optimizer pause/resume은 exact였다. 두 allocation의 합은 942 GPU-sec이며 본 정책 chain 분모에 포함하지 않는다.

100 request 모두 loss iteration 0/1은 exact이고 iteration 2부터 차이가 발생했다. 동일 FP32 forward에서 shared non-leaf `u+(a0-aj)`의 zero-offset AddBackward가 여러 context의 gradient를 모으는 순서를 native leaf `u` 대비 바꾸는 CPU 회귀 사례를 확인했다. 이는 loss 값의 수학적 식 변경 없이도 leaf regularizer gradient와의 합산 순서를 바꿀 수 있다.

r2는 `torch.equal(a0,aj)`인 exact zero-offset 상태에서 동일 leaf `u`를 직접 사용한다. Nonzero offset에서는 원 계약의 `u+(a0-aj)`를 유지한다. u/Adam identity, teacher, regularizer, clamp, 예산, native solver 및 tolerance는 변경하지 않는다. 이 분기는 성능에 따른 선택이 아니다. 실제 causal RCA 확정은 수정 I1의 native exact 비교 결과와 구분한다.

추가 GPU 기술 실행은 수정 I1 B100 한 개만 수행한다. 완료 native B100의 terminal/payload/full SHA를 재사용하며 중복 native 실행을 비용 절감을 위해 생략한다. 수정 attempt의 comparison receipt에 두 execution lock과 입력 SHA를 함께 기록한다. CPU PASS는 GPU parity PASS가 아니다. Main 네 정책은 exact 비교 PASS 이후에만 제출한다.

원본 BLUE 및 기존 NativeSingletonFitter.fit 수정 0. 실패 출력 삭제/덮어쓰기 0. 정책 선별/후속 Late 실행 0.
