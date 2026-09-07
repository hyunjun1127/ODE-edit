# Alpha JV 병렬 연구 계획

2026-09-07. 설계 문서이며 신규 GPU 실행 또는 method promotion이 아니다.

- [상세 보고서](report-ko.md): 완료 결과, 두 Server4 task에 따른 claim 분기, Llama 진단, 모델별 Euler sweep, 최소 functional-history variant, method closure와 ablation.
- [작업 행렬](task-matrix.csv): 우선순위·초기 범위·의존성.
- [GH 기존 10개 후보 연결표](gh-candidate-disposition.csv): 통합 분석 §13의 후보를 유지하면서 이번 독립 병렬 run의 우선순위로 배치.
- [설계 manifest](design-manifest.json): source reference와 제안 설정. 실제 runtime/science/resource lock을 대신하지 않는다.

`odeedit_orbode_cum_s4_r1`와 `odeedit_alpha_l8_s4_takeover`의 source/process/GPU/미완료 산출물은 변경·사용하지 않았다. 기존 완료 publication과 봉인된 실행 source만 근거로 사용했다.
