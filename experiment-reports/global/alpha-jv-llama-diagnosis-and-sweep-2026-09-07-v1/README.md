# Llama 원인 진단과 hparam sweep: 독립 병렬 실험 설계

2026-09-07. 이번 산출물은 설계와 전달 지시문이다. 새 runtime 구현·GPU 제출은 하지 않았다.

- [진단 설계와 판단 기준](design-ko.md)
- [GH에게 그대로 전달할 실행 지시문](gh-execution-prompt-ko.md)
- [실험 configuration 표](candidate-grid.csv)
- [설계 manifest](design-manifest.json)

두 트랙은 서로의 완료를 기다리지 않는다: **D = historical Llama B10 원인 진단**, **S = 정상 development state의 Llama/Qwen λ·T·N sweep**. 둘 모두 Server4의 `odeedit_orbode_cum_s4_r1`, `odeedit_alpha_l8_s4_takeover`와 독립 자원·source·output에서 병렬 진행하도록 설계했다.

상위 연구 계획: [AlphaEdit JV: Server4 병렬 작업과 Llama 개선 중심의 연구 계획](../alpha-jv-parallel-research-plan-2026-09-07-v1/report-ko.md). 기존 GH 통합 v2의 후보 ID/설정은 유지하며, 사용자 추가 요청에 따라 5개 λ의 actual development 비교를 진단과 병렬인 기본 범위로 올렸다.
