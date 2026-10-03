# Q1 actual qualification

Job57698 COMPLETED/0:0. Source c2d5fb107a0435491d8b4705b43b75f6177bbb5c, execution lock cf03a94ccf873d93f36bdfe3c0a378998bc2340ffbb4cda4d5b18914c31cbe23.

A/B 독립 cold W0/H0, 각각 case541/6693→16935/17306의 BS2×2, 총100후보96update를 완료했다. Native full-block hook reference와 Tprime dense reference, reversed MB1 whole-B gradient, fixed-candidate B1/B3 shape, 실제 rollback, H once, own W/H/RNG/context/ledger 연속성 검사를 수행했다. Q1 state는 main에 전달하지 않는다.

Dense gradient RMS 최대차8.102680131515143e-9; reversed MB1 최대2.9494929576802316e-7. 고정 forward1e-5+1e-4relative/gradientRMS1e-6+1e-3relative 안이다. Full native hook reference loss/gradient 차0. Actual evaluated endpoint를 모델에 commit한 뒤 측정한 NLL 최대차0. Same-A solve residual은 약1e-14이며1e-8 경계 이내다. 전체 model/다른host bitwise equivalence나 임의 batch의 보편적 증명은 아니다.

Scheduler parent425초×1GPU=425GPU-sec(약0.1181GPUh). 프로그램422.996초는 그 안에 포함되며 두 수치를 더하지 않는다. Torch peak76,617,735,680B(약71.36GiB), process peak40,387,804KiB(약38.52GiB). Q1의 실제 비용이며 main B100의 시간/peak는 별도다. Component 진단과 fixed-candidate qualification은 추가 backward 비용이고 main250후보 budget에 끼워 넣지 않는다.

Main A57699 RUNNING을 확인했다. 이 글 작성 시 main B1 commit/observer→B2는 아직 미확인이다. Main PASS/500완료 또는 B chain 완료로 표시하지 않는다. NoCP를 유지하며 과거 reference/model 복원·전송0이다.
