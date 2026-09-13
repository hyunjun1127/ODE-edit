# E01 원본 대비 replay 성능 차이 — 기존 관측 보충

동일 case/prompt/target identity와 request order가 일치하는 Current-B100 관측만 paired 비교했다. RS/PS는 new NLL < true NLL, NS는 true NLL < new NLL이며 tie는 실패다. Δpp는 replay−원본이다.

| Endpoint | 지표 | 원본 n/d | Replay n/d | Δpp | 성공→실패 | 실패→성공 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| B001 | RS | 100/100 | 100/100 | +0.000 | 0 | 0 |
| B001 | PS | 190/200 | 190/200 | +0.000 | 0 | 0 |
| B001 | NS | 867/1000 | 867/1000 | +0.000 | 0 | 0 |
| B011 | RS | 100/100 | 100/100 | +0.000 | 0 | 0 |
| B011 | PS | 195/200 | 195/200 | +0.000 | 0 | 0 |
| B011 | NS | 813/1000 | 812/1000 | -0.100 | 1 | 0 |

true/new NLL 및 desired-margin의 원본/replay/paired delta mean·median·p90·max는 CSV에 기록했다. 원래 raw margin은 수정하지 않았다. W/M 차이는 checkpoint-comparison-recall-r2의 별도 tensor 비교이며 성능 유사성이 trajectory fidelity PASS를 뜻하지 않는다.

B011은 B020 endpoint가 아니다. B020 replay의 같은-state 성능은 NOT_MEASURED다. B011 Historical128에 해당하는 원본 B011 seen-full 관측은 수신 inventory에 없으므로 다른 checkpoint의 full-prefix 총점으로 대체하지 않았다. Cold B001은 원본 B001과 직접 비교했다. B060/B100은 아직 replay terminal 관측이 없으며 기존 runner는 최초 B051/B091 성능을 저장한다. Terminal 성능 보충 forward는 다음 사용자 recall의 별도 관측으로 남긴다.

원본 source BLUE311b076a 및 singleton L4/L2=1, 같은 fixed10k order·Llama revision·evaluator 규약을 사용한다. 실행 host와 replay source 및 W/context/target/RNG 차이는 원 cold/warm report와 input locks에 별도로 결속되어 있으며 완전 수치동일성으로 간주하지 않는다. 신규 model/GPU/forward=0, scientific_promotion=false.

재현: `python -m project.run_scripts.baseline_mechanism_first.performance_compare --plan /mnt/raid5/janghj/.codex/worktrees/odeeditsh1-baseline-mechanism-first-e01-v1/local/baseline-mechanism-first-e01/20260912-v1/locks/performance-comparison-recall-r1/plan.json --output <new-directory>`
