# E0/E1 warm L4 B011–B020 사실 보고 — 부분 완료

job45805는 Scheduler COMPLETED 0:0이며 native 10 batches / 1000 cell-request executions를 저장했다. 최초 B011은 E1 cell로 재사용하므로 추가 100회를 더하지 않는다. Cold L4의 B001 1회와 합해 관측 native batch는 11개이고, E1 native-first-batch coverage는 2/20 cells다. E0/E1 전체 완료나 원 trajectory 동등성 PASS가 아니다.

## 무결성과 복원

저장된 10개 W/M checkpoint 및 native raw member의 size/SHA, FP32 shape/finite, checkpoint tensor SHA, history entry→직전 endpoint 10개 연결을 검산했다. 저장된 연속 W의 실제 Δ norm과 native receipt의 Δ norm 잔차는 batch 표에 보존했다. 그러나 각 batch의 before-W SHA가 별도 기록되지 않았으므로 이 값만으로 모든 내부 시점 W byte 연속성의 독립 증명을 주장하지 않는다.
원 실행은 각 native 호출에서 selected pointer·nonselected bytes/version·P 불변을 확인하고 W/M rollback/reinstall을 검사했다. 종료 receipt는 W0 pointer/bytes 및 RNG restore PASS, version 증가 1개는 NOT_CLAIMED다. Hook registry 복원과 마지막 counter-hook remove는 source 경로를 확인했지만 terminal이 finally 이전에 저장되어 별도 최종 hook inventory receipt는 없다. 저장된 매 batch RNG는 schema/identity를 봉인했지만 다음 batch entry-RNG hash가 없어 독립 byte 연결은 미기록이다.
S2 B020 comparison checkpoint는 실행 lock에 없었고 resume_fidelity는 REFERENCE_NOT_YET_RECEIVED다. 추후 exact reference 비교는 별도 artifact로 해야 하며 이 보고에서 원본과 동등하다고 주장하지 않는다.
B011 target recomputation byte-exact는 0/100, max absolute delta 0.0156555101. 원인을 hardware/solver에 자동 귀속하지 않는다.

## B011 관측 평가

RS/PS = NLLnew < NLLtrue, NS = NLLtrue < NLLnew이며 tie는 failure다. NLL은 해당 continuation에 대해 lower-is-better다. Current100과 outcome-independent Historical128은 같은 request/prompt/order와 evaluator identity로 W0/ENTRY/NATIVE에 반복 평가했다. Historical은 전체 seen1000 점수로 확대하지 않는다.

| Panel | Endpoint | RS | PS | NS |
|---|---|---:|---:|---:|
| current | W0 | 10/100 (10.0000%) | 22/200 (11.0000%) | 900/1000 (90.0000%) |
| current | ENTRY | 19/100 (19.0000%) | 33/200 (16.5000%) | 832/1000 (83.2000%) |
| current | NATIVE | 100/100 (100.0000%) | 195/200 (97.5000%) | 812/1000 (81.2000%) |
| historical | W0 | 13/128 (10.1562%) | 25/256 (9.7656%) | 1085/1280 (84.7656%) |
| historical | ENTRY | 128/128 (100.0000%) | 246/256 (96.0938%) | 1004/1280 (78.4375%) |
| historical | NATIVE | 128/128 (100.0000%) | 247/256 (96.4844%) | 994/1280 (77.6562%) |

Current 1300 + Historical 1664 = endpoint당 2964 prompt pairs, 세 endpoint 총 8892 pairs / 17784 true-new sequences다. 반복 endpoint와 서로 다른 panel을 독립 unique population으로 합치지 않는다. B012–B020에는 이 프로그램의 별도 endpoint RS/PS/NS가 저장되지 않았다.

## 실제 비용과 미계측

단일 할당 8810 GPU-sec (2.447222 GPUh), 프로그램 8803.425018s, 초기 gate 906.820731s다. .batch/.extern을 더하지 않는다. Model load, batch-native, 첫 batch panel evaluation, 기타 overhead를 분리했다. target/key/solve/copy/history-verification은 native 안의 nested host time으로 총합에 다시 더하지 않는다.
Peak allocated 37272451584 B / reserved 42469425152 B. Forward 호출 26208, input positions 3795583, nonpadding positions 3071966. 이 값은 FLOPs 또는 kernel 실행시간이 아니다.
새 general128, all-position query exposure, projected spectrum, 우선 L4/L8 n5000/9000 signed backward/FD는 미완료다. compute-z final training NLL/iterations/stop/clamp는 NOT_OBSERVED다. Cold 보고서와 E1-A 10-arm CPU package는 기존 receipt만 연결하고 재실행/전수 재감사하지 않았다. 미측정 결과를 0 또는 기존 aggregate로 대체하지 않는다.

## 재현과 범위

`python -m project.run_scripts.baseline_mechanism_first.warm_analysis --attempt <sealed attempt> --destination <new directory> --allocated-gpu-seconds 8810 --scheduler-identity "45805|janghj|COMPLETED|0:0|gres/gpu=1"`

CPU artifact 검산만 수행하며 model/GPU/Slurm/새 evaluation=0. 최종 원인종합은 GH 소유다. scientific_promotion=false.
