# E01 L4 재개: 원 checkpoint 비교 보충

Job 45805는 COMPLETED 0:0이며 B011–B020 native 10 batches, 1,000 request executions를 완료했다. Allocation은 8,810 GPU-seconds, 프로그램 시간은 8,803.425초다. Initial 906.821초는 전체 비용이 아니다. 기존 warm 분석 및 원 실행 bytes는 변경하지 않았다.

S2 보존 catalog의 승인된 35 checkpoint 중 S1의 정확한 3개를 재사용하고 32개를 선택 수신했다. 수신량은 33,824,003,616 bytes다. 삭제된 S4 checkpoint를 사용하거나 재생성하지 않았다.

| 비교 | 실제 W 차이 norm | W 상대 norm | M 상대 norm | 판정 |
| --- | ---: | ---: | ---: | --- |
| cold L4 B001 | 0.00462702349 | 0.00005948168 | 0.00000354105 | NONEXACT_CAUSE_UNRESOLVED |
| warm L4 B020 | 10.40744161 | 0.12083878923 | 0.00000120938 | NONEXACT_CAUSE_UNRESOLVED |

각 비교에서 batch, seen IDs, model revision, contexts, Python/NumPy/Torch/CUDA RNG 및 covariance metadata는 일치했다. 이는 W/M tensor 동일성 또는 전체 trajectory 동등성을 의미하지 않는다. Warm B011 재계산 target은 100개 모두 byte-exact하지 않았으며 최대 절대 차이는 0.0156555101이다. Cold target 차이와 누적 endpoint 차이를 하드웨어 원인으로 확정하지 않는다.

원 trajectory equivalence는 UNRESOLVED로 유지한다. 이미 생성된 독립 native 결과와 E1-A 수치는 보존하며 원 trajectory의 대체값으로 쓰지 않는다. 다음 L4 n5000 실행은 원 B050에서 독립 시작하고 수신한 원 B060과 비교한다. Warm B020을 이어 붙이지 않는다. Native 식·target 최적화·P·M·precision은 바꾸지 않는다.

Warm B011 Current: W0/ENTRY/NATIVE의 RS는 10/19/100 (분모100), PS는 22/33/195 (분모200), NS는 900/832/812 (분모1000)이다. 상세 분석은 sibling `../warm-l4-recall-r2/factual-report-ko.md`를 참조한다. E01 전체 완료가 아니다.

## 근거 및 한계

- `cold-l4-B001.json`, `warm-l4-B020.json`: 실제/reference 파일 SHA와 chunk FP64 tensor 차이, metadata 비교.
- `project/run_scripts/baseline_mechanism_first/checkpoint_compare.py`: 재사용 가능한 CPU 비교 코드. 실행 인자는 각 JSON의 actual/reference 경로 및 layer4.
- 기존 warm package는 reference가 실행 당시에 없었다는 사실을 유지한다. 이번 수신 후 비교를 별도 supplement로 추가했다.
- 원 endpoint exact 여부와 intermediate target/context/order/RNG fidelity는 별개다. 원인 합성은 GH 소유이며 scientific_promotion=false.
