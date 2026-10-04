# JLZ v12 SH4 → SH3 이식·등록

상태: **IMPLEMENTING_NOT_SUBMITTED**. 실제 S3 GPU qualification과 main 완료는 미관측이다.

Nonce `ODEEDIT-GH-SH4-SH3-JLZ-V12-MIGRATION-20261004-R1`. S4 실행 source `7852d66ed5467f94ebda73507b6ccdbf5c921328`의 수학 코드를 그대로 사용하며 S3 전용 prepare/submit과 tests만 추가했다. 원 정본27개 SHA/size 및 CSV2000행의 모든 identity 필드를 검산했다. CPU reference29·production7·이식5 검사가 통과했다. owner audit이며 별도 독립 agent 검토는 수행하지 않았다.

SH4 main58173/collector58174는 source 소유자가 취소했고 할당 GPU초0이다. S4 pilot58172는 완료·410GPU초의 역사 자료이며 S3 qualification으로 쓰지 않는다. 소형 config/lock/provenance6개121342B만 allowlist로 수신했다. 원자료 KEEP, 큰 자산 전송0.

계획 DAG는 S3 pilot BS2×2 → 동일source/config technical READY를 검사하는 fresh V12_MAIN BS100×20 → afterany CPUcollector다. 최대동시1GPU, 각8CPU/59392MiB, pilot4h/main168h 상한, collector0GPU/8CPU/24576MiB/4h이다. Wall은 ETA가 아니다. 최대논리평가 pilot100/main50000, update96/48000은 사전상한이지 실측이 아니다.

S3 기존 Python3.12.3/torch2.9.1+cu128을 재사용하고 task-local dependencies-r1에 transformers4.57.1/tokenizers0.22.1/jsonschema4.10.3/pyrsistent0.20.0을 고정했다. 공유 venv4.44.2는 변경하지 않았다. S4 runtime와 비교한11개 소스 SHA는 일치한다. Model FP32/eager/TF32off, geometry FP64, historyCPUFP32를 유지한다. Native BLUE311b076은 task-local 기존 사본이고 evaluator는 원 jlz_realization/observe.py·inputs.py를 보존한다.

모델 revision8afb486c1db24fe5011ec46dfbe5b5dccdb575c2, fixed10k dataset, native6contexts, L4–8 C0는 기존 S3 fullSHA receipt와 현재 size/inode/mtime·논리 경로로 재결속했다. S4 W0 raw는 수신하지 않는다. 승인된 S3 fresh first2k W0 관측1회를 main 안에서 수행하며 학습에 피드백하지 않는다.

RAM 계획: model CPU loading peak 약34GiB, fit/write phase history+rollback 약8.8GiB, entrycache3GiB, 1층 FP64 system/solve 등12GiB 및 여유4GiB. 두 phase peak를 중복 합산하지 않는다. 과거 S4 pilot MaxRSS33.12GiB/GPU35.62GiB는 참고이며 S3 peak 보증이 아니다. S3 request58GiB를 상향하지 않는다. H200 GPU 계획 model30+write3+activation20+solve10+여유8=71GiB. 디스크 출력·temp·여유24GiB를 reserve하고 최신 실제 free/inode는 admission receipt에 기록한다.

`save_checkpoints=false`, exact resume 불가. 설정·스칼라·평가 raw만 저장하고 W/H/delta/RNG 복원 bundle은 저장하지 않는다. v10/v11과 다른 중지 task는 재개하지 않는다. Release 후 resource/dependency snapshot1회에서 agent monitoring을 중단한다. 본 프로그램은 pilot qualification 후 W20와 CPUcollector까지 자연 진행한다.
