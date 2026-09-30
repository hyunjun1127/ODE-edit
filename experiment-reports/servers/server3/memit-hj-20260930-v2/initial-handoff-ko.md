# MEMIT HJ v2 초기 gate 및 자율 DAG 인계

**대표 초기 gate ACTUAL_PASS. Agent 상태는 MONITORING_PAUSED_AWAITING_USER.**
전체 실험 완료 보고가 아니다. 등록된 runner/collector는 자연 진행하며 초기 경계 이후 scheduler/log/result polling·heartbeat·자동recall을 하지 않는다.

## 등록된 전체 계획

|job|단계/cell|dependency|source|
|---|---|---|---|
|56007|P: actual T0a, W0 전체10k — COMPLETED|없음|a8126eb65cbe9a1814b381007798d857003555ed|
|56033|A: W0 E0/T1,000/001,T0b,writer12/history8 총22cell|afterok56007|f1d7a995460e2fce8fb10134f723ba4680303a31|
|56034|B:100/101|afterok56007,afterany56033|동일 후속source|
|56035|C:010/011|afterok56007,afterany56034 + calibration PASS|동일 후속source|
|56036|D:110/111|afterok56007,afterany56035 + calibration PASS|동일 후속source|
|56037|독립 CPU reducer/보고/그림/실패·차단 수집|afterany56007:56033:56034:56035:56036|동일 후속source|

28cell 전량 mapping은 `submission-r2.json`에 있다. 모든 후속은 held owner/source/fullargv/node/resource/dependency 검사 후 release했고 임시hold는 남기지 않았다.
cap1: 각 GPU job1GPU/8CPU/119GiB/exportNONE/Requeue0/ubuntu, CPU collector0GPU/8CPU/32GiB.720h GPU wall은 provisional partition ceiling이며 전체 ETA나 GPUh hard budget이 아니다.
총GPUh/완료예상은 PENDING_CALIBRATION. T0b의 실제 SPG call·시간 및 native/진단/관측 성분으로 프로그램이 cost-lock을 게시한다. 별도 사용자 승인 없이 허용된 후속이 진행된다.
Z가 교정되지 않으면010/011/110/111만BLOCKED이며 native000/100/001/101과 독립 진단은 계속한다. NOT_FIRED는 parent artifact alias다.

## 실제 확인한 초기 경계

Job56033의 `writer_0_divisor`(E0/T1에 공유되는 W0 BS10 진단)에서 exact fixed-order 첫10요청을 편집했다.
H0=0에서 native z10회, native solve5회, native key10회, 모든5층write 후 post-key history append각1회가 기록되었다.
유한한W/H와cache_c반환을 확인했고, Current R/P/N 관측 이후W/H/RNG/context/ledger가 변하지 않았다.
C00010 commit의전체endpoint identity와C00020 entry identity가exact일치한다. 이검사는방법의성능개선gate가아니다.

|지표|성공/분모|NLL preference|TF token micro|TF prompt macro|TF strict|new NLL|true NLL|
|---|---:|---:|---:|---:|---:|---:|---:|
|RS|10/10|100.00%|100.00%|100.00%|100.00%|0.049312|10.361290|
|PS|15/20|75.00%|45.00%|45.00%|45.00%|3.287610|6.752019|
|NS|83/100|83.00%|10.00%|10.00%|10.00%|11.494316|6.315219|

RS/PS의desired는new, NS는true다. TF는자유생성정확도가아니며 위작은첫10표본을전체10k성능이나방법간우열로해석하지않는다.
첫write 45.884s, observer 16.803s, step 63.671s, entry RAM snapshot 6.746s. 중첩timer를합산하지않는다.

## T0, W0, 수리와 미검증 범위

P56007: BS100 native/adapter z/key/W/H/RNG/context/ledger 및평가행exact PASS, L4 key불변, 실제임시CP재로드/output복원PASS.
5,284,843,077B 기술CP는본task manifest의owner/hash/refcount0 확인후단일파일만tombstone삭제했다. 과거CP/원자료삭제0.
W0전체10k의분모는R10000/P20000/N100000이며동일observer비변이검사PASS. Source-bound readiness와W0 raw SHA를후속이소비한다.
Cached/native 초기loss차이0이지만gradientrelative1.0194003152719233e-5/bitwise=false이다. Oracle precision은NOT_ESTABLISHED_AT_ORIGIN이며T0b/Z PASS로승격하지않는다.
P 실제parent allocation2738GPU-sec=0.760556GPUh. batch/extern을더하지않는다. T0기술200요청/W0평가는과학physical요청과분리한다.
P process peak host 32.892GiB, GPU allocated 50.954GiB/reserved 58.010GiB. FP64교정·장기branch peak가검증됐다는뜻이아니다.
CPU27검사PASS. 독립reviewer는없으며owner audit이다. SPG plateau에서초기점을제외하고Armijo수락6점을요구하는경계수리를실제SPG시작전에반영했다. 임계값/방법/표본변경0.
P가소비한native/T0/W0함수bytes와입력은동일하고SPG호출0이었다. P원source를명시재사용하며새후속source로P를실행했다고기록하지않는다.
구source미실행 A56008/B56009/C56010/D56011/CPU56019 및앞선CPU56012는정확pending확인후교체취소했다. 기존submission/source/cancel증거보존, 중복scienceprefix실행0, 타job변경0.
현재보지않은것: joint진단GPU경로, T0b32요청교정, SPG·refresh본체, 남은cell endpoint, 전체terminal. 등록/CPU PASS/대표initial PASS를그완료로대체하지않는다.

## 정확 source/lock/경로

- 후속source: `f1d7a995460e2fce8fb10134f723ba4680303a31`.
- 후속lock SHA: `023d92c47d33e7d57b7eb4eee98d4f37ffe561c46b06e3ae5b6657d7102ed9f1`.
- 후속lock: `/data/janghj/ODE-edit/local/memit-hj/20260930-v2/attempt-r2/execution-prerequisite-bound.lock.json`.
- P source/lock: `a8126eb65cbe9a1814b381007798d857003555ed` / `1aa28cb09de1cae5c00824bdafb86eb2a7f83dda24422cf5cf11906a055d43c5`.
- 공통raw output: `/data/janghj/ODE-edit/local/memit-hj/20260930-v2/attempt-v1/output/`.
- 초기gate: 위output의`actual-initial-gate.json`; firstcommit `cells/writer_0_divisor/C00010/commit.json`; nextentry `C00020/entry.json`.
- 전체held/release/mapping: `audits/servers/server3/memit-hj-20260930-v2/submission-r2.json`.
- 고정초기증거: `audits/servers/server3/memit-hj-20260930-v2/actual-initial-handoff.json`.

전체완료회수·상세결과리뷰는사용자recall에서수행한다. 등록job은중단·hold하지않는다.
