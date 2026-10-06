# AlphaEdit 6-cell 실행 전 owner 검토

- nonce: `USER-GH-SH4-JLZ-PRICE-ALPHA-WRITER-2K-20261007-R1`.
- 실제 server4/session/origin 및 전용 worktree 경계를 확인했다. root dirty와 기존 MEMIT archive/job/raw는 변경하지 않았다. 독립 reviewer는 사용하지 않았으며 owner source audit이다.
- GH envelope SHA `95ada04d7cf1e6cc19a3fc9be10fd10fa2381705c83c7a681e52e0b7cf2ead80`, package manifest SHA `147ea5d9ae3432a623bcbed8181486399133424b1e47da6dd86de41ef7c3581b`. execution JSON, handoff, authorized contract, review, 두 manifest를 정독했다. manifest의 4 member와 manifest 포함 5 regular/37,453 bytes를 검산했다.
- 구현 기준은 MEMIT 실행 source `87a5a736c455d5082f5f666ae562f9edc4e5d3b2`. 새 namespace에 필요한 controller/runner/collector adapter를 provenance copy했다. 기존 shared source 변경은 0이다. 기존 entry packing/mean reduction/subject active mask/absolute EfficiencyAdam/projection/observer/transaction을 재사용하며 과거 actual PASS를 Alpha PASS로 표시하지 않는다.
- Alpha 연산자는 `A0=I+NH`, FP64 LU 1회/layer/batch, whole-B thin solve이다. 가격 mean M, raw-context pullback, terminal materialized FP32 weight가 동일 Q를 사용한다. fresh upper keys를 매 candidate 계산하며 첫 eligible key/factor cache만 같은 entry에서 재사용한다. dK/dQ/reverse 호출은 없다.
- Alpha LOO는 `(I+NH)q+NKminus(Kminus.Tq)-Nkr`이며 원래 unprojected key norm으로 정규화한다. B1 고정 두 pair/모든 층 cached matvec만 추가한다. operator residual 1e-8, LOO relative 1e-6/zero absolute 1e-8 및 coefficient 원기준을 고정했다.
- C0는 native FP32 count division 후 FP64 product, H는 CPU FP32이다. Q_C0와 Q_H는 각각 직접 계산한다. C0 scale1이며 비대칭 A0를 에너지로 쓰지 않는다. projector .02/lambda_alpha1 고정, hybrid15000C0·사후투영·Cholesky·SVD·tracking·divisor·역실현 보정은 없다.
- native cap .75/base .75 또는1/진짜 uncapped null, max(base,.75maxprice), endpoint tagged repair, tiny report-only, 25eval24update, grace12/4stage, active mask와 terminal exact copy/H once를 유지했다.
- Llama/Qwen projector 전체 SHA가 정본과 일치했다. CPU mmap shape/dtype와 physical layer4..8 index를 결속했다. 모델/입력20pack/observer26000행/runtime/native/C0는 기존 exact receipt와 현재 stat으로 재결속했다.
- Source AST/import/CLI/DAG/config 검사는 CPU-only이다. synthetic/toy/pilot/추가 fit/forward/backward/solve/모델 로딩은 0이며 actual B1은 `NOT_OBSERVED`다. 정적 receipt 최초 파일은 보존했고 source 수정 후 새 final receipt를 사용한다.
- RAM/VRAM 계획: Llama host37.9151/GPU69.8838 GiB, Qwen host47.2371/GPU78.9637 GiB. full projector mmap residency, FP32 C0/H+rollback, LU, factor transient, W&B sidecar4GiB를 포함한 코드 기반 상한 계획이며 실측 peak/ETA가 아니다.
- 6개 출력 serializer+spool/temp/error reserve14.8513GiB, 선행 MEMIT reserve까지 합29.7026GiB를 admission에서 확인했다. quota 미제공은 별도 기록하며 공유 filesystem 예약으로 주장하지 않는다. 매 batch 다음 최대 write+error reserve를 다시 검사한다. noCP, durable W/H/M/P/K/R/activation/optimizer dump0.
- W&B는 SH1 helper read-only, 새 job startup online 필수, task/model/writer/arm identity는 AE cell과 local runtime에 결속한다. scalar whitelist 외 raw/key/console/code/watch upload0. 기존 science pin 변경0.
- 선행 제출/lock/source와 59931→59932→59933,59934→59935→59936의 afterany graph를 확인했다. 양 Alpha 첫 job은 공통 frontier59933/59936 뒤에만 시작한다. 총 사용자 cap2이며 새 2GPU 추가 할당이 아니다. submit 직전 fresh admission, held fullargv/source/resource/dependency 검사 후 release한다.
- No broadcast: 신규 과학 raw가 아직 없고 소형 source/receipt는 Git으로 게시한다. `NO_BROADCAST_NOT_REQUIRED`. 자동 retry/monitor/heartbeat0.

## 실제 검토 한계

수학적·실제 GPU parity PASS를 정적 검사로 대신하지 않는다. 실제 operator/LOO/KKT/FP32 feasibility/payload/H 검사는 각 실제 B1의 기존 계산에 통합되어 실패 시 typed failure로 기록된다. 낮은 성능·실현률·finite budget 미수렴은 중단 gate가 아니다.
