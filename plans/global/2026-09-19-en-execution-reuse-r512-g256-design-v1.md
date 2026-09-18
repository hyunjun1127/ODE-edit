# EN 중복 실행 제거와 R512·W0 생성256 전환 상세 설계

작성: 2026-09-19. 상태: DESIGN_ONLY_IMPLEMENTATION_PENDING.
근거 source: 3a399b93b1a4ff14909c23f029655f5c0ef5bba5.
사용자 지시: reference512 전부 사용, BASE 최대 생성256, EN 중복 실행 비용 제거를 우선한다.

기계 판독용 범위·조건: [설계 계약](2026-09-19-en-execution-reuse-r512-g256-contract-v1.json).

이 문서는 구현·검증의 명세다. 실행 코드 변경, GPU/model 실행, teacher 재생성, job 제출, sequential 재개는 수행하지 않았다. 기존 BPCW reference128 미완료 변경과 별도 worktree에서 작성한다. 과거 EN/BPCW 산출물은 그대로 둔다.

## 1. 이번 변경의 정확한 경계

먼저 같은 입력·teacher·native·보정 정책에서 중복 실행만 제거한다. 그 뒤 동일한 실행 경로에 R512·generated256 데이터를 연결한다. 비용 비교에서도 두 변경을 분리한다.

|항목|정의|
|---|---|
|기본 구조|EN-F: 현재 edit 반응을 유지하는 single L4 correction|
|Reference|기존 봉인된 R512 전부. objective·gradient·후보 objective 검증에서 모두 사용|
|W0 continuation|greedy raw argmax, max_new_tokens=256, 실제 configured EOS에서 조기 종료|
|문서 입력|기존 BOS+128 자연 tokens, 총129 IDs 유지|
|Reference gradient|필요한 gradient sweep마다512문서 전체, 실제 생성된 모든 위치, 전체 vocabulary|
|Gradient 횟수|EN-F 기존1회/보정 batch 유지. 문서별 backward512회를 포함하는 전체 sweep1회|
|Trial|최대8, 기존 Polyak 제안·반감·첫 통과 수용 유지|
|현재 edit 보호|K_E 전체 valid-token key, 동일 projector·rank 판정·FP32 검사|
|History|중복 제거 단계는 기존 Past64 선정·own-WN anchor·NLL/ID 조건 유지|
|공식 P/N, Dev128|선택 후 observer. gradient·scale 선택에 사용하지 않음|
|GSS|이번 reference512 선택/생략에는 사용하지 않음. history 방향 구성은 후속 별도 방법 변경|
|데이터/생성 ID|EN-R512-G256-v1. 실행 중복 제거 ID와 분리|
|실행 권한|이 문서는 배치 실행/dispatch가 아니다. 자동 sequential 진입 없음|

History를 보정 목적에 넣자는 앞선 목표는 유지한다. 그러나 history loss·recency weight·anchor까지 동시에 바꾸면 중복 제거의 수치 동등성을 판단할 수 없다. 이번 인터페이스에는 history observation의 source/version/anchor를 보존하고, 목적함수 변경은 별도 objective ID로 이어간다.

Reference512를 모두 사용한다는 것은 전체 bank를 forward만 하고 gradient는 일부에서 계산한다는 뜻이 아니다. 무작위/GSS subsampling, 짧은 문서만 선택, token 위치 축소, top-k KL 대체를 하지 않는다. 조건부 no-op/fallback은 기존 controller 규칙에 따른다. 성능이나 비용 때문에 문서를 분모에서 제거하지 않는다.

## 2. 실제 병목과 변경 우선순위

근거:
- [EN sequential 비용](../../experiment-reports/servers/server4/single-layer-edit-preserving-correction-2026-09-18-v1/sequential-four-r1/en-f-completed-review-r1/diagnostic-report-ko.md)
- [EN controller 비용 원표](../../experiment-reports/servers/server4/single-layer-edit-preserving-correction-2026-09-18-v1/sequential-four-r1/en-f-completed-review-r1/compute.csv)
- [All-token oracle](../../project/run_scripts/single_layer_edit_preserving_correction/alltoken.py)
- [Guard](../../project/run_scripts/single_layer_edit_preserving_correction/binding.py)
- [Invariant](../../project/run_scripts/single_layer_edit_preserving_correction/runtime.py)
- [Sequential 호출 순서](../../project/run_scripts/single_layer_edit_preserving_correction/sequential_runner.py)

|SH4 EN-F S10 기록|초|포함 관계|
|---|---:|---|
|보정 전체|8251.672250|전체 보정 구간|
|Current/Past guard|1676.573320|보정 내부|
|Invariant|3298.760432|보정 내부|
|Objective|812.515334|보정 내부|
|Reference backward|226.295531|Objective와 중첩|
|Geometry|721.487410|보정 timer와 별도 구간|
|새 native fit|2688.872249|B2–B10,9회|
|Program wall|17807.506516|연구 observer 포함|

Guard+invariant는 보정의60.29%다. 해당 시간 전체가 weight 전송이라는 뜻은 아니다. 전송·중복 forward·CPU geometry 각각의 소요 시간은 추가 계측해야 한다. 미분리 잔여 보정 시간을 모두 I/O/hash로 이름 붙이지 않는다.

실제 trial 수는 [2,2,2,2,2,1,1,1,3,4]였다. 8은 최대 후보 수이며 gradient8회가 아니다. 이번에는 상한을2로 바꾸지 않는다. B9/B10에서 기존 Past guard를 통과한 작은 scale을 누락하지 않는다.

우선순위:
1. 동일 CPU endpoint weight의 sequence별 GPU 재전송 제거.
2. Guard·invariant가 같은 endpoint에서 재실행하는 suffix forward와 score_rows 제거.
3. 동일 native와 key에 대한 고정 geometry 통계 재사용.
4. 필요한 위치별 head 계산과 reference teacher 스트리밍 정리.
5. 계측 후 별도 검토: hash/event 비용, head fusion, microbatch, history 기반 방향 변경.

## 3. 유지할 수학과 FP32 수용 의미

W_N은 원 native L4 write의 실제 FP32 endpoint, K_E는 현재 보호 입력의 전체 valid-token L4 key다.

\[
\mathcal D_E=\{D:D=DQ_E\},\qquad Q_EK_E=0.
\]

기존 EN의 한 방향은

\[
G=\nabla_W L(W_N),\quad H=GQ_E,\quad
\chi=\|H\|_F^2,\quad \eta_0=L(W_N)/\chi,\quad \eta_j=2^{-j}\eta_0.
\]

실제 후보는

\[
W_j=\operatorname{FP32}(W_N-\eta_j H),\qquad
D^{actual}_j=\operatorname{FP64}(W_j)-\operatorname{FP64}(W_N).
\]

Native로부터 재물질화하는 순서, 반감, Armijo, signed KL, loss floor, finite 처리와 fallback 의미를 유지한다. 수학적 DK=0만 확인하고 downstream 검사를 제거하지 않는다.

기존 주요 수치 조건:
- ideal DK 상대오차 ≤1e-10.
- actual token별 native 반응 정규화 오차 ≤1e-5.
- actual allowed-space leakage ≤1e-5.
- 보호 logits 최대 차이 ≤1e-3, RMS ≤1e-4.
- 보호 NLL 절대 차이 ≤1e-4, strict/pair 성공 ID symmetric difference0.
- Current/Past new-target 개별 NLL 증가 ≤1e-4 및 기존 성공 ID 유지.

위 한도는 구형↔신형 구현의 동등성 허용량이 아니다. 구현 parity는 먼저 byte-exact/scalar-exact를 요구하고, 불일치가 생기면 원인과 차이를 별도 보고한다. 보호 한도 안에 있다는 이유로 서로 다른 scale/endpoint를 동일 실행으로 인정하지 않는다.

## 4. EndpointSession: weight는 endpoint마다 한 번 준비

### 4.1 현재 중복 경로

score_rows → logits_at → hidden → suffix_hidden → _weight 경로에서 CPU weight의 전체 finite 검사와 GPU 전송이 sequence마다 반복된다. Invariant의 compare_logits는 W_N과 candidate 모두에 같은 경로를 사용한다. KL은 문서 loop 이전에 weight를 GPU로 준비하므로 전송 구조가 다르다.

L4 [4096,14336] FP32 하나는234,881,024 bytes,224MiB다.

### 4.2 제안 인터페이스

새 endpoint_session.py는 다음 객체를 소유한다. 아래 이름은 구현 계약이며 기존 코드에 이미 존재한다는 뜻이 아니다.

    EndpointSession(model_identity, batch_identity, runtime_policy)
      bind_native(cpu_fp32_weight) -> EndpointHandle
      bind_candidate(cpu_fp32_weight, trial_identity) -> EndpointHandle
      evaluate_handle(handle) -> 읽기 전용 GPU FP32 weight
      release_candidate()
      close()

Handle identity:
- tensor header+bytes SHA와 그 convention.
- shape/dtype/device, batch ID, endpoint ID.
- model의 비선택 parameter/buffer/hook/mode epoch.
- CPU owner와 GPU owner의 mutation version.
- source tensor의 외부 writable alias를 공유하지 않는 소유권.
- 데이터 입력 identity 및 observation identity는 아래 별도 key로 결속.

CPU controller의 immutable FP32 endpoint를 authoritative 값으로 유지한다. 입장 때 shape/dtype/finite/SHA를 검증하고 GPU로 한 번 복사한다. 모델의 실제 selected Parameter를 문장마다 바꾸지 않고 기존 absolute-W functional path에 GPU handle을 전달한다.

WN와 현재 candidate 두 slot을 기본으로 유지한다. inference용 weight residency는448MiB다. KL gradient용 leaf와 gradient·suffix graph의 메모리는 별도 계측하며 이448MiB에 포함됐다고 쓰지 않는다. Gradient용 leaf는 inference cache와 분리하고 gradient graph를 후보 observation에 저장하지 않는다.

### 4.3 수명과 무효화

- Session은 batch의 native 확정 이후 열고, physical observer/commit/next native 이전 닫는다.
- Candidate slot 교체 시 이전 candidate observation을 모두 폐기한다.
- CPU/GPU endpoint mutation, model guard epoch, dtype/device, pack/mask/position/teacher identity가 달라지면 hit가 아니다.
- Pointer 또는 tensor._version만으로 외부 alias mutation까지 byte identity가 증명됐다고 쓰지 않는다. 소유권 격리와 기존 필수 hash 검사를 유지한다.
- 예외 시 finally에서 handle·hidden·graph를 해제한다. 실제 Parameter를 설치한 독립 physical 검증은 기존 exact restore 경로를 사용한다.
- OOM·teacher corruption·stale identity는 technical failure다. Native 성공으로 숨기지 않는다.
- 미완료 observation에는 complete marker를 발급하지 않는다. 다음 후보가 partial 결과를 사용할 수 없다.

## 5. ObservationBundle: 한 번 계산한 current 반응을 공유

### 5.1 먼저 hidden과 score만 공유한다

초기 최적화에서는 기존 LM-head 호출 shape를 그대로 둔다.
- Guard: 기존 target-position union으로 head 계산.
- Full-token invariant: 기존 valid positions의16-position chunk로 head 계산.

두 head를 즉시 합쳐 다른 row-shape의 GEMM으로 바꾸지 않는다. FP32 rounding·tie·guard 경계가 달라질 수 있기 때문이다. Head fusion은 이후 독립 parity를 통과한 변경으로만 검토한다.

현재 accepted candidate의 current sequence당 suffix 실행:
1. Guard의 candidate score.
2. Invariant의 candidate score 재계산.
3. Invariant 비교의 candidate hidden.
4. Invariant 비교의 native hidden.
Native 최초 anchor1회는 이4회와 별도다.

새 경로는 native anchor에서 native final-normalized hidden을 보존하고, candidate guard에서 candidate final-normalized hidden을 보존한다. Invariant는 같은 rows/hidden을 사용한다.

|대상|기존|새 경로, cache 수용 시|
|---|---:|---:|
|Native current anchor|C suffix|C suffix, batch당1회|
|Accepted candidate current guard+invariant|4C suffix|C suffix|
|Candidate current score_rows|2회|1회|
|Candidate full-token invariant 범위|모든 valid token|동일|
|LM-head guard/비교 schedule|별도|첫 단계는 그대로|
|Past candidate score|1회|1회, weight handle만 공유|

C는 정확히 같은 current input cache 수다. 위4→1은 이 구간의 suffix 호출 수이며 EN 전체 시간4배 개선 주장이 아니다. Cache eviction/recompute가 있으면 실제 호출 수와 원인을 기록한다.

### 5.2 NativeObservation / CandidateObservation

    NativeObservation:
      endpoint_identity, input_manifest, rows_identity
      anchor_current_rows, anchor_past_rows
      final_hidden_by_current_cache
      geometry_anchor_stats
      complete_coverage

    CandidateObservation:
      endpoint_identity, trial_identity, input_manifest, rows_identity
      current_rows, past_rows, quality_decisions
      final_hidden_by_current_cache
      invariant_stats, complete_coverage

final hidden은 detach한 CPU FP32로 저장하고 문서별 GPU staging한다. KV cache나 full-vocabulary logits 전체를 저장하지 않는다. Host 필요량은 두 endpoint 기준 대략
2 × 4 × hidden_size × Σ current_padded_length bytes이며 actual packs로 preflight 계산한다. 디스크/host bounded cache를 쓸 경우 eviction과 재계산을 계측한다. 숨긴 토큰 축소는 허용하지 않는다.

Coverage identity에는 모든 current input·valid position·score row를 포함한다. Invariant에 전달되는 rows의 endpoint·입력·label·position identity가 일치해야 한다. 다른 arm/과거 batch의 score를 label만 같다는 이유로 재사용하지 않는다.

Runtime.invariant는 이미 계산된 candidate rows와 native/candidate hidden을 받을 수 있게 분리한다. 값이 없는 legacy caller는 기존 경로로 계산한다. 독립 physical parity 경로는 최적화 cache를 우회해야 한다.

### 5.3 수용 순서 보존

    CPU candidate materialization 및 기존 cheap proposal checks
      → full R512 objective
      → Current/Past quality guard
      → actual geometry + full-token functional invariant
      → 첫 통과 후보 수용 또는 기존 backtracking

Heavy invariant를 모든 Armijo 탈락 후보에도 실행하지 않는다. 첫 버전은 current/past 검사 순서도 유지한다. Past-first fail-fast는 거절 사유 coverage가 달라질 수 있으므로 이후 별도 변경으로 계측한다.

## 6. Geometry·key·teacher 재사용 경계

K_E와 W_N이 고정된 batch 안에서는 다음을 캐시한다.
- 기존 key block 순서의 ||K_E||_F².
- 각 key의 max(1,||W_N k||) 분모.
- 동일 ideal proposal에 대해 이미 계산한 null-response numerator·denominator와 결과.

새 candidate의 actual D K_E, actual leakage, FP32 rounding 차이, 실제 logits/NLL 검사는 매번 필요한 경로에서 수행한다. 이전 scale의 실제 오차를 선형 비례로 대체하지 않는다. Allowed projector와 rank threshold는 바꾸지 않는다.

현재 proposal checker는 torch, invariant geometry는 NumPy/SciPy reduction을 사용한다. 같은 수식을 쓴다는 이유만으로 proposal의 scalar를 기존 invariant scalar의 정확한 값으로 대체하지 않는다. 공유 계산의 dtype·block/reduction 순서·backend를 맞추고 각각의 기존 값과 비교해야 하며, 이 동등성이 확보되지 않으면 ideal-response 결과 공유는 미적용으로 남긴다. 고정 native/key 통계의 캐시도 동일 numerical route에서 생성한다.

Reference의 고정 token sequence는 L4 key와 MLP 직전 residual을 batch 간 재사용할 수 있다. 그러나 아래 값은 고정이 아니다.
- 다른 weight의 L4 이후 hidden/KV 및 downstream gradient.
- 다른 current edit 집합에서 얻은 Q_E.
- 다른 teacher prefix의 log-probability.
- 오래된 history target/version의 label과 평가 결과.

Reference K/residual 저장은 문서 단위로 하고, 큰 R512 행렬을 current K_E에 합치지 않는다. Reference cache는 예산 내 host 또는 local mmap로 운영하며 전체 bank의 참여와 resident 문서 수는 구분한다. Dev128 cache는 observer 시점에 별도 로드할 수 있다.

## 7. R512와 generated256의 데이터·손실 계약

### 7.1 입력과 생성

기존 exact R512=S64+Reserve320+AdditionalTrain128을 재사용한다.
- inputs SHA256: 507b202108acc90e10810455f57da60df14e9e9e31ba567c8c75f51cf76afaeb.
- Dev128은 독립 observer.
- 문서 재추출, fixed input 변경, future request/P/N로 reference 선정 금지.
- Prompt129 IDs, raw argmax, lowest-token-ID tie, logits processor 없음.
- max_new_tokens=256, 모델 configured EOS ID를 generation identity에 저장.
- EOS가 실제 나오면 포함하고 종료. 최소256 강제·가짜 EOS 추가 없음.
- W0 생성 y_i와 실제 길이 T_i∈[1,256], stop reason, length-censored 여부 저장.
- W0 및 tokenizer/model revision, prompt SHA, generation policy, y_i SHA, teacher 위치를 결속.

생성 결과는 W0 행동 기준이다. 사실적 정답 인증이라고 부르지 않는다.

### 7.2 정확한 prefix와 위치

\[
s_i=[x_i,y^0_{i,1},\ldots,y^0_{i,T_i-1}],\quad
|s_i|=129+T_i-1,\quad
I_i=\{128,\ldots,128+T_i-1\}.
\]

W0 teacher와 모든 candidate는 같은 s_i와 I_i를 사용한다. T_i=256이면 TF 입력384, 마지막 위치383이며 실제 전체 생성 sequence는385 IDs다. Early EOS 문서도 정상적인 한 문서로 평균에 포함한다.

### 7.3 EN식 목적: 모든 문서·위치·vocabulary

동일 구현 비교의 초기 기준은 EN의 full-vocabulary forward KL을 유지한다.

\[
L_R(W)=\frac1{512}\sum_{i=1}^{512}
  \frac1{T_i}\sum_{s=1}^{T_i}
  \sum_v q^0_{i,s}(v)\log\frac{q^0_{i,s}(v)}{p_W(v\mid c^0_{i,s})}.
\]

q^0는 고정 W0, c^0는 고정 W0-generated prefix다. 문서 평균을 먼저 정의하므로 짧은 EOS 문서를 삭제하거나 긴 문서에 길이만큼 가중하지 않는다. Short final chunk를 다른 chunk와 같은 무게로 평균내지 않는다.

Gradient는 문서 순서와 dtype을 고정하여 전체512개를 누적한다.
- 모든 문서가 objective forward에 참여.
- gradient sweep이면 모든 문서가 backward에 참여.
- 후보 objective도 실제 생성 위치 전부·전체 vocabulary.
- teacher와 gradient에 official P/N·Dev·future edit가 없음.

전체512 gradient를 구한 뒤512개 dense gradient를 저장하거나 pair Gram을 만들 필요는 없다. EN-F는 최종 합산 gradient 하나를 투영한다. GSS 기반 per-reference gradient 비교를 이번 경로에 추가하지 않는다.
- Reference512 중 일부만 backward하는 GSS/활성 subset 정책은 없음.

이 full-KL은 효율 변경을 판별하기 위한 명확한 EN 기준이다. KL 감소의 locality 전달력이 검증됐다는 뜻은 아니다. 이후 judgment 손상 중심 목적/history 목적은 별도 objective ID와 별도 과학 비교가 필요하다.

### 7.4 별도 GeneratedTeacherStore

기존 TeacherStore는 원문257 입력·128 scoring·S64/Dev128 schema를 강하게 검사한다. 이를 일반 wildcard로 느슨하게 하지 않고 GeneratedTeacherStore를 별도 구현한다.
- ragged input/score positions, actual T_i.
- 각 문서의 W0 FP32 logp [T_i,V].
- payload dtype/shape/SHA, generation/capsule identity.
- R512 membership·순서·Dev separation.
- partial teacher와 complete teacher 구분.
- old natural-prefix teacher, old16/128 생성 capsule/cache는 새 data ID로 재사용하지 않음.
- Old raw prompt membership은 재사용 가능하나 generated suffix teacher는 재생성.

논리적인 K/residual/teacher cache hit는 input+teacher+model+policy ID로 정한다. 파일 이름 또는 문서 ID만으로 hit를 허용하지 않는다.

### 7.5 메모리와 chunk

V=128256, EOS 없이 상한까지 생성했을 때:

|Payload|R512|Dev128|합계|
|---|---:|---:|---:|
|보호 위치|131072|32768|163840|
|FP32 full-vocab teacher|62.625GiB|15.65625GiB|78.28125GiB|
|FP32 key+residual, 길이384|13.5GiB|3.375GiB|16.875GiB|

이는 payload 산술이며 peak GPU/host나 전체 디스크 reserve가 아니다. Native·geometry·CP·임시 파일·filesystem 여유는 따로 합산하고 EOS 실제 길이로 갱신한다. 기존16-token storage reserve를 그대로 재사용하지 않는다.

문서별 mmap/streaming으로 teacher 전체를 GPU에 올리지 않는다. 문서 하나의 full teacher도 최대125.25MiB다. 학생/teacher FP64 KL 중간값과 suffix graph는 추가다.

첫 parity 단계는 기존 head/gradient schedule을 유지한다. 이후 R512·256 지원에서 head position chunking이 필요하면:
- 각 chunk의 KL 합을 T_i로 나눠 문서 손실을 구성.
- Head gradient를 final-hidden 좌표에 누적한 뒤 suffix backward를 문서당 한 번 수행하는 경로를 별도 검증.
- 결과를 FP64 CPU 문서별 gradient로 누적 후512로 나눔.
- 전체 문서 backward 수나 all-position/full-vocab 범위 축소로 소개하지 않음.
- 다른 GEMM shape·FP32 gradient 합산 순서에 따른 차이를 기록하며 purededup parity와 별도 평가.

추가 전송 최적화 후보는 문서별 weight gradient의 누적 위치다. 기존 kl은 각 문서의 FP32 gradient를 FP64로 변환해 CPU로 보낸 뒤 누적한다. L4 FP64 gradient448MiB를512번 보내면 한 sweep에서224GiB의 gradient D2H payload다. 모든 backward는 유지하면서 GPU의 FP64 accumulator에 동일 문서 순서로 더한 뒤 마지막에 한 번448MiB를 전송하는 경로를 검토할 수 있다. 이 경우 FP32 누적이나 문서 합산 순서 변경은 하지 않는다. GPU accumulator448MiB와 문서 gradient/temporary peak를 따로 admission하고, 기존 CPU 누적과 loss/gradient/후보/선택 동등성을 검증한다. Weight residency·hidden reuse 뒤의 별도 최적화이며, 해당 payload가 실제 시간 병목이라는 측정 결과로 쓰지 않는다.

### 7.6 생성용 KV cache는 별도 준비 비용 최적화

동일 W0 한 답변 생성 중 KV cache는 검토한다. 후보 weight 간 upper-layer KV 공유는 금지한다.
- 생성 전에 raw-greedy·EOS·tie rule 고정.
- cached/noncached 생성 token ID·EOS·censor 비교.
- 모든 완성 capsule에서 canonical W0 teacher forcing argmax와 저장 y0 일치 검사.
- 불일치 문서를 조용히 버리거나 label을 교체하지 않음. teacher capsule ready를 멈추고 원인 확인.
- KV가 달라지는 numerical route면 생성 identity에 남기고 새로운 capsule로 생성.
- 생성 중 score를 canonical teacher로 바로 사용하려면 TF teacher parity가 별도로 필요. 기본은 완성 prefix의 W0 TF에서 teacher를 확정.

완성 prefix의 canonical W0 teacher-forcing forward 한 번에서 teacher 추출·생성 token parity 확인·L4 key/residual capture를 함께 수행한다. 같은 W0/입력의 준비 forward를 이 세 목적 때문에 각각 반복하지 않는다. 다만 cached-vs-physical의 독립 구현 검증은 이 준비 공유와 구별하여 필요한 범위에서 수행한다.

현재 no-KV의129 prompt·256 생성은 문서당 총65664 input-token 처리다. KV 사용 시 단순 token-linear 입력 기준은129+255=384이며 attention/head 비용까지 같은 배율로 준다는 뜻은 아니다. Setup 가속은 batch correction 가속과 따로 보고한다.

## 8. 수정 대상과 구현 순서

|순서|파일/새 모듈|변경 내용|유지할 것|
|---|---|---|---|
|1|새 endpoint_session.py, alltoken.py|validated GPU handle·수명·전송 계측|absolute FP32 W functional suffix|
|2|새 observation.py, binding.py|hidden/score bundle, score-from-hidden|기존 guard head shape·NLL reduction·ID|
|3|runtime.py, sequential_runner.py, runner.py|guard 결과를 invariant에 전달, WN hidden 재사용|기존 검사의 수용 의미·순서|
|4|geometry.py, runner.py ideal_check|고정 native/key 통계·동일 proposal 검사 재사용|rank·projector·actual 검증|
|5|GeneratedTeacherStore, generated_reference.py, oracle adapter|R512·ragged256 full-KL 데이터 지원|구형 teacher strict schema|
|6|reference generation/teacher tooling|W0 capsule256·bounded teacher/cache·KV parity|전체512·W0 teacher·실제 EOS|
|7|계측·report|method/setup/audit 분리, cache coverage·hit/miss|실패/미측정/재사용 provenance|

최초 변경 범위에서는 optimizer.py의 수치 정책을 바꾸지 않는다. Observation 전달용 optional callback context가 필요하면 기존 callback 의미와 legacy 경로를 유지한다. 해시 비용 자체를 줄이는 수정은 첫 구현에서 제외한다.

새 data ID를 supported라고 표시하려면 입구 require-reference-policy부터 report의 denominators까지 함께 구현해야 한다. 생성 상한 상수 하나만 바꾸고 기존 S64/128 hardcoding을 남겨두면 완료가 아니다.

## 9. 검증 설계: 구현 동등성과 데이터 확대 분리

### 9.1 CPU fixtures

모델 다운로드나 실제8B 실행 없이 검증할 계약:
- 같은 endpoint handle 재사용 시 transfer/finite validation count가 sequence 수에 비례하지 않음.
- CPU alias/GPU mutation, model epoch, teacher/pack/position 변경 시 잘못된 hit 차단.
- 다른 candidate·거절된 후보·부분 observation reuse 차단.
- Gradient graph가 inference bundle에 들어가지 않음.
- Cached/legacy score row의 NLL·strict/pair IDs·max/RMS 합산 일치.
- EOS 길이1/중간/256, 위치128…383, fake EOS 없음, chunk 마지막 짧은 경우.
- R512 누락/중복/순서 변경 및 Dev 혼입 거절. 각 sweep coverage가512와 실제 positions를 만족.
- Teacher mismatch/nonfinite·OOM·evidence write failure가 정상 fallback으로 변환되지 않음.
- Accepted/fallback/duplicate candidate·empty Past·quality rejection 각각의 cache 수명.

테스트는 실제 계약과 stale-state 방지에 집중한다. 구현 함수를 그대로 다시 쓰는 계산 복제 테스트를 동등성 증거로 삼지 않는다.

### 9.2 동일 데이터에서의 legacy schedule 대 optimized schedule

같은 cold W0 batch, 같은 native W_N·K_E·Q_E·reference teacher를 사용한다. Native와 teacher를 공유한 실제 연구 시간, 각 방법을 단독 실행했을 때의 비용을 구분한다.

먼저 기존 S64/corpus128 fixture는 historical regression 용도로만 사용한다. 새로운 주 비교는 같은 R512/generated256에서:
- 기준: 새 데이터 adapter + 기존 중복 실행 schedule.
- 변경: 같은 adapter·loss·native + endpoint/observation reuse.

두 경로 모두512전부 사용한다. 과거 S64 실행 시간과 새R512 실행 시간의 비율을 dedup speedup으로 쓰지 않는다.

검증 항목:
- 전체 loss rows와 분모, gradient/projected gradient/eta0.
- 모든 실제 trial의 FP32 endpoint SHA와 Armijo/quality/invariant 결과.
- 최초 수용 trial·selected weight bytes·fallback reason.
- Reference/Current/Past 입력 ID 및 coverage.
- History/M append 횟수·state 연결, optimizer 내부 history append0.
- Physical/cached 독립 parity·actual FP32 key/logit 검사.

Purededup 경로는 exact candidate/decision/endpoint 일치를 우선 통과 기준으로 한다. 작은 scalar 차이가 있어도 candidate/decision이 달라지면 동등성 실패다. 기존 보호 한도만 통과한 다른 endpoint는 별도 numerical variant로 기록한다. Head chunking/KV route를 추가한 변경의 scalar/gradient 비교는 사전에 별도 허용 정책을 봉인한다.

### 9.3 시간 계측

다음 wall/counter를 구분한다.
- native z/key/write.
- one-time W0 생성·teacher 생성·reference cache.
- endpoint CPU 검증/H2D 횟수·bytes·seconds.
- reference suffix/head/loss/backward·teacher I/O.
- current/past suffix, head, guard, invariant, hidden cache hit/miss.
- geometry proposal/actual/native-anchor stats.
- hash/event/checkpoint.
- 공식 observer와 독립 numerical audit.

새 전송 timer는 _weight 바깥 또는 실제 upload 자체를 감싸서 기존 suffix timer의 사각지대를 없앤다. CUDA events와 wall 경계를 명시하고, 문서마다 불필요한 synchronize를 넣어 benchmark 자체를 바꾸지 않는다.

구조적 기대:
- session의 resident WN+candidate는 고유 endpoint당 한 번 업로드. Gradient leaf/physical 검증 전송은 별도.
- accepted candidate current suffix는 cache miss가 없다면4C→C.
- 이미 계산된 current row 재계산0.
- teacher/objective/gradient coverage는512와 실제 위치 합계 유지.
- 보존 기준·결과가 동일할 때만 시간 개선을 인정.

B1 한 번의 timing을 p50/p90 또는 안정된 배수라고 쓰지 않는다. 수학적 개선 없이 overhead만 줄였는지 확인한 뒤 필요할 때만 동일 계산의 반복 timing을 수행한다. 긴 sequential은 이 문서에서 자동 실행하지 않는다.

## 10. 완료 판정과 후속 연결

구현 완료는 다음이 모두 있을 때다.
1. R512·256 policy와 실제 ragged teacher/capsule 데이터 입구가 결속됨.
2. Endpoint/observation/geometry reuse의 coverage·수명 검증 통과.
3. Legacy schedule 대비 동일 데이터의 candidate·decision·selected endpoint 검증.
4. 전체512 gradient·모든 실제 위치의 objective 참여가 counter로 확인됨.
5. 중복 전송/forward 감소와 setup/online/audit 비용이 실제로 분리 계측됨.
6. 소스·데이터·teacher·cache·report의 version 연결 완료.

현재 이 문서는 위 완료를 주장하지 않는다. 측정된 speedup도 없다.

효율 검증 이후 history를 reference와 같은 보정 목적에 넣을 때에는 최신 유효 target, entry/native/성공 당시 anchor, overwrite registry, history gradient·recency policy를 별도 objective ID로 정의한다. Reference512 전부 사용은 그 단계에도 유지하며 GSS는 reference 축소 수단으로 되돌리지 않는다.
