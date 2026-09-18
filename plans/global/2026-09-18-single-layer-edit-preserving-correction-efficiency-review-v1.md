# EN 계열 계산량 점검과 경량화 제안

작성: 2026-09-18. 상태: 기존 설계·코드·역사적 시간 기록에 기반한 분석. 새 모델 실행이나 실제 속도 측정은 하지 않았다. 기존 correction design/contract/cells v1을 변경하지 않는 별도 제안이다.

## 1. 판단

**계산량을 줄일 여지가 있다. 우선 S64 gradient와 full-token edit lock을 유지하면서 반복 후보 평가를 줄이고, 그래도 비싸면 작은 문서 표본에서 gradient를 구하되 최종 S64 검증은 유지하는 순서가 적절하다.**

추가 z=0은 추가 비용=0을 뜻하지 않는다. 현재 EN-F의 gradient 1회는 64문서 전체를 처리하는 sweep이며, 최악에는 후보 objective forward 8회와 Current/Past 검사가 반복된다. 반대로 아직 EN을 실제 모델에서 실행하지 않았으므로 원본 AlphaEdit보다 몇 배 느리다는 측정 결과도 없다.

효율화가 보존해야 할 핵심은 `D K_E=0`인 단일 L4 correction과 원지식의 functional loss다. 반복 optimizer, 다층 탐색, 추가 z 최적화는 핵심 요건이 아니다. 후보 수나 gradient 문서 수를 줄이는 것은 계산 예산 변경이며, 동일한 endpoint나 성능을 보장하는 동등 변환은 아니다.

## 2. 비용의 기준

실행 가능한 주 방법의 배치 비용을 다음처럼 분리한다.

`T_EN = T_native_L4 + T_key/basis + T_KL_gradient + T_KL_trials + T_final_guards + T_cache/I/O`

W0 teacher/projector의 일회성 준비 비용과 공식 평가·기술 감사 비용은 각각 별도 열로 기록한다. 전체 실험 비용도 함께 공개한다. 최종 후보를 받아들이는 데 쓰이는 forward는 평가 비용으로 빼면 안 된다.

- Original AlphaEdit [4..8]: 일반적으로 요청당 마지막 layer z 한 번을 계산하고 여러 layer에 write한다. 5개 layer라고 요청당 z가 5회인 것은 아니다.
- N4 및 EN-F: 요청당 L4 local-z 한 번. EN-F의 추가 z는 0회지만 별도의 correction 계산이 추가된다. L4와 L8은 역전파 경로가 달라 z 호출 수만 같다고 같은 시간이 아니다.
- 기존 cost_summary.csv의 `AlphaEdit_ORIGINAL`은 이 감사에서 BLUE [4,8]를 뜻한다. 이를 native AlphaEdit [4..8] 시간으로 사용하면 안 된다.

역사적 L4-only fixed10k 기록은 100 batch에 edit 28,456.63초, target 27,532.75초다. 배치당 각각 284.57초, 275.33초이고 target 비중은 96.75%다. 이는 당시 N4 비용 구조를 보여줄 뿐, 신규 runtime의 예산이나 원본 AlphaEdit 대비 EN 속도비를 확정하지 않는다.

근거: [비용 원표](/mnt/raid5/janghj/layer_allocation/empirical/blue_audit_v1/mechanism/cost_summary.csv), [BLUE/L4 감사](/mnt/raid5/janghj/layer_allocation/06_blue_l4_detailed_audit.md).

## 3. 현재 EN이 비싸지는 지점

1. S64 한 sweep은 64문서 × 257 input tokens = 16,448 tokens다. backward는 loss를 점수화하는 8,192 위치에만 국한되지 않는다.
2. 후보마다 S64의 full-vocabulary KL을 계산한다. teacher만 FP32 기준 4,202,692,608 bytes(약 4.20 GB)다. 후보 8개라면 이 forward가 최대 8회 반복된다.
3. Current는 B100 × native context 6개 × old/new 경로로 중복 제거 전 최대 1,200개 sequence다. 모든 token 반응을 확인하는 forward를 반복하면 guard 자체가 상당한 비용이 될 수 있다. 실제 길이·중복 제거 후 크기는 아직 계측되지 않았다.
4. K_E factorization 및 rank 진단도 공짜가 아니다. full-token 제약의 rank가 크면 geometry 비용 또는 허용 공간 고갈이 문제가 될 수 있다.
5. EN-F4는 4 gradient와 최대 24 trial을 쓰는 계산 예산 진단이다. 이를 주 방법 비용과 동일하게 취급하면 안 된다. EN-COV는 neural KL backward 대신 activation 목적을 사용하므로 다른 EN arm과 비용 구조가 다르다.

## 4. 1순위: full-gradient 경량형

임시 비교 이름은 **EN-F/2**로 둔다. 최적화 방향과 목적함수를 먼저 유지한다.

1. 기존과 같은 native L4 write 및 K_E를 사용한다.
2. S64 전체에서 gradient를 한 번 계산하고 같은 Q_E로 투영한다.
3. 기존 Polyak 초기 제안 및 Armijo 조건을 사용하되 objective trial을 최대 2회로 제한한다.
4. 각 제안은 immutable W_N에서 FP32 materialize한다. weight 유한성·actual D의 K_E 반응·허용 공간 이탈을 먼저 검사한다. 비용이 낮은 검사에서 탈락하면 downstream forward를 생략한다.
5. objective 조건을 처음 통과한 후보에만 전체 Current invariant/finite guard와 Past64 guard를 한 번 수행한다. 보호 logits forward와 Current NLL/strict 검사는 동일 forward에서 함께 계산한다.
6. 최종 guard 실패 또는 2회 안에 유효 후보가 없으면 N4로 복귀한다. guard 실패 후 추가 탐색하지 않는다. 기준 W_N의 guard 값은 한 번만 준비하며 그 비용도 포함한다.

FP32 실제 weight 검사, full-vocabulary KL, full-token lock, 과거 요청 guard의 기준은 완화하지 않는다. 수학적 null 조건만 보고 최종 forward 확인을 제거하지 않는다. EN-S/KL-P/CA에 같은 경량 controller를 적용할 때도 각 arm의 최종 guard는 유지한다.

**교환조건:** 기존 8회 안에서는 찾을 수 있었던 작은 step을 놓치거나, 첫 KL 통과 후보의 guard 실패 후 다른 후보를 놓칠 수 있다. 따라서 fallback 빈도·성공 correction 크기·최종 NS/PS를 원형과 함께 비교해야 한다. 이는 무손실 구현 최적화가 아니다.

## 5. 2순위: 작은 gradient + 전체 최종 검증

EN-F/2에서도 backward가 병목일 때만 **EN-F/16g**를 비교한다.

- S64 bank와 최종 objective 정의는 유지한다.
- loss를 보지 않고 정한 공통 seed 순열로 문서 16개씩 순환한다. 네 batch마다 64문서를 한 번씩 gradient에 사용한다. 새 paraphrase 데이터는 쓰지 않는다.
- W_N의 S64 전체 forward를 한 번 수행하되 선택한 16문서에서만 backward한다. loss reduction은 16문서 평균으로 맞춘다. 나머지 48문서는 no-grad로 처리한다.
- 그 16문서 gradient를 Q_E에 투영하고 S16 objective에서 최대 2개 trial을 검사한다. 첫 Armijo 통과 후보 하나만 후속 검증한다.
- 최종 후보를 S64 전체에서 다시 평가하여 W_N 대비 KL의 분해 가능한 감소를 요구한다. 이어 Current/Past 검사를 한 번 한다. 하나라도 실패하면 N4로 복귀하고 추가 탐색하지 않는다.
- S64의 최종 감소는 S16 방향에 대한 full-bank Armijo 보장을 뜻하지 않는다. S16 direction의 분산, 최종 S64 reject 비율, correction 적용률을 보고한다.

16은 첫 계산 예산 비교점이지 이론적으로 최적인 값이 아니다. 이 안은 gradient 추정량을 바꾸므로 EN-F/2보다 과학적 변경이 크다. S64에서의 감소만으로 unseen locality나 PS 보존을 보장하지 않는다. 공식 P/N 및 Dev128은 기존처럼 online 선택에 사용하지 않는다.

## 6. 추가 calibration 작업량의 상한

여기서 F/B는 각각 **문서 한 개의 suffix forward/backward 처리량**이다. GPU 호출 횟수나 전체 모델 FLOPs가 아니다. 한 문서는 input 257 tokens와 KL scored positions 128개다. Current/Past 검사, geometry, cache 준비, teacher 생성, native z, 공식 평가는 이 표 밖이며 별도 합산한다. early stop과 cheap-screen 탈락이 있으면 실제 비용은 더 작다.

|설정|native endpoint에서 calibration|후보 calibration|F 문서 합계|B 문서 합계|최종 후보 Current/Past 검사 상한|
|---|---|---|---:|---:|---:|
|EN-F v1|S64 F+B|최대 8 × S64 F|576|64|8|
|EN-F/2|S64 F+B|최대 2 × S64 F|192|64|1|
|EN-F/16g|S64 F, 그중 S16 B|최대 2 × S16 F + 최종 S64 F|160|16|1|

EN-F/16g는 native S64 forward와 gradient용 S16 forward를 공유한 계산이다. 공유하지 못하면 F=176, B=16이다. 최종 후보가 native S64와 달라 재검증하는 forward는 생략할 수 없다.

설명용으로만 B의 비용을 F의 2배라고 놓으면 calibration 상한은 704 → 320 → 192 문서-F 상당량이다. 원형 대비 각각 54.5%, 72.7% 감소다. **측정된 시간 감소율이 아니며, EN 전체의 가속률도 아니다.** 원형이 첫 trial에서 통과하면 원형은 F=128/B=64이고 EN-F/2의 calibration 비용도 같아진다. 따라서 최대 8회 비용을 평균 비용처럼 보고해서는 안 된다.

## 7. 수학적 목적을 유지하는 구현 최적화

### 7.1 고정 upstream과 teacher 재사용

L4 down_proj만 바꾸므로 고정 입력의 L4 input key는 변하지 않는다. S64의 K 및 W0 reference를 준비해 재사용할 수 있다. 각 후보는 actual D를 이용해 L4 출력을 바꾸고 L5 이후 suffix를 실행한다. 다만 32개 block 중 L5–31이 남으므로 대부분의 downstream 계산이 없어지는 것은 아니다.

기존 [Llama adapter](/mnt/raid5/janghj/.codex/worktrees/odeedit-local-z-independent-audit-20260917/project/run_scripts/single_layer_zflow/llama_adapter.py)는 cached suffix와 필요한 위치의 full-vocabulary head 계산을 이미 구현한다. 따라서 이를 새 절약분으로 이중 계산하면 안 된다. 현재 adapter의 좌표는 기존 writer-space용이므로 EN의 전체 허용 방향으로 그대로 사용할 수 있는 완성 runner라는 뜻도 아니다. 새 구현의 direct/cached forward·gradient·actual-weight parity 검사가 필요하다.

teacher 4.20 GB를 반복 디스크 읽기/재계산하지 않도록 보관하고 transfer를 계측한다. 동일 text라도 다른 prefix/position/mask인 key를 잘못 합치지 않는다. 완전히 동일한 causal prefix만 provenance와 함께 중복 제거한다. 서로 다른 arm의 B2 이후 gradient, z, native endpoint는 공유할 수 없다.

### 7.2 거대한 dense Q를 만들지 않기

F0가 range(I−P_star)의 직교 basis이고, U가 `(I−F0 F0ᵀ) K_E`의 range를 나타내는 직교 basis이면

`Q_E = I − F0 F0ᵀ − U Uᵀ`

`G Q_E = G − (G F0) F0ᵀ − (G U) Uᵀ`.

F0와 U의 직교성이 검증됐을 때의 식이다. 이 방식은 dense 14,336 × 14,336 Q를 매번 만들 필요를 없앤다. 그 크기의 FP64 행렬 하나는 약 1.64 GB다. 원형도 matrix-free 구현을 허용하므로 dense 구현을 기준으로만 가속률을 부풀리지 않는다.

F0는 한 번 준비하고, current batch projected keys를 factorize한다. rank threshold는 기존 `VᵀK_E`의 차원·spectrum 정의와 일치시켜야 한다. 큰 full-coordinate 행렬의 shape로 threshold를 조용히 바꾸거나, 일반 QR/저랭크 sketch를 exact numerical-rank 판정으로 대체하지 않는다. 실제 rank가 커지면 이 방식도 싸지 않다. 원래 native 출력 span으로 correction을 제한하는 것은 별도 방법 변경이며, 유용한 방향을 없앨 수 있다.

### 7.3 감사와 온라인 수용 검사의 구분

AD–FD 검사, 대규모 spectrum 분석, RAND±, 후보가 확정된 뒤 공식 P/N 평가 등은 연구 검증 비용이다. 대표 기술 episode와 지정 checkpoint에서 수행하고 온라인 method 비용과 별도 보고할 수 있다. accepted 후보의 실제 invariant 및 품질 검사는 계속 method 비용에 포함한다. 감사 횟수를 줄였다고 알고리즘이 그만큼 빨라졌다고 주장하지 않는다.

## 8. 처음부터 줄이지 않을 항목

- Native z step을 임의로 줄이지 않는다. 기존 L4 대부분이 최대 Adam step을 사용하지만, 그 이유가 target NLL 부족이라는 근거는 없고 조기 종료 변경의 actual-write/PS 효과도 미확인이다.
- Full-vocabulary KL을 top-k KL로 바꾸거나 scored positions를 크게 줄이는 안을 첫 효율화로 택하지 않는다. loss가 달라지며 suffix 전체 비용은 그대로 남을 수 있다.
- Functional KL을 activation covariance로 대체한 EN-COV는 대조군이다. activation proxy 개선이 실제 보존 개선을 보장한다는 증거가 없어 주 방법의 값싼 대체재로 선정하지 않는다.
- 매 k batch만 correction하는 것은 누락 batch의 arrival damage 및 장기 경로를 바꾸므로 별도 정책이다. 과거 exact-lock을 누적하지 않는 현재 설계에서 단순 무료 절약으로 간주하지 않는다.
- EN arm 수를 줄이는 것은 총 연구비 절약이다. arm 하나의 온라인 계산량 감소와 구별한다.

## 9. 다음 비교의 범위와 판정

먼저 W0에서 시작하는 동일 B100 cold episode에서 N4, EN-F v1, EN-F/2, EN-F/16g를 비교하면 된다. 기존 10개 독립 batch를 사용하면 새 요청 표본을 만들 필요는 없다. 동일 native endpoint를 공유하되 각 방법 단독 실행 환산 시간과 실제 공유 시간은 둘 다 기록한다. 본 문서는 실행 요청이나 기존 전체 arm 실험을 시작하는 지시가 아니다.

시간은 동일 하드웨어·dtype·microbatch·cache 정책에서 동기화해 측정한다. 원본 AlphaEdit도 같은 episode에서 별도로 계측해야 native AlphaEdit 대비 배수를 말할 수 있다. setup 포함/제외 시간, p50/p90 batch latency, peak memory, 실제 F/B tokens, trial 수, guard 수, fallback 원인, geometry와 I/O 시간을 보고한다. 현 단계의 목표는 임의의 속도 숫자가 아니라 **동일 RS/PS 조건의 NS 이득 대 추가 시간** 곡선을 얻는 것이다.

EN-F/2가 원형과 비슷한 보존 효과를 내면 이후 주 controller 후보로 채택하고 새 contract를 봉인한다. EN-F/16g는 그다음이다. 근거는 gradient/guard 절감률 자체가 아니라 독립 observer와 NS, RS/PS, 적용률이다. 작은 표본에서 차이가 유의하지 않다는 이유만으로 성능 동등을 주장하지 않는다. 어느 것도 실제 모델에서 검증된 결과는 아직 아니다.
