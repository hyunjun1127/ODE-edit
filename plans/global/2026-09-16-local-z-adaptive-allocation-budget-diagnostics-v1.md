# Local-z adaptive allocation: 다층 순효용과 누적 보존 비용 진단

2026-09-16. 사용자 후속 질문을 반영한 [W0 일곱-arm 사양](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-16-local-z-adaptive-allocation-design-v1.md)의 계측 보완이다. 실험 시작은 모든 arm의 동일 W0/M0=0이며 7×10=70 batches를 유지한다. Model runner·GPU 제출 결과가 아니다. 새 hard budget·추가 chain·learned router를 도입하지 않는다.

## 결정: 동적 대상과 target 위치

주 동적 변수는 batch별 residual 적용 강도 a4와 추가 L8 write 강도 a8이다. Current/Past 품질 조건 아래 고정 W0-reference 손상을 최소화한다. L4-only가 충분하면 a8=0을 선택하는 것이 정상 결과다. Layer 개수 균등화, norm 50:50, softmax 합1을 목표로 하지 않는다.

Target은 주방법에서 local-z4를 사용하고, L8 후보를 만들 때 각 partial-L4 state에서 fresh local-z8을 구한다. Terminal L8-z는 고정 gate 및 같은 controller의 대조로 유지한다. Local-z가 optimal write나 preservation을 보장한다는 주장은 하지 않는다.

첫 dynamic 구현은 후보를 실제 평가하는 selector다. Learned policy는 이 선택에 실효적인 이득이 있는지 확인한 뒤 candidate 평가 비용을 줄이는 후속 단계다. 이후 가능한 입력은 local residual·실제 write 실현도·batch key 간섭·history·누적 W0 drift다. Raw z만으로 layer의 보존 비용을 알 수 있다고 가정하지 않는다. 네트워크를 추가하는 것 자체를 novelty로 주장하지 않는다.

## 다층이 본질적으로 더 손상되는가

일차 보존 출력 반응은 J4*d4+J8*d8이다. 그 제곱 비용에는 cross term이 있으므로 추가 layer는 기존 손상을 강화하거나 상쇄할 수 있다. 같은 품질·보존 목적에서 d8=0을 허용한 전체 search space는 단층을 포함한다. 그러나 현재 native-direction menu와 greedy selector가 그 전체 최적값을 찾는 것은 아니다.

현재 관측으로 허용할 표현은 '기존 BLUE 정책의 추가 layer가 순 preservation 이득을 만들지 못했다'다. 'Layer를 하나 더 수정하면 언제나 손상이 늘어난다' 또는 'L4 intrinsic capacity가 이미 소진됐다'는 결론은 아니다.

L4 down-projection만 수정하면 고정 token sequence의 입력 K4는 구조적으로 고정이다. L8에서는 L4 변화로 K8가 변한다. A8=W8,t-W8,0, delta_k8=k8,t-k8,0라 하면 module output 차이는 정확히

    W8,t*k8,t-W8,0*k8,0
    = A8*k8,0 + W8,0*delta_k8 + A8*delta_k8.

세 항은 direct weight change, upstream key change, interaction이다. Module 출력의 분해이며 최종 logit/NS의 인과 기여율이 아니다. 작은 L8 weight norm만으로 안전하다고 판단하지 않는다.

## Budget의 세 의미를 분리한다

1. **누적 사용량:** 현재까지 실제로 변한 weight·mapping·출력의 양. 측정할 수 있다.
2. **현재 요청의 편집-보존 교환관계:** 같은 edit 품질에서 각 후보가 만드는 손상. 현재 유한 menu 안에서 측정한다.
3. **허용 상한 또는 남은 capacity:** 허용 손상 기준과 미래 request 분포를 정하지 않으면 하나의 고유 숫자로 알 수 없다.

이번에는 1·2를 계측한다. 임의의 norm cap, epsilon KL 상한, 남은 capacity %를 신설하지 않는다. 기존에 철회한 사전 TV/KL budget을 복원하는 수정도 아니다.

누적 변화 A_l,t=W_l,t-W_l,0와 이번 update d_l,t를 따로 둔다.

| 계측 | 식/정의 | 해석 |
|---|---|---|
| 이번 weight 크기 | ||d_l,t||F | 이번 batch의 실제 parameter 이동 |
| 누적 net 변화 | ||A_l,t||F / ||W_l,0||F | 최초 대비 최종 위치 |
| 경로 길이 | sum_s ||d_l,s||F | 반복·상쇄를 포함한 이동량 |
| 고정 reference mapping drift | E_l,t=mean_j ||A_l,t*k_l,0,j||² | 고정된 표본 key에서 누적 operator 변화 |
| 이번 stored-history 영향 | tr(d_l,t*M_l,entry*d_l,t^T) | 저장된 과거 평균 key에 대한 local perturbation; optional |
| 누적 출력 손상 | D0(t)=mean KL(p_W0||p_Wt) | 고정 reference에서 실제 출력 drift |

Mapping drift에는 cross term이 포함된다. C0=mean_j k0,j*k0,j^T에서

    E(A+d)-E(A) = 2 tr(d C0 A^T) + tr(d C0 d^T).

따라서 norm² 또는 batch mapping cost의 합을 누적 drift로 보고하지 않는다. 후속 update가 이전 변화를 상쇄하면 E가 줄어들 수 있다. 이 C0는 아래 고정 diagnostic sample의 covariance이며 native Wikipedia covariance 자산과 이름·분모를 구분한다.

출력 ledger는 D0(t), delta_D0=D0(t)-D0(t-1), running max D0, 누적 증가 sum max(delta_D0,0), 누적 감소 sum max(-delta_D0,0)를 남긴다. 마지막 두 값의 차이는 D0(t)-D0(0)이다. 증가분 합을 회복 불가능한 capacity 사용량이라고 부르지 않는다.

Stepwise KL(p_Wt-1||p_Wt)의 합은 D0(t)가 아니다. KL은 가산 거리도 아니며 step subdivision에 따라 합이 달라진다. S64 D0와 Dev128 D0를 분리한다. 분포 변화가 NS 정확도 손실이나 전체 pretrained knowledge 손실률과 같은 것도 아니다.

## Layer 진단의 구체 입력과 시점

원 W0의 고정 S64에서 문서마다 이미 선언한128 scored positions 중 두 개를 hash 순서로 선택한다. 우선순위는 SHA256('LZ-ALLOC-DIAG-v1|document_id|scored_position')다. 이128 위치의 L4/L8 down-projection input key와 W0 module output을 저장한다. Token/position/mask/hash를 고정하며 label·loss·future request로 위치를 고르지 않는다.

이 DiagKeys128은 geometry용 작은 표본이다. Online S64 출력 KL은 여전히 전체 선언된 scored positions와 모든 token의 write 효과를 사용한다. 선택된 두 위치를 전체 문서 보존의 증거로 바꾸지 않는다.

매 batch: layer별 실제 d/A norm, path length, 현재 target residual norm과 actual d*K realization, S64 D0와 delta_D0를 기록한다. Gate0이어도 과거 누적 A8가0이라고 가정하지 않는다. Geometry 자체를 selector feature나 penalty로 추가하지 않는다.

B1/B5/B10 selected endpoint: DiagKeys128에서 E_l과 K8 drift 및 위3항을 기록한다. 가능한 geometry 연산은 endpoint 평가에서 capture한 key와 보존 W0 weight로 수행하며 별도 prefix 계산·tensor copy 비용을 기록한다. Full C0 matrix 구성·eig decomposition은 필수가 아니다.

Full q_H는 이미 원 M과 update가 있어도 추가 dense 연산이 클 수 있으므로 optional로 둔다. 실제로 계산한 경우 원 M의 합 기준과 M/n_history의 평균 기준을 따로 보고한다. n_history는 M에 실제 append한 key-column 총수이며 unique/active fact 수가 아니다. B1의 n_history=0이면 mean은 null/NOT_APPLICABLE이다. Sparse sampled past-key perturbation을 사용하면 q_H_full이라고 명명하지 않는다.

Projected history rank나 trace는 capacity가 아니다. Native projector의 rank는 허용 입력 방향 차원이고 M의 유효 rank는 저장된 key 기하의 요약이다. 필요하면 projector idempotence 오차, 요청별 projected-key norm, actual residual realization을 보조 진단하되 이것을 남은 사실 수나 capacity %로 환산하지 않는다.

## 같은 entry에서 추가 layer의 순효용을 측정한다

LD가 이미 만드는 후보 중 C4는 a8=0인 두 후보, C48은 전체6후보다. 동일 Current/Past feasibility 조건을 적용한다. 공통 N4가 양쪽에 있어 정상 batch에서 둘 다 비지 않는다.

    gain8(t) = min_{c in C4 feasible} D0(c)
               - min_{c in C48 feasible} D0(c).

정확 산술상 gain8>=0이며 실제 선택은 기존 수치 tie 규약을 따른다. 이 값은 **같은 현재 모델에서 이번 L8 write 후보를 추가한 가치**다. 이전 batch에 L8를 이미 수정했다면 C4도 그 과거 L8를 포함하므로 독립 lifelong L4-only chain과 같지 않다. 그 비교는 LD-L4D arm 비교로 별도 수행한다.

Gain8이0이면 품질 탈락, feasible하지만 비용이 큼, actual L8 update가 거의0, numerical tie를 구분한다. Gain8>0인 batch에서도 Dev/공식 N으로 일반화하는지 별도 확인한다. 좋은 S64 후보가 있었다는 사실만으로 실제 locality capacity가 늘었다고 주장하지 않는다.

## Selector 실패와 후보 family 실패를 분리하는 observer

LD/TD의 B1/B5/B10에서 candidate 생성·훈련 metric·선택 ID를 먼저 봉인한다. 그 뒤 별도 observer가 deduplicated 후보의 공식 Current P/N을 평가한다. B5/B10에는 첫 B1의100요청을 고정 cohort로 하여 그 P/N도 같은 후보에서 평가한다. Candidate 선택은 다시 수행하지 않는다. 공식 P/N은 target, gate, memory sampling, threshold에 사용하지 않는다.

선택된 endpoint 평가는 기존 매-batch 관측과 중복 사용하지 않는다. 최대 추가 candidate states는 [(6-1)+(7-1)]*3=33개다. Current P/N의 최대 추가 prompt-state는33*1200=39,600개, 첫100 고정 cohort는22*1200=26,400개다. 총66,000은 dedup/기존 관측 재사용 전 상한이다. 이는 두 target의 NLL scoring을 포함한 sequence 수나 forward/token 수가 아니므로 실제 연산량을 별도로 센다. Model editing chains는70 batches 그대로이며 observer 비용이 늘어났음을 별도 ledger에 기록한다.

이 후보들은 한-step 반사실 상태다. Candidate NS를 보고 뒤 batch까지 가상으로 이어 붙이거나 사후 best-N chain을 실제 정책 결과처럼 만들지 않는다. 이 데이터는 개발 분석이므로 이후 정책을 바꾸면 새 정책 버전과 평가 구간을 선언한다.

| 관측 | 다음 해석 |
|---|---|
| LD가 L4D보다 S64·N 모두 개선 | 검사한 다층 후보의 추가 실효성 |
| 후보 중 N이 좋은 다층 write가 있는데 선택하지 않음 | proxy 또는 selector 문제 |
| S64에서는 좋은데 N에서는 악화 | 보존 proxy 전이 문제 |
| 검사한 feasible 다층 후보가 N에서 L4 후보보다 열세 | 해당 local-target/gate family의 한계 |
| L4D만 N4보다 좋음 | 동적 강도 조절은 지지, 다층 분배 기여는 미지지 |
| 후반에 feasible 후보·실현도가 줄고 품질/보존 trade-off가 악화 | 현재 writer/정책의 유효 여력 악화; intrinsic capacity 포화 증명 아님 |

## 이후 learned routing 또는 hard budget을 논할 조건

같은 request라도 history·batch 간섭·출력 drift에 따라 유리한 후보가 바뀌고 그 선택이 독립 reference/공식 N으로 일반화할 때, 이 온라인 선택을 작은 predictor에 distill할 근거가 생긴다. Predictor 입력은 z만이 아니라 관측한 state와 response다. 학습 stream과 검증 stream을 분리하고 실제 candidate screen 대조를 남긴다.

Hard budget이 꼭 필요해지면 D0(t)<=epsilon처럼 기준·단위·허용치를 명시해야 한다. 이때 slack epsilon-D0(t)은 선언된 제약의 여유이지 layer의 intrinsic storage remainder가 아니다. 현재는 epsilon을 임의로 만들지 않고 각 시점의 edit-quality 대 preservation frontier와 비용 ledger를 먼저 산출한다.

1,000-request 진단만으로 10k capacity를 추정하지 않는다. 후속 W0→10k에서 같은 지표·고정 cohort로 누적 양상을 검증한다. 첫 결과가 다층에 불리하면 그 결과를 받아들이고, L4 request별 강도/방향 조절은 별도 가설로 평가한다.
