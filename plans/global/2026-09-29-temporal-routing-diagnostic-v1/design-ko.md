**저장된 AlphaEdit BASE checkpoint에서 이어가는 BS1 진단 — 2026-09-29**

저장된 BASE AlphaEdit의 초기·중기·후기 상태를 복원하고, 각 상태에서 같은 새 edit들을 한 층에 순차 누적한다. 기존 1,000/5,000/9,000개 편집은 재실행하지 않는다. 이 문서는 이전에 제안했던 W0부터의 random-routing run과 소수 edit fork 설계를 대체한다.

현재 완료 범위는 설계, checkpoint metadata 연결, 입력 ID 선정, 실행·weight 저장 표 생성이다. 모델 forward·편집·GPU job 제출·새 모델 weight 저장은 수행하지 않았다. 생성 스크립트는 실행 runner가 아니다.

**실험 구성.**

| 출발 checkpoint | 기존 누적 edit | 독립적인 continuation 분기 | 분기당 추가 편집 |
|---|---:|---|---:|
| BASE B010 | 1,000 | L4 / L5 / L6 / L7 / L8 단독 기록 | 동일 100개, BS1 |
| BASE B050 | 5,000 | L4 / L5 / L6 / L7 / L8 단독 기록 | 동일 100개, BS1 |
| BASE B090 | 9,000 | L4 / L5 / L6 / L7 / L8 단독 기록 | 동일 100개, BS1 |

총 15개 분기, 추가 native fit 1,500회다. 각 분기는 checkpoint 전체를 복원한 뒤 100개를 실제로 누적한다. L4 분기의 끝에서 L5 분기를 시작하지 않는다. 매 edit의 목표는 해당 분기의 한 층에 전부 기록하고, scale=1로 고정한다.

판별할 질문은 어느 층을 반복해서 쓸 때 추가 손상이 커지는지, 이를 누적 오차·현재 입력·실제 출력 중 무엇이 설명하는지, 앞층 편집이 뒤층의 기존 기록에 들어가는 입력을 바꿔 망각을 일으키는지다.

Step 1은 같은 checkpoint·요청의 층간 비교다. Step 2 이후에는 분기별 모델과 history가 달라지므로 **서로 다른 누적 경로의 비교**다. 이를 같은 live 상태에서 다음 한 번의 최적 기록층을 식별한 결과라고 표현하지 않는다. 이번 실험은 temporal routing 정책에 필요한 상태의 진단이다.

**공통 입력과 보호 패널.**

Source ordinal 9000 이후에서, 앞 9,000개와 정규화 subject 문자열이 겹치지 않는 서로 다른 subject 100개를 선정한다. Paraphrase 2개·neighborhood 10개 이상과 다른 새 목표답이라는 metadata 조건만 적용하고 통과 사례의 원래 상대 순서를 유지한다. 모델 성공 여부로 선별하지 않는다. 모든 15개 분기에서 동일 순서를 사용한다.

이는 checkpoint별 원래 다음 batch가 아니라 출발 상태를 비교하기 위한 공통 continuation이다. Subject 문자열 분리는 alias·의미적 독립성까지 보장하지 않는다. 출발 checkpoint는 원래 BS100 실행에서 형성됐으며, 새 continuation만 BS1이다.

| 패널 | 크기 | 용도 |
|---|---:|---|
| Base sensor | 공통 16개 | 원본 행동에 대한 local response 및 출력 신호 |
| Base observer | 공통 32개 | 별도 입력에서 실제 추가 손상 확인 |
| History sensor | checkpoint당 8개 | 기존 edit의 유효 목표 보존 신호 |
| History observer | checkpoint당 16개 | 기존 edit 상실·회복 확인 |
| Continuation | 공통 100개 | 현재 성공 및 이후 망각; canonical + paraphrase 2개 |
| Neighborhood | 현재 edit당 고정 10개 | 상세 시점에서 가까운 사실의 locality |

Base panel은 앞 9,000개 및 continuation과 subject가 겹치지 않는 reserve에서 선정한다. History는 checkpoint까지의 최신 유효 subject–relation 기록 중 oldest/newest quarter를 절반씩 사용한다. Subject hash로 sensor/observer 역할을 전체 checkpoint에 걸쳐 분리하며 checkpoint 안에서는 subject 중복을 막는다. ID는 [panel-ids.csv](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-29-temporal-routing-diagnostic-v1/panel-ids.csv)에 고정했다.

기존 history는 **checkpoint 진입 시 성공했던 사례**의 이후 상실률을 보고한다. 원래 at-write 성공을 관측했다고 가정하지 않는다. 진입 시 실패와 이후 회복은 별도로 표시한다. 새 100개는 실제 at-write 성공을 기록하고 그 성공 집합에 대한 retention과 acquisition failure를 함께 보고한다.

이 작은 factual panel은 EN-R512-G256의 대체 reference나 범용 능력 평가가 아니다. Paraphrase와 observer를 writer 학습에 사용하지 않는다.

**복원 및 writer.**

- BASE의 다섯 edited weight와 다섯 native history Gram을 모두 복원한다. W0는 원본 행동 비교 reference로 유지한다.
- Native AlphaEdit `layers=[ℓ]`, `blue=false`, `L2=10`, scale=1, FP32/eager를 사용한다. Target은 선택층에서 다시 계산하고 target cache는 끈다. 단일층이므로 residual divisor는 1이다.
- `v_num_grad_steps=25`, `v_lr=.1`, decay=.5, clamp=.75, KL factor=.0625, TF32 matmul=false/cuDNN=true 및 원실행 native context를 유지한다.
- 저장 P/M의 원래 slot은 `physical_layer−4`다. Singleton에는 해당 slice를 slot 0으로 연결한다.
- 매 write 뒤 **선택층의 native history append를 정확히 한 번 수행**한다. 다른 층의 Gram은 그대로 둔다. 분기 안에서 기록층을 바꾸지 않으므로 all-layer history 갱신을 새로 도입하지 않는다.
- Finite update의 기능적 실패도 weight/history에 남기고 기록한다. 다른 층 재시도나 성공 사례 교체는 하지 않는다. NaN 또는 복원·원본성 검증 실패는 기술 실패로 중지한다.
- 다음 분기 전에 checkpoint W/M 및 결속된 RNG/context 상태를 복원한다. 선택층 이외의 parameter는 분기 진입 상태와 같아야 한다.

기존 [singleton adapter](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-18-sequential-local-z-allocation-review/executed-source/native.py)는 BLUE/L2=1을 강제하므로 그대로 쓸 수 없다. [확인한 native 본체](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-22-alphaedit-original-failure-audit/official/AlphaEdit/AlphaEdit_main.py)에 맞춘 L2=10 singleton 연결과 계측이 필요하다. Native target/solve 식은 유지한다.

**측정 시점과 세 reference.**

상세 시점은 **0, 1, 10, 25, 50, 100**이다. n=0은 분기 진입 상태다.

| 주기 | 측정 |
|---|---|
| 매 edit | Target/anchor/value norm, 실제 weight delta norm, 선택층 A/B/C, current canonical·두 paraphrase의 NLL/margin/target-prefix strict, sensor의 직전→직후 출력 변화 |
| 상세 시점 | Fixed base/history observer, 누적 continuation 전체 retention, 현재 edit neighborhood, 선택층·뒤층 key/readout 및 MLP/residual norm |
| 상세 시점 write 직전에도 | 같은 fixed observer를 평가하여 마지막 한 write의 추가 손상과 구간 누적 손상을 구별 |
| L4 분기 n=100 | 뒤층 historical-delta patch; target refit 없이 forward만 수행 |

Sensor는 고정 패널이므로 직전 post-write 결과를 다음 pre-write로 재사용할 수 있다. 모든 key 비교는 같은 teacher-forced prefix·valid token 위치를 사용한다. 생성으로 바뀐 prefix를 key drift에 섞지 않으며, teacher-forced strict나 NLL margin을 자유 생성 정확도라고 부르지 않는다.

W0→checkpoint는 이미 존재하던 손상, checkpoint→n은 이번 continuation의 누적 변화, n−1→n은 한 write의 추가 변화다. Base의 original-teacher KL·true-answer NLL/strict와 history의 유효 새 목표 NLL/strict를 각각 보고한다. 과거 edit 보존을 W0 출력 복원과 동일시하지 않는다.

**위험 신호 A/B/C/F.**

선택층의 write 직전을 Wt, 실제 write를 d, 원본을 W0라 하자. Base sensor의 같은 입력에서 얻은 key를 K0/Kt, E=Wt−W0, 질문별 공통 분모를 s=||W0K0||²_F+ε로 둔다. Raw 값과 질문별 정규화 평균을 함께 저장한다.

\[
A=\mathbb E_q\frac{\|dK_0\|_F^2}{s_q},\qquad
B=\mathbb E_q\frac{2\langle EK_0,dK_0\rangle_F+\|dK_0\|_F^2}{s_q},
\]
\[
C=\mathbb E_q\frac{2\langle W_tK_t-W_0K_0,dK_t\rangle_F+\|dK_t\|_F^2}{s_q}.
\]

A는 새 write의 원본 입력 반응, B는 누적 mapping 오차와의 정렬까지, C는 live 입력까지 포함한다. B/C는 음수일 수 있다. 이들은 local mapping 진단이며 실제 기능 손상의 보증은 아니다.

F는 write 전후의 sensor 출력 변화다. Base는 KL(p0||p_after)−KL(p0||p_before), history는 유효 목표 NLL 증분을 따로 쓴다. KL은 정해진 answer-prediction 위치에서 token mean 후 question mean으로 집계한다. F는 실제 write 적용 후 얻는 진단이며 무료 사전 routing score가 아니다. Base/history를 임의의 한 가중합으로 합치지 않는다. Sensor 신호는 별도 observer의 실제 손상과 대조한다.

**고정층 분기의 해석:** 선택한 MLP output weight 앞의 parameter는 continuation 중 바뀌지 않는다. 동일 입력에서 `K_selected,n = K_selected,entry`여야 한다. 선택층의 Kt−K0는 BASE checkpoint에서 물려받은 차이다. 이번 continuation의 새로운 입력 이동은 선택층보다 뒤에서 관측한다. 이 불변성은 계측 점검에도 사용한다.

A→B는 live 누적 오차 정렬의 추가 정보, B→C는 이 설계에서는 물려받은 입력 변화의 추가 정보다. 이를 선택층 입력이 매 write마다 움직이는 현상으로 설명하지 않는다. Target/value inflation 및 일반 입력에서 MLP output과 residual의 norm 비중도 별도 설명 후보로 추적하되 norm 증가만으로 원인을 확정하지 않는다. 문헌 근거는 [보존 연구 기반 손상 신호 메모](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-29-knowledge-editing-survey/damage-signals-from-preservation-studies-ko.md)에 정리돼 있다.

층별 새 편집 품질을 항상 함께 표시한다. Canonical strict·두 paraphrase strict가 같고 canonical mean NLL 차이가 0.1 이하인 결과를 보조적으로 비교할 수 있지만 이를 맞추기 위한 strength tuning은 하지 않는다. 편집이 약해 보존만 좋아진 경우는 개선으로 결론내리지 않는다. Step 2 이후는 경로 비교라는 한계를 유지하고, 상관된 step·층쌍을 독립 반복으로 세지 않는다.

**작은 개입: 앞층 편집이 뒤층의 기존 기록을 다르게 작동시키는가.**

각 checkpoint의 L4-only 분기 n=100에서 L8을 본다. W8은 변하지 않았지만 그 입력은 달라질 수 있다. BASE가 저장한 변화 D8=W8_checkpoint−W8_original에 대해 live MLP8 output에 다음 항을 더한다.

\[
\delta m_8=-D_8(K_{8,100}-K_{8,entry}).
\]

기존 downstream weight 변화가 이번 입력 이동 때문에 다르게 작용하는 성분만 제거한다. W0,8의 새 입력에 대한 반응은 유지한다. Native endpoint, zero hook, 목표 patch, 같은 위치·같은 norm의 고정 seed 무작위 방향 patch를 비교한다.

Base/history observer 회복과 새 100개 retention을 동시에 평가한다. 과거답을 회복하면서 새 edit가 사라지면 보존–편집 교환이다. L8-only 분기는 K8이 고정되어 위 patch가 0이어야 하는 음성 대조다. 최대 3건, 기존 native endpoint 외 추가 패널 pass는 사건당 3회이며 target refit은 없다. Zero-hook parity 실패 시 해석하지 않는다. 이는 hybrid forward에서의 조건부 인과 근거이며 유일한 원인의 증명이나 배포 가능한 복구 정책이 아니다.

**사후 진단을 위한 실제 weight 저장.**

모든 분기에서 **n=50과 n=100 직후 두 번만** 저장한다. n=0은 기존 BASE checkpoint를 참조한다. 총 30개 신규 weight snapshot이며, 손상 결과를 보고 유리한 시점을 골라 저장하지 않는다. 다른 상세 측정 시점에는 scalar·평가·정해진 capture만 남긴다.

각 snapshot에는 선택층의 **실제로 적용된 full FP32 weight tensor** `[4096, 14336]`를 저장한다. Low-rank factor·norm·누적 delta만 저장하는 것으로 대체하지 않는다. 선택층 외 모든 parameter가 BASE 진입 상태와 같다는 조건에서 다음 순서로 해당 시점의 전체 모델을 복원한다.

1. 고정된 원본 모델 revision을 로드한다.
2. 해당 BASE checkpoint의 다섯 edited weight를 적용한다.
3. snapshot의 선택층 weight를 덮어쓴다.

이렇게 복원하면 새로운 prompt, activation, spectral 진단을 사후에 실행할 수 있다. 원본 모델과 세 BASE archive는 snapshot의 필수 부모이므로 보존해야 한다. SHA manifest만 남기고 부모 payload를 삭제하면 복원할 수 없다.

Snapshot마다 branch/checkpoint/step, parameter name, shape/dtype, parent model revision 및 checkpoint SHA, 실제 저장 파일 SHA, 마지막 case ID와 누적 edit ID 목록, writer 설정·코드 hash·context/tokenizer identity를 metadata로 남긴다. 저장 시 실제 tensor를 독립된 CPU copy로 만든 뒤 임시 파일에 기록하고 atomic rename한다. 저장 후 재로드하여 tensor의 bitwise 일치와 원래 모델 출력 parity를 확인한다. Patch 실험은 저장된 native weight를 변경하지 않는다.

선택층 이외의 weight 불변성도 검증한다. 실패 시 이 overlay snapshot을 완전한 모델 상태로 인정하지 않고 해당 분기를 기술 실패로 처리한다. Full model 전체나 모든 layer weight를 매번 중복 저장할 필요는 없다.

Tensor당 234,881,024 bytes(224 MiB), 30개 합계 7,046,430,720 bytes(**6.5625 GiB**)이며 파일 header·metadata는 별도다. 출발 BASE archive의 기존 용량은 신규 저장량에 포함하지 않는다. Weight 저장 표는 [weight-snapshots.csv](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-29-temporal-routing-diagnostic-v1/weight-snapshots.csv)에 고정한다.

이 저장물은 **모델을 복원하여 사후 분석하는 용도**다. 중간 시점부터 optimizer/editor를 동일하게 재개하려면 그 시점의 history Gram·RNG 등 추가 editor state가 필요하다. 이번 weight snapshot만으로 정확한 편집 재개까지 보장한다고 표현하지 않는다. 이미 측정한 직전→직후 결과도 계속 보관하지만, 저장하지 않은 모든 중간 모델을 무재실행으로 복원할 수 있는 것은 아니다.

**실행 준비와 결과 판단.**

최종 결과는 checkpoint별 다섯 층의 편집 성공/과거 retention/원래 행동 곡선, 추가 손상과 A/B/C/F 대조, 뒤층 key 이동과 historical-delta 반응, 최대 세 patch 비교표다. 누적 상태에 따라 위험층이 바뀌는지와 어떤 상태 신호가 필요한지 판단한다. 실제 switch나 edit별 최적 routing의 우위는 이 실험만으로 확정하지 않는다.

Scientific fit 1,500회, 기술 parity 중복 최대 2회를 포함한 상한 1,502회다. Native optimizer 기준 최대 loss evaluation 37,550회, Adam update 36,048회다. Panel evaluation·history capture·I/O 비용은 별도다. Weight 저장은 추가 target fit을 요구하지 않는다. 불명확한 결과에 따라 대규모 sweep을 자동 추가하지 않는다.

데이터/source lock hash와 모델 snapshot 네 shard의 로컬 존재를 확인했다. BASE B010/B050/B090의 경로·크기·기록 hash는 [checkpoint-bindings.csv](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-29-temporal-routing-diagnostic-v1/checkpoint-bindings.csv)에 연결했다. 기록상 보관 host는 server2이고 현재 host에서는 해당 경로가 보이지 않는다. 보관 host payload를 이번 설계에서 검증했다고 주장하지 않는다.

Archive W/M의 실제 결속, 원실행 P/context/tokenizer/dependencies 연결, L2=10 continuation controller·계측·weight 저장 구현은 실행 준비에 남아 있다. 저장 checkpoint를 새로 만드는 작업은 필요하지 않다. [contract.json](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-29-temporal-routing-diagnostic-v1/contract.json), [fit-cells.csv](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-29-temporal-routing-diagnostic-v1/fit-cells.csv)에 설계를 고정했다. [design-checks.json](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-29-temporal-routing-diagnostic-v1/design-checks.json)의 PASS는 설계 구조 점검이며 모델 실험 통과를 뜻하지 않는다.
