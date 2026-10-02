# Adam과 subject 위치 δ 주입 유지에 대한 점검

2026-10-02. 사용자 요청에 따른 코드·선행연구 점검이다. 새 실험, production 코드 변경, 기존 method 정본 변경은 하지 않았다.

**결론: Adam을 사용할 수 있다. Subject 위치의 δ 주입도 그대로 유지할 수 있으며, 사용자가 의도한 모든 층 local-z 공동 최적화와 양립한다.** v5의 공통 scalar proximal solver와 모든 토큰에 작용하는 가중치 task graph는 필수 조건이 아니라 기존 설계의 선택이었다. 다음 설계에서 subject 주입은 유지 조건으로 취급한다.

## 1. 실제 native와 선행연구

실제 실행 reference인 `/mnt/raid5/janghj/EasyEdit/easyeditor/models/memit/compute_z.py`의 SHA256은 `6e43c1f03bc03c87dcff70111dff1a93f4e00ea31b4bfeb5e8eab4a516ea7ae4`다. v5 execution lock과 일치한다.

- 137–152행: native lookup 위치의 block output에 δ를 더한다.
- 159행: `torch.optim.Adam([delta], lr=hparams.v_lr)`.
- 216–218행: nonsquared norm `λ ||δ|| / ||h0||²`를 loss에 직접 넣는다.
- 226–240행: native stopping, 마지막 평가와 Adam update의 구분, update 후 norm clamp.

저자들의 [MEMIT 공식 compute_z](https://github.com/kmeng01/memit/blob/main/memit/compute_z.py)와 [ROME 공식 compute_v](https://github.com/kmeng01/rome/blob/main/rome/compute_v.py)도 lookup 위치 δ 주입·Adam·명시적 norm penalty·ball projection을 사용한다. ROME와 MEMIT의 구체적인 module hook은 다르므로 “subject 위치”만 같다고 임의의 module로 옮기지 않는다.

[ROME 논문](https://arxiv.org/abs/2202.05262)은 subject 처리 위치의 중간층 MLP와 factual association 편집에 경험적 근거를 제시한다. 이것은 모든 모델·benchmark에서 항상 최적이라는 보편적 증명은 아니다. [BLUE 논문](https://papers.neurips.cc/paper_files/paper/2025/file/70d4ef44dc973586cfa3ea92b4868b72-Paper-Conference.pdf)의 문제 제기는 주로 최종층에서 계산한 residual의 층간 분배에 있으며, subject intervention을 없애야 한다는 주장은 아니다.

## 2. v4가 이미 구현한 조합

검토 worktree의 `project/run_scripts/jlz_native_joint/` 기준:

- `optimize.py:10–11`: 모든 eligible layer와 요청의 δ를 만들고 하나의 Adam optimizer에 등록.
- `adapter.py:69–85`: 각 층에서 해당 요청의 subject lookup 위치에만 δ를 직접 주입.
- `oracle.py:66–108`: logical batch의 microbatch gradient를 누적.
- `optimize.py:26–29`: native-form norm을 loss에 포함.
- `optimize.py:64–69`: 전체 gradient 누적 후 한 번 Adam step, 각 층·요청별 clamp.

Llama의 해당 block은 residual skip과 down_proj 출력이 더해지는 구조다. 고정된 block 입력에서 subject block output에 δ를 더하는 것은 그 위치 down_proj contribution에 같은 δ를 추가하는 것과 같은 출력 좌표에 놓인다. 단, norm anchor는 native full block output을 유지해야 하며 down_proj 출력 norm으로 바꾸면 다른 규제가 된다. 다른 모델은 native module 매핑을 별도 확인해야 한다.

## 3. 권고하는 학습 골격

모델 W_t를 batch entry에 고정하고 모든 eligible layer의 절대 좌표 δ_(l,r)를 0에서 시작한다. 요청 r의 모든 native context c에서 동일한 δ_(l,r)를 그 context의 subject 위치 s_(r,c)에 주입한다.

\[
h_{l,r,c}[s_{r,c}]\leftarrow h_{l,r,c}[s_{r,c}]+\delta_{l,r}.
\]

직접 주입하지 않은 토큰도 이후 attention 등을 통해 바뀔 수 있다. 이 자연스러운 전달은 허용하고 공동 gradient에 포함한다. 모든 층을 함께 활성화한 forward/backward로 학습하므로 하층 δ가 상층 상태와 손실에 미치는 효과가 포함된다. 층마다 독립적으로 완성된 z를 만든 뒤 가중치만 나누는 방식과 다르다.

Native 주목적은 이 subject-injected 모델에서 NLL, current‖entry KL, 각 층의 nonsquared δ norm을 계산한다. 모든 문장·context/token 평균·lookup·readout은 해당 baseline profile을 유지한다. Adam은 그 공동 목적을 푸는 도구로 사용한다.

이 골격만으로 모든 요청 사이의 실제 shared-weight 간섭까지 포함되는 것은 아니다. 각 요청의 가상 δ는 다른 요청의 forward에 자동 적용되지 않는다. 순차 history와 실제 writer에 따른 안전 비용을 δ 학습에 연결하는 배분 설계가 여전히 필요하다. Adam 그 자체를 배분 정책으로 주장하지 않는다.

## 4. Adam으로 바꿀 때 지켜야 할 의미

1. Native learning rate는 **절대 δ 좌표**에서 해석한다. v5의 상대 좌표 v=D/a에 같은 수치를 적용하면 실제 D의 update 규모가 달라진다.
2. `weight_decay=0`으로 두고 native nonsquared norm을 loss에 직접 포함한다. Adam의 L2 decay나 AdamW decay로 대체하지 않는다.
3. Microbatch마다 Adam step을 하지 않는다. 전체 logical batch gradient를 누적하고 한 번 step한다.
4. 각 층·요청의 own-entry native anchor에 따른 ball로 update 후 clamp한다. 층 사용 수·균등 배분·최소 몫을 강제하지 않는다.
5. Adam은 exact group sparsity나 목적값의 매 step 감소를 보장하지 않는다. 이것은 사용 불가 사유가 아니라 solver 성질의 차이다.
6. 고정 25후보면 최대24 update다. Native의 개별 조기 종료를 joint 문제에 어떻게 확장할지는 별도 정의한다. Batch 평균 loss가 작다는 것을 모든 요청의 성공으로 읽지 않는다.

δ=0에서 native norm의 선택된 gradient는 0이고 task gradient로 활성화할 수 있다. Adam은 좌표별 gradient 규모를 보정하므로 초기에는 여러 층이 동시에 큰 update를 받을 수 있다. 같은 Adam을 쓴다고 층별 부담이 자동으로 적절해지는 것은 아니다.

BS100·5층·4096차원 FP32 δ에 대한 두 Adam 상태는 약16MiB다. 전체 비용의 중심은 optimizer 연산보다 모델 forward/backward다. 기존 subject-prefix 재사용·selected-position head는 유지 가능한 최적화 후보지만 새 writer 보조 경로까지 자동으로 동일 비용이라고 주장하지 않는다.

## 5. 실제 관측이 말하는 주의점

v4 A는 이미 subject 공동 δ+Adam으로 P98.7%를 얻었지만 N62.24%였다. v5 A는 P73.2%, N87.46%다. 두 실행은 graph·geometry·writer·optimizer가 함께 다르므로 Adam만의 인과 효과를 추출할 수 없다.

v4 A의 보관된 fit norm을 own-entry anchor로 나누어 재집계했다. `relative norm >= 0.75−1e−5`를 cap 부근으로 정의하면:

| Batch | cap 부근 group /500 |
|---|---:|
| B1 | 496 |
| B2 | 494 |
| B3 | 483 |
| B4 | 480 |
| B5 | 486 |
| 전체 | **2,439/2,500 = 97.56%** |

이 결과는 native Adam의 전 층 확장이 거의 모든 층·요청을 cap까지 사용하는 상태로 끝날 수 있음을 보여준다. Norm이 실제 의미 변화량이나 locality 손상의 직접 척도라는 뜻은 아니며 Adam 단독 원인도 입증하지 않는다. 하지만 v4를 그대로 복구하면 효과적인 배분 정책까지 복구된다고 주장할 수는 없다.

## 6. subject 주입을 유지하면서 남겨야 할 writer 문제

v5 optimizer만 Adam으로 교체하면 `W+DPᵀ`의 모든 토큰 task graph가 그대로이므로 사용자가 요청한 subject 주입 복원이 아니다. 반대로 v4의 subject 주입+Adam만 복구하면 이전 writer의 실현오차와 locality 문제도 남는다.

v4 `writer.py:109–120`은 current key로 P를 계산해 U_l=D_l P_lᵀ를 쓰고, U_l K_l−D_l을 진단한다. 가상 forward의 절대 local target z를 저장해 실제 prewrite 상태에서 차감하는 writer는 아니다. 이는 기존 incremental δ 정의이지 단순 누락 버그가 아니다.

절대 target을 맞추는 대안을 택한다면

\[
z^{virt}_{l,r,c}=h^{virt,pre}_{l,r,c}+\delta_{l,r},\qquad
E_{l,r,c}=z^{virt}_{l,r,c}-h^{actual,pre}_{l,r,c}
\]

를 구분해야 한다. E는 context마다 달라질 수 있으며 하층 실현오차의 보상도 포함한다. Full block z에서 down_proj 출력만 빼면 residual skip을 잘못 처리한다. E를 사용한다고 실현이나 보존이 보장되는 것도 아니므로 새로운 writer 정의와 비용이 필요하다.

따라서 다음 설계에서 고정할 방향은 **native subject 주입·Adam·전 층 공동 δ**다. 추가 연구 대상은 **이 δ가 실제 writer에서 얼마나 실현되고 다른 요청·과거 지식에 어떤 비용을 만드는지 학습 단계의 배분에 반영하는 연결**이다. 주 task를 다시 전 토큰 weight loss로 바꾸지 않고, writer 실현 및 보존 항의 역할을 구분해야 한다. 구체적 보조 항·계수·refresh 빈도·계산 예산은 아직 확정하지 않았으며 이 문서는 새 method의 완료 명세가 아니다.
