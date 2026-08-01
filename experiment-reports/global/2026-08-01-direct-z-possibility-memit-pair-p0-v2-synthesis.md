# MEMIT direct-z 가능성 진단 — p0 v2 pair synthesis

## Pair-level 결정: 공통 가능성, 비공통 BF 방향성

이 문서는 두 모델의 raw utility를 합산하거나 평균내지 않는다. 고정된 fresh
8-case/two-snapshot panel에서의 모델별 analyzer verdict는 다음과 같이 **그대로
분리**된다.

| 모델 | 기술 유효성 | oracle ceiling | synchronous cone feasibility | 현재 BF 방향성 | analyzer verdict |
|---|---:|---:|---:|---:|---|
| `llama3-8b-inst` | 8/8, valid | true | true | true | `CONE_FEASIBLE_CURRENT_BF_GAP` |
| `qwen2.5-7b-inst` | 8/8, valid | true | true | false | `CONE_FEASIBLE_NO_DIRECTIONAL_BF_GAP_OBSERVED` |

즉, pair synthesis의 공통 결론은 “두 architecture 모두에서 bounded
synchronous response cone이 frozen target 쪽으로 이동할 수 있다”는
**가능성/국소 controllability**뿐이다. 현재 four-hop BF가 그 기하를 일관되게
더 잘 활용한다는 pair-level 결론은 없다. 기호로 쓰면, 모델 집합
`M={Llama,Qwen}`에 대해

\[
\forall m\in M:\ O_m=1\ \land\ C_m=1,\qquad B_{\mathrm{Llama}}=1,\quad
B_{\mathrm{Qwen}}=0.
\]

여기서 `O`, `C`, `B`는 각각 oracle ceiling, cone feasibility, current-BF
directional-gap flag이다. 따라서 `B`의 공통 참값도, pair utility도 정의하지
않는다.

## 범위, audit, 용어

- 이 합성은 current direct-z spec, 각 모델의 p0 v2 analysis report, 해당
  sanitized analyzer JSON/MD, 그리고 parent Slurm job `15740`의 `sacct`
  행만 사용했다. raw outcome/feature/action, 구현, 다른 report는 읽거나
  해석하지 않았다.
- 두 analyzer는 모두 48 outcomes, 실패 0, ITD `8/8`, direct-z
  once-per-case, receipt-before-outcome, scalar firewall, exact rollback을
  기록한다. Gate는 positive mean과 strict majority `5/8`이며 bootstrap CI는
  설명용이다.
- parent `15740` (`odeedit_dzf_pair_v2`)는 `COMPLETED`, `0:0`, `02:47:34`,
  16 CPU/2 GPU/130000M, node `devbox`로 종료됐다. 이는 실행 provenance일
  뿐 과학적 효과의 근거가 아니다.
- 사양상 `BF`는 `bf_current_refreshed_k4`이고, 정식 arm ID로서의 `DF`는
  없다. Qwen report가 부르는 “DF cone”은
  `sync_z_cone_oracle`의 별칭이다. 아래의 “BF vs cone(DF-cone)”은 이
  **비배포성 diagnostic ceiling**과의 대비이지 두 deployable method의
  ranking이 아니다.

## direct-z baseline과 실제 write fidelity

주 representation scalar는

\[
r_z(W_a)=
\frac{\lVert h_8^{\mathrm{canonical}}(W_a)-z^*\rVert}
{\lVert z^*-h_8^{\mathrm{canonical}}(W_0)\rVert},
\qquad r_z(W_0)=1.
\]

`oracle_do_z`는 layer-8 subject position에 frozen delta를 더할 뿐
weight를 쓰지 않는다. 따라서 oracle의 거의 완전한 residual 제거는
**activation-level ceiling**이지 actual-write fidelity의 증거가 아니다.
Actual endpoint write로 읽을 수 있는 것은 C-budget 안의 cone, native
alpha, BF arm들이다. Cone은 BF endpoint C cap 아래 W0의 다섯 unit-C
synchronous layer response로 만든 non-negative ridge/L2-ball diagnostic
해이며, 개략적으로

\[
\Delta W_{\mathrm{cone}}\in
\Big\{\sum_{\ell=4}^{8}a_\ell U_\ell(W_0):a_\ell\ge0,\ \lVert a\rVert_2
\text{ is C-capped}\Big\}
\]

에서 direct-z residual을 줄이는 방향을 찾는다. 이는 deployable controller가
아니다.

| 모델 | 무가중치 oracle: no-op 대비 z gain / output-NLL gain | actual cone write: no-op 대비 z-residual gain (95% CI; cases) | cone 후 잔여 residual의 근사치 | cone output-NLL gain | cone preservation gain |
|---|---|---|---:|---:|---:|
| Llama | `1.0000` / `10.4220` (8/8) | `0.5603` (`[0.5376, 0.5818]`; 8/8) | `~0.4397` | `9.4744` (8/8) | `-0.4034` (0/8 positive) |
| Qwen | `1.0000` / `10.5401` (8/8) | `0.4475` (`[0.4000, 0.4923]`; 8/8) | `~0.5525` | `10.4881` (8/8) | `-1.0369` (0/8 positive) |

`preservation_score=-heldout_kl`이므로 마지막 열의 음수는 preservation
proxy 악화다. Cone-to-oracle residual gap도 Llama `0.4397`, Qwen `0.5525`
(각 8/8)여서, 실제 write가 oracle target을 완전히 실현한 것은 아니다.
그럼에도 two model 모두에서 actual C-bounded write가 no-op보다 target에
가까워지고 output-NLL도 개선된다는 점은 oracle-only 현상과 구별되는 핵심
신호다. 동시에 두 모델 모두 cone의 preservation은 악화하므로 fidelity를
quality의 단조 대리변수로 읽을 수 없다.

## BF, cone(DF-cone), native-alpha의 비교

`native_alpha_c_matched`와 BF의 endpoint C-energy는 두 모델 모두 matched다.
따라서 아래 BF-native 차이는 서로 다른 endpoint C-budget 때문으로
설명할 수 없다. 수치는 analyzer field의 원래 comparator 방향을 보존했다.

| 축 / scalar | Llama | Qwen | pair synthesis에서 허용되는 읽기 |
|---|---:|---:|---|
| BF−cone direct-z objective, `z_objective_gap_bf_minus_cone` | `+0.0422`, 6/8 positive | `-0.2575`, 0/8 positive | **방향 불일치**: Llama만 current-BF directional gap이 열리고 Qwen은 닫힌다. |
| cone−BF output objective | `-0.8099`, 0/8 positive | `-0.0450`, 0/8 positive | 원래 부호상 BF output objective가 cone보다 높다. 이는 direct-z fidelity 우월의 증명이 아니다. |
| BF−native-alpha z gain | `-0.1350`, 0/8 positive | `-0.0546`, 2/8 positive | 두 모델 모두 matched-C native control보다 BF의 direct-z gain이 낮다. |
| BF−native-alpha output-NLL gain | `+0.1607`, 8/8 positive | `+0.000021`, 2/8 positive | Llama의 edit proxy 상승과 Qwen의 사실상 mixed/no-clear output 차이를 분리한다. |
| BF−native-alpha preservation gain | `-0.0963`, 0/8 positive | `-0.0870`, 2/8 positive | 두 모델 모두 mean상 preservation proxy가 낮다. |
| BF−native-alpha Frobenius norm | `+0.0409`, 8/8 positive | `-0.8302`, 1/8 positive; CI crosses 0 | norm-efficiency sign도 architecture에 따라 다르며 공통 이점이 아니다. |
| BF−cone off-token spill | `+0.0642`, 8/8 positive | `+0.0275`, 8/8 positive | BF가 두 모델 모두에서 full-sequence off-token spill이 더 크다. |
| BF−cone generated mean error | `+0.0560`, 7/8 positive | `-0.2394`, 0/8 positive | generated-context fidelity는 서로 반대 방향이다. |

이 표는 edit, direct-z fidelity, preservation, norm, spill을 하나의 utility로
압축하지 않는다. 특히 Llama에서는 BF가 matched native alpha보다 output
NLL/progress/margin을 8/8에서 높이면서 z gain과 preservation을 8/8에서
낮췄다. Qwen에서는 BF가 cone보다 output objective와 margin에서 유리한
부호를 보이지만, cone direct-z objective보다 낮고 matched native alpha보다
z gain/preservation도 낮다. Qwen의 paraphrase-only gain (`+0.2092`, 7/8)은
이 primary fidelity/preservation 결과를 뒤집지 않는다.

따라서 BF의 output-side 반례는 “frozen target을 더 잘 따라야만 edit가
개선된다”는 단순 가설을 깨지만, BF 또는 ODE-Edit의 일반적 우월성을 만들지
않는다. Llama의 positive BF directionality 역시 Qwen의 negative directionality와
서로 상쇄하거나 평균낼 수 없는 architecture-conditional 결과다.

## Alpha 후속 실험으로 넘길 수 있는 ODE signal

넘길 수 있는 것은 controller 우월성이 아니라, fixed C cap 아래의
**reachable-set signal**이다.

\[
\forall m\in\{\mathrm{Llama,Qwen}\}:\quad
r_z(W_{\mathrm{cone},m}) < r_z(W_{0,m})
\quad\land\quad
\Delta\mathrm{NLL}_{\mathrm{cone}-\mathrm{noop},m}>0.
\]

즉 Alpha follow-up은 “frozen target이 실제 bounded write space에서 부분적으로
도달 가능하다”는 기하적 가설을 시험할 근거는 받는다. 하지만 다음은
전달되지 않는다.

\[
\text{cone reachable} \not\Rightarrow
\text{BF trajectory exploits it},\qquad
\text{oracle/cone diagnostic} \not\Rightarrow
\text{deployable ODE editor}.
\]

그러므로 Alpha 평가에서는 모델별로, 같은 direct-z lineage와 명시된 C-budget
아래에서 (i) actual-write `r_z`, (ii) edit/output, (iii) `-heldout_kl`
preservation, (iv) low-rank Frobenius norm, (v) off-token spill을 native
control 및 no-op과 분리해 보고해야 한다. Alpha가 실제로 새 signal을 보이려면
그 다섯 축의 paired contrast를 공개해야 하며, oracle 또는 cone의 ceiling을
자신의 성능으로 재명명해서는 안 된다. 특히 Llama/Qwen을 합산한 acceptance
score로 BF 불일치를 가릴 수 없다.

## lifelong 및 method-superiority 비보장

이 panel은 새 atomic case의 두 snapshot만 측정했으며 sequential history나
retention outcome을 포함하지 않는다. 관측된 가능성은 수식적으로 다음보다
강하지 않다.

\[
\big[r_z(W_{\mathrm{cone}})<1\ \land\ \Delta\mathrm{NLL}>0\big]
_{\text{fresh 8-case, two-snapshot}}
\not\Rightarrow
\big[\Delta\mathrm{preservation}\ge0\big]
\not\Rightarrow
\big[\mathrm{retention}_{t+1:T}\uparrow\big].
\]

첫 비함의는 실제로 두 모델의 negative cone preservation gain으로 이미
드러난다. 따라서 이 자료는 lifelong/sequential-collapse 방지, 장기 안정성,
benchmark/general downstream gain, causal z→edit→preservation mediation,
또는 MEMIT/BF/cone/AlphaEdit/ODE-Edit 사이의 method superiority를 보장하지
않는다. 최종적으로 남는 판단은 모델별 conditional controllability evidence와
서로 보존된 BF verdict의 불일치다.
