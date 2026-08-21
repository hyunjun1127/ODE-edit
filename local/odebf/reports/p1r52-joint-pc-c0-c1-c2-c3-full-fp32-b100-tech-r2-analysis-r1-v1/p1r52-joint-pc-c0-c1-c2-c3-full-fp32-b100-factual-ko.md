# P1R52 Joint P+C C0/C1/C2/C3 + baseline FULL-FP32 1×B100 factual report

- 작성 시각: 2026-08-22 04:27:16 KST
- instruction: `ODEEDIT-S05-P1R52-JOINT-PC-C0-C1-C2-C3-FULL-FP32-B100-V1`
- 범위: Llama3-8B-Instruct, 독립 1×B100, C0/C1/C2/C3와 pre-edit/MEMIT/Official AlphaEdit/P1R52 J0
- scheduler: job `22489`, `COMPLETED`, ExitCode `0:0`, elapsed `00:58:51`, MaxRSS `21,587,516 KiB`
- source HEAD/tree: `6319748fc181c69eabe5d48ef5f84df7a6ffac1e` / `c93db49663465ccb45fe38dcfe773e9ca8190808`
- scientific_promotion: `false`

## 1. 한눈에 보는 결론

100/100 requests가 유효하며 technical failure 0, typed scientific failure 0, retry/imputation 0이다. C0–C3는 P1R52 accepted-z SHA `5772bc53771f446591d9c5a05cd598388fc26bcb41e5695f30438ac5bea45e57`와 writer-entry W0를 공유했고 각 arm 후 W0 bytes/pointer가 복원됐다.

FULL-FP32 C0/C1/C3의 rewrite NLL은 `0.051422/0.050670/0.050549`, rephrase NLL은 `2.169413/2.185594/2.188111`이다. C1−C0은 rewrite `-0.000752`, rephrase `+0.016181`, energy `-0.279690`이다. C2는 energy가 `12.756557`이나 rewrite/rephrase NLL이 `0.205147/2.524817`이다.

Official native-z 직접 평가에서 MEMIT/AlphaEdit rewrite z-NLL은 `0.000870/0.001156`이다. 기존 `7.x` 값은 이 실행의 동일 evaluator 직접-z 결과가 아니다. C3는 Official AlphaEdit **writer entrypoint**를 직접 호출하지만 target은 Official AlphaEdit native-z가 아니라 P1R52 accepted-z이다.

## 2. endpoint 절대 지표

| endpoint | rewrite NLL | rewrite success/acc | rephrase NLL | rephrase success/strict | rephrase acc/strict | LOC | rewrite/rephrase W−z gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| PRE_EDIT_W0 | 11.102700 | 13/100 · 0/100 | 9.821012 | 28/200 · 8/100 | 1/200 · 0/100 | 893/1000 | NOT_RECORDED / NOT_RECORDED |
| OFFICIAL_MEMIT_Z | 0.000870 | 100/100 · 100/100 | 1.092113 | 197/200 · 98/100 | 150/200 · 62/100 | 886/1000 | NOT_RECORDED / NOT_RECORDED |
| OFFICIAL_MEMIT_W | 0.346855 | 98/100 · 96/100 | 2.975614 | 164/200 · 76/100 | 100/200 · 36/100 | 886/1000 | 0.345985 / 1.883501 |
| OFFICIAL_ALPHAEDIT_Z | 0.001156 | 100/100 · 100/100 | 1.099024 | 197/200 · 98/100 | 151/200 · 63/100 | 871/1000 | NOT_RECORDED / NOT_RECORDED |
| OFFICIAL_ALPHAEDIT_W | 0.001398 | 100/100 · 100/100 | 1.920829 | 186/200 · 88/100 | 122/200 · 44/100 | 871/1000 | 0.000242 / 0.821805 |
| P1R52_ACCEPTED_Z | 0.053052 | 100/100 · 99/100 | 1.340614 | 196/200 · 96/100 | 138/200 · 53/100 | 881/1000 | NOT_RECORDED / NOT_RECORDED |
| P1R52_J0_W | 0.075676 | 100/100 · 98/100 | 2.218105 | 184/200 · 87/100 | 112/200 · 37/100 | 881/1000 | 0.022624 / 0.877490 |
| C0-PIRU-CONTROL | 0.051422 | 100/100 · 99/100 | 2.169413 | 184/200 · 87/100 | 112/200 · 35/100 | 886/1000 | -0.001630 / 0.828799 |
| C1-JOINT-PC-REMAINING | 0.050670 | 100/100 · 99/100 | 2.185594 | 184/200 · 87/100 | 112/200 · 35/100 | 886/1000 | -0.002382 / 0.844980 |
| C2-JOINT-PC-FIXED-QUOTA | 0.205147 | 99/100 · 96/100 | 2.524817 | 176/200 · 82/100 | 94/200 · 28/100 | 890/1000 | 0.152095 / 1.184203 |
| C3-DIRECT-OFFICIAL-ALPHAEDIT | 0.050549 | 100/100 · 99/100 | 2.188111 | 183/200 · 86/100 | 110/200 · 34/100 | 886/1000 | -0.002503 / 0.847497 |


`success`는 target-new NLL이 target-true NLL보다 낮은 prompt count이고, `accuracy`는 target-new token exact prediction count이다. strict rephrase는 request의 두 rephrase prompt가 모두 통과한 count다. z와 W를 별도 endpoint로 유지했으며 혼합하지 않았다.

## 3. 공통 accepted-z endpoint와 writer transfer

P1R52 accepted-z는 rewrite `100/100`, rephrase `196/200`, strict `96/100`이다. rewrite/rephrase NLL은 `0.053052/1.340614`이다.

| writer | rewrite z-success→W-failure | rephrase prompt z-success→W-failure | strict request z-success→W-failure | energy |
|---|---:|---:|---:|---:|
| P1R52_J0_W | 0 | 3 | 9 | 38.165205 |
| C0-PIRU-CONTROL | 0 | 3 | 9 | 44.807168 |
| C1-JOINT-PC-REMAINING | 0 | 3 | 9 | 44.527478 |
| C2-JOINT-PC-FIXED-QUOTA | 1 | 6 | 14 | 12.756557 |
| C3-DIRECT-OFFICIAL-ALPHAEDIT | 0 | 3 | 10 | 43.219772 |
| OFFICIAL_MEMIT_W | 2 | 11 | 22 | 75.746522 |
| OFFICIAL_ALPHAEDIT_W | 0 | 1 | 10 | 99.173529 |


Official MEMIT/AlphaEdit 행은 각자의 native-z→W 전송이고, P1R52 J0/C0–C3는 공통 P1R52 accepted-z→W 전송이다.

## 4. C0/C1/C2 배분과 층 집중도

- C0 control pi: `[0.197832267, 0.222733540, 0.218789469, 0.174717971, 0.185926753]`
- C1/C2 joint P+C pi: `[0.185755432, 0.214417309, 0.218404636, 0.200858489, 0.180564135]`
- joint minimax t/P-normalized/C-normalized: `0.994323979` / `0.994323979` / `0.994323979`
- solver success/status/stationarity residual: `True` / `0` / `8.742e-16`

| endpoint | L4/L5/L6/L7/L8 energy share | layer-8 share | total energy | final residual |
|---|---|---:|---:|---:|
| C0-PIRU-CONTROL | 0.0718/0.0985/0.1499/0.2003/0.4795 | 0.479540 | 44.807168 | 1.268506 |
| C1-JOINT-PC-REMAINING | 0.0637/0.0887/0.1387/0.2377/0.4713 | 0.471260 | 44.527478 | 1.256330 |
| C2-JOINT-PC-FIXED-QUOTA | 0.2223/0.2302/0.2138/0.1860/0.1476 | 0.147637 | 12.756557 | 26.974922 |
| C3-DIRECT-OFFICIAL-ALPHAEDIT | 0.0761/0.0828/0.1204/0.2231/0.4976 | 0.497622 | 43.219772 | NOT_RECORDED |
| P1R52_J0_W | 0.1618/0.1732/0.1741/0.2105/0.2804 | 0.280402 | 38.165205 | NOT_RECORDED |
| OFFICIAL_ALPHAEDIT_W | 0.0855/0.0945/0.1330/0.2337/0.4533 | 0.453320 | 99.173529 | NOT_RECORDED |
| OFFICIAL_MEMIT_W | 0.0908/0.0905/0.1195/0.2030/0.4961 | 0.496121 | 75.746522 | NOT_RECORDED |


C0와 C1은 suffix-normalized remaining-residual 실행이며 C1은 joint pi를 쓴다. C2는 같은 joint pi를 fixed entry-residual quota로 사용해 suffix normalization/catch-up 없이 실행한다. C3는 pi/P+C router를 쓰지 않으므로 allocation pi는 `NOT_APPLICABLE`; 표의 층 share는 실제 FP32 update energy에서 계산됐다.

## 5. C3 Official AlphaEdit writer identity

- source: `/mnt/raid5/janghj/EasyEdit/easyeditor/models/alphaedit/AlphaEdit_main.py`
- entrypoint: `easyeditor.models.alphaedit.AlphaEdit_main.apply_AlphaEdit_to_model`
- native AlphaEdit `compute_z` call: `0`
- accepted-z SHA: `5772bc53771f446591d9c5a05cd598388fc26bcb41e5695f30438ac5bea45e57` (공통 P1R52 z와 동일)
- P1R52 P+C/barrier decision influence: `0/0`
- static P SHA: `8140fc0ef380b986c907f4b39004447a1d56d2c412fe9e43cbac68a0b722a89a`
- cache_c entry/exit: `UNINITIALIZED` / `INITIALIZED`; logical width `0→100`

C3와 Official AlphaEdit W는 동일 writer entrypoint를 사용하지만 target이 다르다: C3는 P1R52 z, Official AlphaEdit baseline은 native AlphaEdit z다. 따라서 C3 전체 방법을 Official AlphaEdit 전체 방법과 동일하다고 기록하지 않는다.

## 6. FULL-FP32 경계와 이전 BF16 비교

- model/storage/planner/solve/update: 모두 `torch.float32`
- autocast/BF16/FP16 conversion, BF16 materializer: `0/0/0/0`
- C0/C1/C2는 layer당 native FP32 storage assignment 1회, arm당 5회이며 W0 restore PASS다.
- C3와 baseline writer도 FP32 model parameter에 FP32 update를 적용했다.

| arm | BF16→FP32 rewrite NLL delta | rephrase NLL delta | energy delta | final residual old→new | 비교 상태 |
|---|---:|---:|---:|---:|---|
| C0-PIRU-CONTROL | 0.023148 | -0.107089 | 3.402412 | 1.294668→1.268506 | EXACT_SAME_STREAM_METHOD_ARM_PAIRED_BUT_TARGET_DTYPE_DIFFERS |
| C1-JOINT-PC-REMAINING | 0.024273 | -0.111587 | 3.588259 | 1.278058→1.256330 | EXACT_SAME_STREAM_METHOD_ARM_PAIRED_BUT_TARGET_DTYPE_DIFFERS |
| C2-JOINT-PC-FIXED-QUOTA | -0.006389 | -0.108584 | 0.718411 | 25.581469→26.974922 | EXACT_SAME_STREAM_METHOD_ARM_PAIRED_BUT_TARGET_DTYPE_DIFFERS |
| C3-DIRECT-OFFICIAL-ALPHAEDIT | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | 1.301267→NOT_RECORDED | NOT_METHOD_MATCHED_FORMER_C3_WAS_UNIFORM_REMAINING |


C0–C2의 이전/현재 accepted-z SHA는 `467a195af3602208e97606ed5c2a78f1d148b4afd35bb2d2332e4f1b3410e461` / `5772bc53771f446591d9c5a05cd598388fc26bcb41e5695f30438ac5bea45e57`로 서로 다르므로 위 delta는 target와 writer dtype 경계가 함께 바뀐 동일-stream method-arm 비교다. 이전 `C3`는 uniform remaining-residual 구현이고 이번 C3는 direct Official AlphaEdit writer이므로 dtype-only delta를 계산하지 않았다.

## 7. compute와 기술 이력

| 항목 | 값 |
|---|---:|
| scheduler elapsed / MaxRSS | 00:58:51 / 21,587,516 KiB |
| total model forward / tokens | 6296 / 1121014 |
| MEMIT edit core / AlphaEdit edit core / C3 writer core seconds | 733.494 / 776.165 / 140.448 |
| native MEMIT/AlphaEdit target backward | 2400 / 2400 |
| P1R52 target, C0–C2/J0 writer, evaluator split wall | NOT_RECORDED |
| GPU peak bytes | NOT_RECORDED |

- job22475: Python pinned patch mismatch, pre-model-load technical failure.
- job22478: science execution 뒤 raw-free terminal JSON에서 tensor serialization technical failure.
- job22489: brace-safe/typed raw-free tensor metadata adapter 후 terminal valid. 치환 경로는 `$.writer_entry.joint_pi[0..4]`, serialized numeric value count 0, decision influence 0이다.
- technical/scientific failure count: `0/0`; attempted/valid `100/100`; retry/imputation `0/0`.

## 8. 판정 및 증거

- `TECHNICAL_VALIDITY=PASS`
- `FULL_FP32_DTYPE_GATE=PASS`
- `C3_DIRECT_OFFICIAL_WRITER_GATE=PASS`
- `INDEPENDENT_PRODUCTION_NOT_RUN=true`
- `SCIENTIFIC_PROMOTION=false`

단일 1×B100 결과다. 상세 per-request NLL/success, per-layer pi/coefficient/residual/q/update/energy, dtype, BF16 비교, compute 값은 같은 package의 machine-readable tables에 있다.

- result: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh2-s05-p1r52-joint-pc-full-fp32-b100-v1/local/odebf/results/s05-p1r52-joint-pc-c0-c1-c2-c3-full-fp32-b100-tech-r2-v1`
- terminal SHA256: `8a58753a26e451b5954c716415633184758e80890d74c56327e6f3489adf05fb`
- old BF16 C0–C2 result: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh2-s05-p1r52-joint-pc-c1-c2-independent-b100-v1/local/odebf/results/s05-p1r52-joint-pc-c1-c2-pilot-b100-tech-r6-v1/raw`
- old BF16 former-C3 result: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh2-s05-p1r52-joint-pc-uniform-weight-off-pilot-v1/local/odebf/results/s05-p1r52-joint-pc-uniform-weight-off-pilot-b100-v1/raw`
