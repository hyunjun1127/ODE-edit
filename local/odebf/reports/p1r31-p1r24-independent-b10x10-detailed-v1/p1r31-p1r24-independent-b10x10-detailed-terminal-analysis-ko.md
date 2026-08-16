# P1R31 P1R24 독립 B10×10 전체 행렬 상세 종결 분석

- 작성 시각: 2026-08-13 Asia/Seoul
- 실험: `ODEEDIT-S05-P1R31-P1R24-INDEPENDENT-B10X10-FULL-MATRIX-DETAILED-V1`
- 실행 checkpoint: `1f37dfe085ec29241f5e3bcfe0592b7c681da69c`; 과학 기반: `ce8c6c36348752f1407f7d713d30e6b5c727379b`
- Slurm: 최초 pre-model 기술 실패 `19231`은 결과/모델 action 0으로 보존; 유효 대체 array `19239` 8/8 scheduler terminal.
- stream root: `74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6`; order: `abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c`.
- 범위: Llama/Qwen × RS/BG × Neutral/Soft, 셀당 같은 10개 B10을 매번 W0에서 독립 편집. sequential/history가 아니다.
- claim boundary: reused development stream의 기계적·기술적 서술 분석이며 `scientific_promotion=false`.

## 1. 무결성, 분모, 실패

| 모델 | 할당 | arm | 성공/시도 | W0 restore | task wall(s) |
|---|---:|---:|---:|---:|---:|
| Llama | BG | Neutral | 10/10 | True | 5340.3 |
| Llama | BG | Soft | 9/10 | True | 4821.1 |
| Llama | RS | Neutral | 10/10 | True | 5297.7 |
| Llama | RS | Soft | 10/10 | True | 5252.4 |
| Qwen | BG | Neutral | 10/10 | True | 6117.3 |
| Qwen | BG | Soft | 10/10 | True | 6116.1 |
| Qwen | RS | Neutral | 10/10 | True | 6073.2 |
| Qwen | RS | Soft | 10/10 | True | 6048.7 |

- FACT: 80개 독립 B10 중 endpoint 79개, typed 실패 1개. 성공 79개는 action-freeze 뒤 k0..k8 9개 snapshot, case manifest/terminal 및 pointer+byte W0 restore를 가진다.
- SCIENTIFIC_FAIL: Llama BG-Soft case06은 k1..k5만 accepted 후 `ODEBFContractError`, message SHA `9f0637ba47f6971169f173556fa3ccf861eae37b0e33f564af9260a6597885d9`; retry/imputation 0, W0 restore PASS, case07 이후 계속 실행.
- TECHNICAL_FAIL: `19231` session-boundary binding pre-model 실패는 과학 endpoint가 아니며 `19239` TECH-R1이 전체 범위를 새 namespace에서 수행했다.
- 성공 case마다 scientific materialization=8; post-freeze evaluation-only materialization=8. Controller influence는 모두 0이며 inner heldout evaluator access=0.

## 2. 8셀 endpoint 요약

| 모델 | 할당 | arm | endpoint/10 | W E/G/L (분모100/200/1000) | W E NLL | z E/G (분모100/200) | z E NLL | z→W E/G gap | neg step | realization | P 종단 | functional-P | capacity |
|---|---|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Llama | BG | Neutral | 10/10 | 98/175/871 | 0.4188 | 99/179 | 0.2933 | +1/+4 | 1 | 0.3338 | 0.0113 | 0.0249 | 6.85 |
| Llama | BG | Soft | 9/10 | 89/164/787 | 0.2631 | 89/166 | 0.2326 | +0/+2 | 0 | 0.3588 | 0.0081 | 0.0205 | 4.66 |
| Llama | RS | Neutral | 10/10 | 99/178/873 | 0.1430 | 100/181 | 0.0824 | +1/+3 | 2 | 0.3475 | 0.0062 | 0.0194 | 3.77 |
| Llama | RS | Soft | 10/10 | 99/177/873 | 0.1795 | 100/184 | 0.0751 | +1/+7 | 1 | 0.2845 | 0.0068 | 0.0214 | 4.02 |
| Qwen | BG | Neutral | 10/10 | 94/165/843 | 1.2933 | 96/171 | 0.7831 | +2/+6 | 4 | 0.0633 | 0.9977 | 0.0058 | 108.71 |
| Qwen | BG | Soft | 10/10 | 94/164/842 | 1.2732 | 95/166 | 0.8612 | +1/+2 | 7 | -0.0040 | 0.4652 | 0.0190 | 42.85 |
| Qwen | RS | Neutral | 10/10 | 98/155/843 | 0.5851 | 99/162 | 0.3349 | +1/+7 | 3 | 0.0334 | 0.3749 | 0.0062 | 40.00 |
| Qwen | RS | Soft | 10/10 | 100/163/844 | 0.4922 | 99/165 | 0.3859 | +-1/+2 | 1 | 0.2022 | 0.1919 | 0.0198 | 17.72 |

해석: Llama는 RS가 BG보다 연속 E NLL과 용량에서 안정적이다. Qwen은 모든 셀에서 realization이 낮고, 특히 BG-Soft의 step 평균 realization은 -0.0040이며 negative-actual step 7개다. Soft는 Qwen RS에서 W E/G를 +2/+8 개선했지만, 그 자체가 모든 모델·할당에서 재현되는 보존 이득은 아니다.

## 3. P1R24 dynamic z-oracle sufficiency vs Official AlphaEdit direct-z baseline

### 3.1 비교 identity

- immutable Official report: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r23-progress-simplex-independent-b1x100-v1/local/odebf/reports/p1r23-independent-b10x10/p1r23-independent-b10x10-terminal-report-ko.md`; SHA `e82eaf647e00de0f35e051dc487ae6602d447fbf2714f3a61aece3d511c0017e`.
- 79개 성공 P1R24 endpoint에 대해 per-batch request order, evaluator source, aggregator source, evaluation-case identity, W0 E/G/L score identity, `direct_z_semantics=true`를 모두 재검증: `True`.
- global stream/order는 위 lock 및 Official sealed report와 일치한다. 모델 alias와 metric semantics도 동일하다.
- 중요한 경계: Official receipt가 제공하는 값은 `direct_z_semantics=true`인 AlphaEdit의 최종 편집 endpoint다. 별도의 latent-z k0..trajectory는 NOT_RECORDED다. 따라서 아래 “Official direct-z terminal”은 direct-z-driven Official endpoint이고, 새로운 standalone direct-z solve는 실행하지 않았다.
- P1R24 내부 z-oracle은 action trajectory를 먼저 freeze한 뒤 k0..k8에서 관측했으며 controller influence=0이다.

### 3.2 모델×셀 aggregate

| 모델 | 셀 | P1R24 z E/G | z E NLL | Official E/G | Official E NLL | matched-success z−Official E/G | z−Official E NLL | W−z E/G | W−z E NLL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama | BG-Neutral | 99/179 | 0.2933 | 100/185 | 0.0012 | -1/-6 | 0.2921 | 1/4 | 0.1255 |
| Llama | BG-Soft | 89/166 | 0.2326 | 100/185 | 0.0012 | -1/0 | 0.2314 | 0/2 | 0.0305 |
| Llama | RS-Neutral | 100/181 | 0.0824 | 100/185 | 0.0012 | 0/-4 | 0.0813 | 1/3 | 0.0606 |
| Llama | RS-Soft | 100/184 | 0.0751 | 100/185 | 0.0012 | 0/-1 | 0.0739 | 1/7 | 0.1044 |
| Qwen | BG-Neutral | 96/171 | 0.7831 | 100/192 | 0.0326 | -4/-21 | 0.7504 | 2/6 | 0.5102 |
| Qwen | BG-Soft | 95/166 | 0.8612 | 100/192 | 0.0326 | -5/-26 | 0.8286 | 1/2 | 0.4120 |
| Qwen | RS-Neutral | 99/162 | 0.3349 | 100/192 | 0.0326 | -1/-30 | 0.3023 | 1/7 | 0.2502 |
| Qwen | RS-Soft | 99/165 | 0.3859 | 100/192 | 0.0326 | -1/-27 | 0.3533 | -1/2 | 0.1063 |

### 3.3 직접 답변

1. **P1R24 target z-oracle이 Official direct-z baseline만큼 강한가?** 전체적으로 아니다. Llama RS는 z E=100/100이나 Gen=181–184/200로 Official 185/200보다 1–4 낮고 E NLL도 0.075–0.082 대 0.00117이다. Qwen은 z E=95–99/100, Gen=162–171/200로 Official 100/192보다 크게 낮고 E NLL 0.335–0.861 대 0.03265다.
2. **쓰기 전 결손은 얼마인가?** Llama 셀의 z Gen 결손은 Official 대비 1–19(실패 분모 포함 시 BG-Soft), Qwen은 21–30이다. 연속 E NLL 결손은 Llama +0.074–0.292, Qwen +0.302–0.829다.
3. **z에서 BF16 W로 추가되는 결손은?** 성공 endpoint에서 E count 0–2, Gen 2–7이 추가로 손실되고, W E NLL은 z보다 +0.030–0.510 높다. 특히 Qwen BG-Neutral은 +0.510, BG-Soft +0.412로 writer-side 손실도 크다.
4. **under-edit의 주원인은?** Qwen은 target-side와 writer-side가 모두 약하다. Gen 결손의 더 큰 부분은 이미 z 단계에 있으며, 낮은/음의 realization이 추가 손실을 만든다. Llama RS는 target이 Official보다 조금 약하고 writer가 소폭 더 약하다. Llama BG는 target·writer 혼합이며 BG-Soft case06 때문에 완전 식별이 불가하다.

### 3.4 per-batch paired table

| 모델 | 셀 | batch | status | z E/G | z E NLL mean/med/p90/worst | Official E/G | Official E NLL mean/med/p90/worst | W E/G | W E NLL | Δz-Official E/G | ΔW-z E/G | class |
|---|---|---:|---|---|---|---|---|---|---|---|---|---|
| Llama | BG-Neutral | 01 | SUCCESS | 10/18 | 0.0185/0.0098/0.0485/0.0654 | 10/19 | 0.0017/0.0005/0.0022/0.0123 | 10/18 | 0.0362 | 0/-1 | 0/0 | BOTH_WEAK_OR_TYPED_INCOMPLETE |
| Llama | BG-Neutral | 02 | SUCCESS | 10/16 | 0.1037/0.0132/0.1450/0.8984 | 10/20 | 0.0006/0.0005/0.0012/0.0016 | 10/16 | 0.0421 | 0/-4 | 0/0 | Z_TARGET_WEAK |
| Llama | BG-Neutral | 03 | SUCCESS | 10/19 | 0.0192/0.0043/0.0383/0.1258 | 10/18 | 0.0006/0.0003/0.0018/0.0020 | 10/19 | 0.0345 | 0/1 | 0/0 | MIXED_NOT_IDENTIFIABLE |
| Llama | BG-Neutral | 04 | SUCCESS | 10/16 | 0.0893/0.0348/0.2469/0.2539 | 10/15 | 0.0019/0.0009/0.0030/0.0094 | 10/16 | 0.1123 | 0/1 | 0/0 | MIXED_NOT_IDENTIFIABLE |
| Llama | BG-Neutral | 05 | SUCCESS | 10/17 | 0.2530/0.0135/0.3461/2.2656 | 10/17 | 0.0006/0.0006/0.0011/0.0015 | 10/17 | 0.4029 | 0/0 | 0/0 | MIXED_NOT_IDENTIFIABLE |
| Llama | BG-Neutral | 06 | SUCCESS | 10/17 | 0.2945/0.0109/0.4965/2.4688 | 10/19 | 0.0006/0.0006/0.0012/0.0017 | 10/16 | 1.0856 | 0/-2 | 0/1 | BOTH_WEAK_OR_TYPED_INCOMPLETE |
| Llama | BG-Neutral | 07 | SUCCESS | 10/20 | 0.5405/0.0135/0.5983/5.1875 | 10/20 | 0.0012/0.0008/0.0030/0.0032 | 10/19 | 0.3987 | 0/0 | 0/1 | MIXED_NOT_IDENTIFIABLE |
| Llama | BG-Neutral | 08 | SUCCESS | 10/20 | 0.0490/0.0099/0.0903/0.3555 | 10/20 | 0.0018/0.0014/0.0037/0.0049 | 9/19 | 0.4432 | 0/0 | 1/1 | MIXED_NOT_IDENTIFIABLE |
| Llama | BG-Neutral | 09 | SUCCESS | 10/20 | 0.0239/0.0157/0.0571/0.0791 | 10/18 | 0.0006/0.0004/0.0012/0.0013 | 10/20 | 0.0214 | 0/2 | 0/0 | MIXED_NOT_IDENTIFIABLE |
| Llama | BG-Neutral | 10 | SUCCESS | 9/16 | 1.5414/0.0229/4.9687/10.8750 | 10/19 | 0.0021/0.0010/0.0044/0.0099 | 9/15 | 1.6112 | -1/-3 | 0/1 | BOTH_WEAK_OR_TYPED_INCOMPLETE |
| Llama | BG-Soft | 01 | SUCCESS | 10/19 | 0.0091/0.0064/0.0207/0.0270 | 10/19 | 0.0017/0.0005/0.0022/0.0123 | 10/19 | 0.0168 | 0/0 | 0/0 | MIXED_NOT_IDENTIFIABLE |
| Llama | BG-Soft | 02 | SUCCESS | 10/16 | 0.0567/0.0143/0.1089/0.3945 | 10/20 | 0.0006/0.0005/0.0012/0.0016 | 10/16 | 0.0702 | 0/-4 | 0/0 | BOTH_WEAK_OR_TYPED_INCOMPLETE |
| Llama | BG-Soft | 03 | SUCCESS | 10/20 | 0.0192/0.0043/0.0383/0.1251 | 10/18 | 0.0006/0.0003/0.0018/0.0020 | 10/19 | 0.0337 | 0/2 | 0/1 | MIXED_NOT_IDENTIFIABLE |
| Llama | BG-Soft | 04 | SUCCESS | 10/16 | 0.1963/0.0342/0.3368/1.3906 | 10/15 | 0.0019/0.0009/0.0030/0.0094 | 10/16 | 0.1982 | 0/1 | 0/0 | MIXED_NOT_IDENTIFIABLE |
| Llama | BG-Soft | 05 | SUCCESS | 10/17 | 0.2223/0.0121/0.3147/1.9609 | 10/17 | 0.0006/0.0006/0.0011/0.0015 | 10/17 | 0.3335 | 0/0 | 0/0 | MIXED_NOT_IDENTIFIABLE |
| Llama | BG-Soft | 06 | TYPED_CASE_FAILURE | N/R | N/R | 10/19 | 0.0006/0.0006/0.0012/0.0017 | N/R | N/R | N/R | N/R | BOTH_WEAK_OR_TYPED_INCOMPLETE |
| Llama | BG-Soft | 07 | SUCCESS | 10/20 | 0.4488/0.0133/0.4993/4.2812 | 10/20 | 0.0012/0.0008/0.0030/0.0032 | 10/19 | 0.4387 | 0/0 | 0/1 | MIXED_NOT_IDENTIFIABLE |
| Llama | BG-Soft | 08 | SUCCESS | 10/20 | 0.0383/0.0059/0.0679/0.2949 | 10/20 | 0.0018/0.0014/0.0037/0.0049 | 10/20 | 0.0976 | 0/0 | 0/0 | MIXED_NOT_IDENTIFIABLE |
| Llama | BG-Soft | 09 | SUCCESS | 10/20 | 0.0306/0.0216/0.0771/0.0771 | 10/18 | 0.0006/0.0004/0.0012/0.0013 | 10/20 | 0.0525 | 0/2 | 0/0 | MIXED_NOT_IDENTIFIABLE |
| Llama | BG-Soft | 10 | SUCCESS | 9/18 | 1.0720/0.0192/1.1159/10.5000 | 10/19 | 0.0021/0.0010/0.0044/0.0099 | 9/18 | 1.1266 | -1/-1 | 0/0 | BOTH_WEAK_OR_TYPED_INCOMPLETE |
| Llama | RS-Neutral | 01 | SUCCESS | 10/19 | 0.0044/0.0015/0.0171/0.0194 | 10/19 | 0.0017/0.0005/0.0022/0.0123 | 10/19 | 0.0102 | 0/0 | 0/0 | MIXED_NOT_IDENTIFIABLE |
| Llama | RS-Neutral | 02 | SUCCESS | 10/16 | 0.2191/0.0021/0.2360/2.1250 | 10/20 | 0.0006/0.0005/0.0012/0.0016 | 10/16 | 0.0825 | 0/-4 | 0/0 | Z_TARGET_WEAK |
| Llama | RS-Neutral | 03 | SUCCESS | 10/19 | 0.0190/0.0027/0.0377/0.1250 | 10/18 | 0.0006/0.0003/0.0018/0.0020 | 10/19 | 0.0165 | 0/1 | 0/0 | MIXED_NOT_IDENTIFIABLE |
| Llama | RS-Neutral | 04 | SUCCESS | 10/16 | 0.0079/0.0025/0.0132/0.0549 | 10/15 | 0.0019/0.0009/0.0030/0.0094 | 10/16 | 0.0103 | 0/1 | 0/0 | MIXED_NOT_IDENTIFIABLE |
| Llama | RS-Neutral | 05 | SUCCESS | 10/17 | 0.1558/0.0045/0.1675/1.5078 | 10/17 | 0.0006/0.0006/0.0011/0.0015 | 10/17 | 0.2810 | 0/0 | 0/0 | MIXED_NOT_IDENTIFIABLE |
| Llama | RS-Neutral | 06 | SUCCESS | 10/18 | 0.0046/0.0004/0.0067/0.0361 | 10/19 | 0.0006/0.0006/0.0012/0.0017 | 10/17 | 0.0359 | 0/-1 | 0/1 | BOTH_WEAK_OR_TYPED_INCOMPLETE |
| Llama | RS-Neutral | 07 | SUCCESS | 10/19 | 0.0085/0.0032/0.0313/0.0325 | 10/20 | 0.0012/0.0008/0.0030/0.0032 | 9/18 | 0.4236 | 0/-1 | 1/1 | BOTH_WEAK_OR_TYPED_INCOMPLETE |
| Llama | RS-Neutral | 08 | SUCCESS | 10/20 | 0.0023/0.0012/0.0058/0.0103 | 10/20 | 0.0018/0.0014/0.0037/0.0049 | 10/19 | 0.1283 | 0/0 | 0/1 | MIXED_NOT_IDENTIFIABLE |
| Llama | RS-Neutral | 09 | SUCCESS | 10/19 | 0.0045/0.0021/0.0077/0.0269 | 10/18 | 0.0006/0.0004/0.0012/0.0013 | 10/19 | 0.0041 | 0/1 | 0/0 | MIXED_NOT_IDENTIFIABLE |
| Llama | RS-Neutral | 10 | SUCCESS | 10/18 | 0.3983/0.0025/0.4826/3.8594 | 10/19 | 0.0021/0.0010/0.0044/0.0099 | 10/18 | 0.4380 | 0/-1 | 0/0 | BOTH_WEAK_OR_TYPED_INCOMPLETE |
| Llama | RS-Soft | 01 | SUCCESS | 10/19 | 0.0086/0.0016/0.0304/0.0457 | 10/19 | 0.0017/0.0005/0.0022/0.0123 | 10/19 | 0.1288 | 0/0 | 0/0 | MIXED_NOT_IDENTIFIABLE |
| Llama | RS-Soft | 02 | SUCCESS | 10/17 | 0.0949/0.0019/0.1141/0.8789 | 10/20 | 0.0006/0.0005/0.0012/0.0016 | 10/17 | 0.0308 | 0/-3 | 0/0 | Z_TARGET_WEAK |
| Llama | RS-Soft | 03 | SUCCESS | 10/20 | 0.0345/0.0027/0.0516/0.2969 | 10/18 | 0.0006/0.0003/0.0018/0.0020 | 10/19 | 0.0160 | 0/2 | 0/1 | MIXED_NOT_IDENTIFIABLE |
| Llama | RS-Soft | 04 | SUCCESS | 10/16 | 0.0113/0.0024/0.0287/0.0718 | 10/15 | 0.0019/0.0009/0.0030/0.0094 | 10/16 | 0.0178 | 0/1 | 0/0 | MIXED_NOT_IDENTIFIABLE |
| Llama | RS-Soft | 05 | SUCCESS | 10/17 | 0.1659/0.0040/0.1724/1.6172 | 10/17 | 0.0006/0.0006/0.0011/0.0015 | 10/17 | 0.2607 | 0/0 | 0/0 | MIXED_NOT_IDENTIFIABLE |
| Llama | RS-Soft | 06 | SUCCESS | 10/18 | 0.0205/0.0004/0.0521/0.1118 | 10/19 | 0.0006/0.0006/0.0012/0.0017 | 10/17 | 0.0054 | 0/-1 | 0/1 | BOTH_WEAK_OR_TYPED_INCOMPLETE |
| Llama | RS-Soft | 07 | SUCCESS | 10/20 | 0.0143/0.0034/0.0353/0.0776 | 10/20 | 0.0012/0.0008/0.0030/0.0032 | 9/15 | 0.8279 | 0/0 | 1/5 | MIXED_NOT_IDENTIFIABLE |
| Llama | RS-Soft | 08 | SUCCESS | 10/20 | 0.0046/0.0016/0.0123/0.0208 | 10/20 | 0.0018/0.0014/0.0037/0.0049 | 10/20 | 0.0911 | 0/0 | 0/0 | MIXED_NOT_IDENTIFIABLE |
| Llama | RS-Soft | 09 | SUCCESS | 10/19 | 0.0040/0.0022/0.0072/0.0210 | 10/18 | 0.0006/0.0004/0.0012/0.0013 | 10/19 | 0.0044 | 0/1 | 0/0 | MIXED_NOT_IDENTIFIABLE |
| Llama | RS-Soft | 10 | SUCCESS | 10/18 | 0.3925/0.0033/0.5005/3.7656 | 10/19 | 0.0021/0.0010/0.0044/0.0099 | 10/18 | 0.4125 | 0/-1 | 0/0 | BOTH_WEAK_OR_TYPED_INCOMPLETE |
| Qwen | BG-Neutral | 01 | SUCCESS | 10/20 | 0.4231/0.0187/0.4783/4.0312 | 10/20 | 0.0074/0.0049/0.0158/0.0176 | 10/19 | 0.4395 | 0/0 | 0/1 | MIXED_NOT_IDENTIFIABLE |
| Qwen | BG-Neutral | 02 | SUCCESS | 10/17 | 0.2882/0.0331/0.8324/1.8906 | 10/20 | 0.0240/0.0175/0.0412/0.0869 | 10/17 | 0.4519 | 0/-3 | 0/0 | BOTH_WEAK_OR_TYPED_INCOMPLETE |
| Qwen | BG-Neutral | 03 | SUCCESS | 10/17 | 0.8808/0.0388/1.0194/8.3750 | 10/20 | 0.0218/0.0155/0.0335/0.1006 | 9/17 | 2.1755 | 0/-3 | 1/0 | BOTH_WEAK_OR_TYPED_INCOMPLETE |
| Qwen | BG-Neutral | 04 | SUCCESS | 9/15 | 1.8294/0.0380/8.5500/9.5625 | 10/18 | 0.0177/0.0105/0.0405/0.0581 | 9/15 | 1.9090 | -1/-3 | 0/0 | BOTH_WEAK_OR_TYPED_INCOMPLETE |
| Qwen | BG-Neutral | 05 | SUCCESS | 10/19 | 1.0625/0.0894/2.1383/7.5312 | 10/19 | 0.0267/0.0106/0.0488/0.1602 | 10/18 | 1.3254 | 0/0 | 0/1 | MIXED_NOT_IDENTIFIABLE |
| Qwen | BG-Neutral | 06 | SUCCESS | 10/15 | 0.4560/0.0330/0.5673/4.1875 | 10/19 | 0.0689/0.0150/0.2318/0.3320 | 10/15 | 0.4740 | 0/-4 | 0/0 | BOTH_WEAK_OR_TYPED_INCOMPLETE |
| Qwen | BG-Neutral | 07 | SUCCESS | 9/17 | 0.9959/0.1558/2.4594/6.5938 | 10/17 | 0.0871/0.0173/0.1317/0.6094 | 9/16 | 1.6835 | -1/0 | 0/1 | BOTH_WEAK_OR_TYPED_INCOMPLETE |
| Qwen | BG-Neutral | 08 | SUCCESS | 10/18 | 0.1258/0.0509/0.1891/0.7656 | 10/20 | 0.0255/0.0162/0.0572/0.0645 | 10/18 | 0.2133 | 0/-2 | 0/0 | BOTH_WEAK_OR_TYPED_INCOMPLETE |
| Qwen | BG-Neutral | 09 | SUCCESS | 10/18 | 0.0849/0.0491/0.1636/0.3086 | 10/19 | 0.0329/0.0276/0.0618/0.0732 | 9/15 | 2.2145 | 0/-1 | 1/3 | BOTH_WEAK_OR_TYPED_INCOMPLETE |
| Qwen | BG-Neutral | 10 | SUCCESS | 8/15 | 1.6842/0.0964/4.1484/11.8125 | 10/20 | 0.0146/0.0071/0.0307/0.0684 | 8/15 | 2.0466 | -2/-5 | 0/0 | BOTH_WEAK_OR_TYPED_INCOMPLETE |
| Qwen | BG-Soft | 01 | SUCCESS | 10/19 | 0.4364/0.0236/0.7727/3.7188 | 10/20 | 0.0074/0.0049/0.0158/0.0176 | 10/19 | 0.3817 | 0/-1 | 0/0 | Z_TARGET_WEAK |
| Qwen | BG-Soft | 02 | SUCCESS | 10/17 | 0.0463/0.0500/0.0827/0.0928 | 10/20 | 0.0240/0.0175/0.0412/0.0869 | 10/17 | 0.1072 | 0/-3 | 0/0 | BOTH_WEAK_OR_TYPED_INCOMPLETE |
| Qwen | BG-Soft | 03 | SUCCESS | 10/17 | 0.3226/0.0334/0.7219/2.4375 | 10/20 | 0.0218/0.0155/0.0335/0.1006 | 10/18 | 0.9233 | 0/-3 | 0/-1 | BOTH_WEAK_OR_TYPED_INCOMPLETE |
| Qwen | BG-Soft | 04 | SUCCESS | 8/15 | 2.0507/0.0448/8.6500/11.6875 | 10/18 | 0.0177/0.0105/0.0405/0.0581 | 8/15 | 2.6781 | -2/-3 | 0/0 | BOTH_WEAK_OR_TYPED_INCOMPLETE |
| Qwen | BG-Soft | 05 | SUCCESS | 10/18 | 0.9285/0.1577/2.1266/6.5000 | 10/19 | 0.0267/0.0106/0.0488/0.1602 | 10/18 | 1.5002 | 0/-1 | 0/0 | BOTH_WEAK_OR_TYPED_INCOMPLETE |
| Qwen | BG-Soft | 06 | SUCCESS | 10/15 | 0.8302/0.0422/3.7375/4.1875 | 10/19 | 0.0689/0.0150/0.2318/0.3320 | 9/15 | 2.4937 | 0/-4 | 1/0 | BOTH_WEAK_OR_TYPED_INCOMPLETE |
| Qwen | BG-Soft | 07 | SUCCESS | 9/16 | 0.3567/0.1013/1.2148/1.4609 | 10/17 | 0.0871/0.0173/0.1317/0.6094 | 10/16 | 0.3628 | -1/-1 | -1/0 | BOTH_WEAK_OR_TYPED_INCOMPLETE |
| Qwen | BG-Soft | 08 | SUCCESS | 9/17 | 1.4549/0.0874/1.8535/13.4375 | 10/20 | 0.0255/0.0162/0.0572/0.0645 | 9/14 | 2.0058 | -1/-3 | 0/3 | BOTH_WEAK_OR_TYPED_INCOMPLETE |
| Qwen | BG-Soft | 09 | SUCCESS | 10/17 | 0.2299/0.0518/0.2716/1.8594 | 10/19 | 0.0329/0.0276/0.0618/0.0732 | 10/17 | 0.2233 | 0/-2 | 0/0 | Z_TARGET_WEAK |
| Qwen | BG-Soft | 10 | SUCCESS | 9/15 | 1.9562/0.0911/5.7094/11.8125 | 10/20 | 0.0146/0.0071/0.0307/0.0684 | 8/15 | 2.0561 | -1/-5 | 1/0 | BOTH_WEAK_OR_TYPED_INCOMPLETE |
| Qwen | RS-Neutral | 01 | SUCCESS | 10/18 | 0.0209/0.0097/0.0275/0.1416 | 10/20 | 0.0074/0.0049/0.0158/0.0176 | 10/18 | 0.0282 | 0/-2 | 0/0 | BOTH_WEAK_OR_TYPED_INCOMPLETE |
| Qwen | RS-Neutral | 02 | SUCCESS | 10/16 | 0.3838/0.0331/0.4137/3.6094 | 10/20 | 0.0240/0.0175/0.0412/0.0869 | 10/17 | 0.4736 | 0/-4 | 0/-1 | BOTH_WEAK_OR_TYPED_INCOMPLETE |
| Qwen | RS-Neutral | 03 | SUCCESS | 10/16 | 0.3499/0.0063/0.3771/3.4219 | 10/20 | 0.0218/0.0155/0.0335/0.1006 | 10/16 | 0.3333 | 0/-4 | 0/0 | Z_TARGET_WEAK |
| Qwen | RS-Neutral | 04 | SUCCESS | 10/14 | 1.0648/0.0146/3.8172/7.0938 | 10/18 | 0.0177/0.0105/0.0405/0.0581 | 10/14 | 1.0069 | 0/-4 | 0/0 | Z_TARGET_WEAK |
| Qwen | RS-Neutral | 05 | SUCCESS | 10/19 | 0.0923/0.0239/0.3166/0.4238 | 10/19 | 0.0267/0.0106/0.0488/0.1602 | 10/18 | 0.2993 | 0/0 | 0/1 | MIXED_NOT_IDENTIFIABLE |
| Qwen | RS-Neutral | 06 | SUCCESS | 10/15 | 0.0708/0.0107/0.1479/0.4941 | 10/19 | 0.0689/0.0150/0.2318/0.3320 | 9/12 | 1.2521 | 0/-4 | 1/3 | BOTH_WEAK_OR_TYPED_INCOMPLETE |
| Qwen | RS-Neutral | 07 | SUCCESS | 9/17 | 0.4003/0.0333/0.9949/2.5312 | 10/17 | 0.0871/0.0173/0.1317/0.6094 | 10/16 | 0.4701 | -1/0 | -1/1 | BOTH_WEAK_OR_TYPED_INCOMPLETE |
| Qwen | RS-Neutral | 08 | SUCCESS | 10/15 | 0.0146/0.0067/0.0276/0.0698 | 10/20 | 0.0255/0.0162/0.0572/0.0645 | 10/14 | 0.0198 | 0/-5 | 0/1 | MIXED_NOT_IDENTIFIABLE |
| Qwen | RS-Neutral | 09 | SUCCESS | 10/16 | 0.0826/0.0203/0.1246/0.6172 | 10/19 | 0.0329/0.0276/0.0618/0.0732 | 10/16 | 0.1168 | 0/-3 | 0/0 | BOTH_WEAK_OR_TYPED_INCOMPLETE |
| Qwen | RS-Neutral | 10 | SUCCESS | 10/16 | 0.8691/0.0599/1.7242/6.6250 | 10/20 | 0.0146/0.0071/0.0307/0.0684 | 9/14 | 1.8510 | 0/-4 | 1/2 | BOTH_WEAK_OR_TYPED_INCOMPLETE |
| Qwen | RS-Soft | 01 | SUCCESS | 10/17 | 0.0496/0.0095/0.0789/0.3867 | 10/20 | 0.0074/0.0049/0.0158/0.0176 | 10/17 | 0.0807 | 0/-3 | 0/0 | BOTH_WEAK_OR_TYPED_INCOMPLETE |
| Qwen | RS-Soft | 02 | SUCCESS | 10/19 | 0.0522/0.0098/0.0654/0.4297 | 10/20 | 0.0240/0.0175/0.0412/0.0869 | 10/19 | 0.0422 | 0/-1 | 0/0 | Z_TARGET_WEAK |
| Qwen | RS-Soft | 03 | SUCCESS | 10/16 | 0.4333/0.0111/0.5039/4.1250 | 10/20 | 0.0218/0.0155/0.0335/0.1006 | 10/16 | 1.1972 | 0/-4 | 0/0 | BOTH_WEAK_OR_TYPED_INCOMPLETE |
| Qwen | RS-Soft | 04 | SUCCESS | 10/15 | 1.4532/0.0160/5.7156/9.0625 | 10/18 | 0.0177/0.0105/0.0405/0.0581 | 10/15 | 1.4379 | 0/-3 | 0/0 | Z_TARGET_WEAK |
| Qwen | RS-Soft | 05 | SUCCESS | 10/19 | 0.1121/0.0224/0.3287/0.3691 | 10/19 | 0.0267/0.0106/0.0488/0.1602 | 10/17 | 0.4595 | 0/0 | 0/2 | MIXED_NOT_IDENTIFIABLE |
| Qwen | RS-Soft | 06 | SUCCESS | 10/16 | 0.0251/0.0084/0.0602/0.1406 | 10/19 | 0.0689/0.0150/0.2318/0.3320 | 10/16 | 0.0195 | 0/-3 | 0/0 | MIXED_NOT_IDENTIFIABLE |
| Qwen | RS-Soft | 07 | SUCCESS | 9/17 | 0.7913/0.0223/3.2953/4.1250 | 10/17 | 0.0871/0.0173/0.1317/0.6094 | 10/17 | 0.6284 | -1/0 | -1/0 | Z_TARGET_WEAK |
| Qwen | RS-Soft | 08 | SUCCESS | 10/15 | 0.0352/0.0094/0.0548/0.2422 | 10/20 | 0.0255/0.0162/0.0572/0.0645 | 10/15 | 0.0126 | 0/-5 | 0/0 | Z_TARGET_WEAK |
| Qwen | RS-Soft | 09 | SUCCESS | 10/16 | 0.0259/0.0189/0.0475/0.0996 | 10/19 | 0.0329/0.0276/0.0618/0.0732 | 10/16 | 0.0261 | 0/-3 | 0/0 | MIXED_NOT_IDENTIFIABLE |
| Qwen | RS-Soft | 10 | SUCCESS | 10/15 | 0.8813/0.0244/1.4223/7.1562 | 10/20 | 0.0146/0.0071/0.0307/0.0684 | 10/15 | 1.0181 | 0/-5 | 0/0 | BOTH_WEAK_OR_TYPED_INCOMPLETE |

case-level 분류는 새 threshold를 쓰지 않는다. Official 대비 exact E/Gen count 및 연속 E NLL의 방향과 z→W의 exact gap만으로 분류했으며, 서로 다른 방향이면 `MIXED_NOT_IDENTIFIABLE`이다. Llama BG-Soft case06은 endpoint를 보간하지 않고 `BOTH_WEAK_OR_TYPED_INCOMPLETE`로 남겼다.

### 3.5 k0→k8 z/W trajectory

| 모델 | 셀 | k | snapshot case | z E/G | W E/G | z E NLL | W E NLL | W−z NLL | negative actual |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama | BG-Neutral | 0 | 10 | 13/28 | 13/28 | 11.0967 | 11.0967 | 0.0000 | 0 |
| Llama | BG-Neutral | 1 | 10 | 25/54 | 24/52 | 8.2342 | 8.6726 | 0.4384 | 0 |
| Llama | BG-Neutral | 2 | 10 | 53/90 | 54/90 | 5.4537 | 5.7386 | 0.2849 | 0 |
| Llama | BG-Neutral | 3 | 10 | 77/133 | 76/131 | 3.4955 | 3.7277 | 0.2323 | 0 |
| Llama | BG-Neutral | 4 | 10 | 90/153 | 90/155 | 1.9554 | 2.1299 | 0.1745 | 0 |
| Llama | BG-Neutral | 5 | 10 | 95/169 | 96/171 | 1.1211 | 1.1640 | 0.0430 | 0 |
| Llama | BG-Neutral | 6 | 10 | 99/181 | 95/177 | 0.8033 | 1.2491 | 0.4458 | 0 |
| Llama | BG-Neutral | 7 | 10 | 98/181 | 97/175 | 0.4719 | 0.7491 | 0.2772 | 0 |
| Llama | BG-Neutral | 8 | 10 | 99/179 | 98/175 | 0.2933 | 0.4188 | 0.1255 | 1 |
| Llama | BG-Soft | 0 | 9 | 11/22 | 11/22 | 11.2158 | 11.2158 | 0.0000 | 0 |
| Llama | BG-Soft | 1 | 9 | 23/48 | 22/44 | 8.2820 | 8.7440 | 0.4620 | 0 |
| Llama | BG-Soft | 2 | 9 | 47/80 | 48/80 | 5.5134 | 5.8011 | 0.2878 | 0 |
| Llama | BG-Soft | 3 | 9 | 68/120 | 66/118 | 3.5792 | 3.8243 | 0.2451 | 0 |
| Llama | BG-Soft | 4 | 9 | 81/139 | 81/142 | 1.9437 | 2.0907 | 0.1470 | 0 |
| Llama | BG-Soft | 5 | 9 | 86/154 | 86/156 | 1.0619 | 1.1738 | 0.1119 | 0 |
| Llama | BG-Soft | 6 | 9 | 88/163 | 88/162 | 0.7594 | 0.7946 | 0.0351 | 0 |
| Llama | BG-Soft | 7 | 9 | 87/164 | 87/161 | 0.4350 | 0.5549 | 0.1199 | 0 |
| Llama | BG-Soft | 8 | 9 | 89/166 | 89/164 | 0.2326 | 0.2631 | 0.0305 | 0 |
| Llama | RS-Neutral | 0 | 10 | 13/28 | 13/28 | 11.0967 | 11.0967 | 0.0000 | 0 |
| Llama | RS-Neutral | 1 | 10 | 24/52 | 22/51 | 7.8307 | 8.2939 | 0.4632 | 0 |
| Llama | RS-Neutral | 2 | 10 | 60/93 | 59/90 | 4.7603 | 5.1137 | 0.3534 | 0 |
| Llama | RS-Neutral | 3 | 10 | 89/147 | 86/145 | 2.5626 | 2.8757 | 0.3131 | 0 |
| Llama | RS-Neutral | 4 | 10 | 94/167 | 95/164 | 1.2415 | 1.4639 | 0.2224 | 0 |
| Llama | RS-Neutral | 5 | 10 | 99/178 | 97/171 | 0.5253 | 0.6752 | 0.1499 | 0 |
| Llama | RS-Neutral | 6 | 10 | 98/182 | 98/177 | 0.2611 | 0.4120 | 0.1510 | 0 |
| Llama | RS-Neutral | 7 | 10 | 100/184 | 100/175 | 0.1349 | 0.2436 | 0.1087 | 0 |
| Llama | RS-Neutral | 8 | 10 | 100/181 | 99/178 | 0.0824 | 0.1430 | 0.0606 | 2 |
| Llama | RS-Soft | 0 | 10 | 13/28 | 13/28 | 11.0967 | 11.0967 | 0.0000 | 0 |
| Llama | RS-Soft | 1 | 10 | 24/52 | 22/52 | 7.8325 | 8.2963 | 0.4637 | 0 |
| Llama | RS-Soft | 2 | 10 | 61/93 | 58/91 | 4.7552 | 5.1167 | 0.3615 | 0 |
| Llama | RS-Soft | 3 | 10 | 89/148 | 87/145 | 2.5599 | 2.8828 | 0.3230 | 0 |
| Llama | RS-Soft | 4 | 10 | 95/168 | 95/163 | 1.2361 | 1.4680 | 0.2319 | 0 |
| Llama | RS-Soft | 5 | 10 | 99/178 | 97/170 | 0.5260 | 0.6747 | 0.1487 | 0 |
| Llama | RS-Soft | 6 | 10 | 98/184 | 98/177 | 0.2572 | 0.3909 | 0.1337 | 0 |
| Llama | RS-Soft | 7 | 10 | 100/184 | 100/182 | 0.1353 | 0.2380 | 0.1027 | 0 |
| Llama | RS-Soft | 8 | 10 | 100/184 | 99/177 | 0.0751 | 0.1795 | 0.1044 | 1 |
| Qwen | BG-Neutral | 0 | 10 | 11/36 | 11/36 | 10.3780 | 10.3780 | 0.0000 | 0 |
| Qwen | BG-Neutral | 1 | 10 | 26/50 | 25/51 | 7.9578 | 7.9772 | 0.0195 | 0 |
| Qwen | BG-Neutral | 2 | 10 | 60/89 | 59/99 | 5.3682 | 5.4346 | 0.0664 | 0 |
| Qwen | BG-Neutral | 3 | 10 | 77/122 | 80/131 | 3.5716 | 3.6673 | 0.0957 | 0 |
| Qwen | BG-Neutral | 4 | 10 | 85/142 | 85/147 | 2.5788 | 2.5938 | 0.0150 | 0 |
| Qwen | BG-Neutral | 5 | 10 | 88/157 | 89/155 | 1.7553 | 1.9673 | 0.2120 | 0 |
| Qwen | BG-Neutral | 6 | 10 | 89/160 | 91/161 | 1.3464 | 1.4199 | 0.0735 | 0 |
| Qwen | BG-Neutral | 7 | 10 | 92/167 | 90/159 | 1.0212 | 1.4761 | 0.4549 | 0 |
| Qwen | BG-Neutral | 8 | 10 | 96/171 | 94/165 | 0.7831 | 1.2933 | 0.5102 | 4 |
| Qwen | BG-Soft | 0 | 10 | 11/36 | 11/36 | 10.3780 | 10.3780 | 0.0000 | 0 |
| Qwen | BG-Soft | 1 | 10 | 28/49 | 25/54 | 7.9594 | 7.9648 | 0.0054 | 0 |
| Qwen | BG-Soft | 2 | 10 | 59/90 | 58/99 | 5.3565 | 5.4026 | 0.0461 | 0 |
| Qwen | BG-Soft | 3 | 10 | 77/117 | 79/128 | 3.5577 | 3.5865 | 0.0287 | 0 |
| Qwen | BG-Soft | 4 | 10 | 84/141 | 83/149 | 2.5120 | 2.5372 | 0.0251 | 0 |
| Qwen | BG-Soft | 5 | 10 | 90/148 | 91/153 | 1.7238 | 1.8758 | 0.1520 | 0 |
| Qwen | BG-Soft | 6 | 10 | 92/157 | 90/158 | 1.2948 | 1.3954 | 0.1006 | 0 |
| Qwen | BG-Soft | 7 | 10 | 95/164 | 95/160 | 0.9285 | 1.1359 | 0.2074 | 0 |
| Qwen | BG-Soft | 8 | 10 | 95/166 | 94/164 | 0.8612 | 1.2732 | 0.4120 | 7 |
| Qwen | RS-Neutral | 0 | 10 | 11/36 | 11/36 | 10.3780 | 10.3780 | 0.0000 | 0 |
| Qwen | RS-Neutral | 1 | 10 | 29/49 | 31/51 | 7.5506 | 7.5677 | 0.0172 | 0 |
| Qwen | RS-Neutral | 2 | 10 | 63/104 | 63/112 | 4.6895 | 4.7608 | 0.0713 | 0 |
| Qwen | RS-Neutral | 3 | 10 | 85/134 | 86/138 | 2.6682 | 2.7774 | 0.1091 | 0 |
| Qwen | RS-Neutral | 4 | 10 | 92/150 | 91/153 | 1.6423 | 1.8004 | 0.1581 | 0 |
| Qwen | RS-Neutral | 5 | 10 | 92/157 | 93/156 | 1.3263 | 1.5047 | 0.1784 | 0 |
| Qwen | RS-Neutral | 6 | 10 | 95/161 | 96/157 | 0.7629 | 0.9062 | 0.1433 | 0 |
| Qwen | RS-Neutral | 7 | 10 | 97/164 | 94/157 | 0.5423 | 1.1374 | 0.5951 | 0 |
| Qwen | RS-Neutral | 8 | 10 | 99/162 | 98/155 | 0.3349 | 0.5851 | 0.2502 | 3 |
| Qwen | RS-Soft | 0 | 10 | 11/36 | 11/36 | 10.3780 | 10.3780 | 0.0000 | 0 |
| Qwen | RS-Soft | 1 | 10 | 29/49 | 31/49 | 7.5494 | 7.5695 | 0.0200 | 0 |
| Qwen | RS-Soft | 2 | 10 | 62/101 | 62/110 | 4.6870 | 4.7198 | 0.0328 | 0 |
| Qwen | RS-Soft | 3 | 10 | 84/135 | 85/137 | 2.6734 | 2.7275 | 0.0541 | 0 |
| Qwen | RS-Soft | 4 | 10 | 93/149 | 93/151 | 1.6280 | 1.7273 | 0.0993 | 0 |
| Qwen | RS-Soft | 5 | 10 | 93/155 | 92/155 | 1.2888 | 1.3993 | 0.1105 | 0 |
| Qwen | RS-Soft | 6 | 10 | 97/160 | 96/160 | 0.7965 | 1.0229 | 0.2264 | 0 |
| Qwen | RS-Soft | 7 | 10 | 97/164 | 95/160 | 0.5692 | 0.6801 | 0.1109 | 0 |
| Qwen | RS-Soft | 8 | 10 | 99/165 | 100/163 | 0.3859 | 0.4922 | 0.1063 | 1 |

first-hit/best/plateau는 case table에 기록했다. terminal discrete plateau는 terminal E/G count pair가 이후 exact하게 유지되는 가장 이른 k이며 tolerance가 없다. z-oracle E=10 최초 k, E NLL 최소 k도 별도 저장됐다.

## 4. typed failure의 last-valid prefix

- Llama BG-Soft case06은 raw accepted k1..k5의 internal full-six target-new와 routing/progress/P/capacity만 있다. action-freeze 전 실패했으므로 heldout z-oracle E/G/Loc 및 W stepwise evaluator prefix는 NOT_RECORDED이다.

| k | full6 z target-new | full6 W target-new | rho | predicted | actual | realization | P_after | capacity |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.4193 | N/R | 2.7229 | 2.7229 | N/R | N/R | 0.0002 | 0.0501 |
| 2 | 7.6280 | N/R | 2.1153 | 2.1153 | N/R | N/R | 0.0005 | 0.0514 |
| 3 | 5.4017 | N/R | 3.8604 | 3.8604 | N/R | N/R | 0.0009 | 0.0517 |
| 4 | 3.2238 | N/R | 6.4668 | 6.4668 | N/R | N/R | 0.0019 | 0.2461 |
| 5 | 2.3452 | N/R | 3.8084 | 3.8084 | N/R | N/R | 0.0030 | 0.2397 |

## 5. writer realization, routing, Structural-P

- FACT: 모든 성공 step의 matched-strength equality residual은 table에 보존되며 max는 machine table에서 직접 재검증 가능하다. h는 1회, second remaining division은 0으로 기록됐다.
- Qwen은 15 negative-actual steps(BG N4, BG S7, RS N3, RS S1)을 보여 locally predicted progress의 BF16 realization이 불안정하다. Llama에서는 BG-N 1, RS-N 2, RS-S 1개가 별도로 관찰되었다. 이는 candidate retry/rollback 없이 관찰된 값이다.
- Soft는 구조 P를 대체로 낮췄다: Qwen BG 0.998→0.465, RS 0.375→0.192; Llama BG 0.0113→0.0081(성공 조건부). 그러나 functional-P는 Qwen BG/RS와 Llama RS에서 오히려 높아 proxy alignment가 일반적이지 않다.
- Llama RS Soft는 Neutral보다 Gen -1, Loc 동률, P와 capacity가 소폭 높다. Qwen RS Soft는 E/Gen/Loc +2/+8/+1이고 구조 P/capacity는 낮지만 functional-P는 높다. Qwen BG Soft는 E 동률, Gen/Loc -1/-1이며 구조 P만 낮다. 따라서 Soft preservation은 universal하지 않다.
- per-step CSV/JSON에는 slopes, active/DOF, q/rho, pi/v/c=h·v, layer progress, entropy/Neff/top1, energy/capacity, P offset/cross/self, BF16 energy/capacity가 포함된다.

## 6. compute 및 evaluator firewall

| 모델 | 셀 | edit-core mean(s) | post-freeze k0..k8 eval mean(s) | ratio eval/edit |
|---|---|---:|---:|---:|
| Llama | BG-Neutral | 86.79 | 417.74 | 4.81 |
| Llama | BG-Soft | 81.08 | 416.76 | 5.14 |
| Llama | RS-Neutral | 84.17 | 416.21 | 4.94 |
| Llama | RS-Soft | 80.18 | 415.78 | 5.19 |
| Qwen | BG-Neutral | 105.89 | 472.50 | 4.46 |
| Qwen | BG-Soft | 104.39 | 473.70 | 4.54 |
| Qwen | RS-Neutral | 102.53 | 471.43 | 4.60 |
| Qwen | RS-Soft | 99.32 | 472.00 | 4.75 |

- post-freeze stepwise evaluator가 edit-core보다 약 4.5–5.2배 길며 controller influence 0이다. 이 비용은 scientific K8 compute가 아니라 분석 전용 replay다.
- case table에는 model F/B, target/slope backward, processed tokens, materialization, target/field/slope/router/materialization wall을 분리했다. scheduler MaxRSS/peak GPU memory는 authoritative case receipt에 일관되게 직렬화되지 않아 NOT_RECORDED이며 task-level scheduler 정보와 혼합하지 않았다.

## 7. FACT / INFERENCE / 경계

### FACT
- 79/80 endpoint, 실패 1/80; 모든 case는 성공/typed failure 뒤 W0 restore PASS.
- Llama W endpoint: BG-N 98/175/871, BG-S 89/164/787(9 endpoints), RS-N 99/178/873, RS-S 99/177/873.
- Qwen W endpoint: BG-N 94/165/843, BG-S 94/164/842, RS-N 98/155/843, RS-S 100/163/844.
- Official AlphaEdit same-stream: Llama 100/185/859, Qwen 100/192/828. Direct-z semantics flag는 20/20 true.
### INFERENCE
- 높은 locality count를 strength-matched preservation gain으로 승격할 수 없다. z 자체가 Official보다 약하고 W가 추가로 뒤처지는 셀이 많다.
- Qwen under-edit은 target-side와 writer-side 모두에 있다. Llama RS는 target deficit이 더 작고 writer deficit은 상대적으로 작다.
- Soft의 cumulative Structural-P 감소가 functional-P 또는 Gen/Loc 개선으로 일관되게 변환되지 않는다.
### SCIENTIFIC_FAIL
- Llama BG-Soft case06의 typed certificate boundary. 분모에는 남기고 endpoint는 impute하지 않았다.
### TECHNICAL_FAIL
- 최초 19231 pre-model session binding; 과학 action 0. 유효 19239 결과와 분리했다.
### NOT_RECORDED
- frozen internal target-new plan의 full-six target-old stepwise 값.
- failed case06의 heldout z-oracle/W-only k0..k5 evaluator prefix 및 endpoint.
- Official AlphaEdit latent-z k0..trajectory; Official은 direct-z-driven terminal endpoint만 제공한다.
- P1R24 z−zbase cumulative norm per k, endpoint scheduler MaxRSS/peak GPU memory의 case-level authoritative 직렬화.
- source schema 간 동일 W0 parameter hash 직접 교차값은 없지만 W0 E/G/L score identity 및 case/evaluator identities는 exact match다.

## 8. 최종 판정

P1R24는 이 10-batch development stream에서 efficacy count는 대체로 높지만, direct-z-driven Official AlphaEdit와 비교하면 내부 target z부터 특히 Gen/continuous NLL이 약하고 BF16 W가 추가 결손을 만든다. 따라서 관측된 under-edit은 **target-side + writer-side 혼합**이며 Qwen에서 두 결손이 가장 뚜렷하다. Soft는 구조 P를 낮추는 경우가 많으나 functional-P와 endpoint 보존을 보편적으로 개선하지 않는다. 이 결과는 독립 Atomic audit이며 sequential/Historical 또는 scientific promotion을 지지하지 않는다.

## 9. 산출물 및 재현 경계

- `p1r31-per-case.json/csv`: 80개 시도, typed failure 분모, terminal/best/first-hit/plateau, Official paired delta.
- `p1r31-per-step.json/csv`: 716 rows(79×9 post-freeze + failed prefix 5), per-request aggregate NLL, z-W gap, routing/P/capacity/compute/hashes.
- 분석 중 model/evaluator/Slurm action=0, scientific source/result mutation=0. raw prompts/targets/generations/tensors/weights는 읽거나 직렬화하지 않았다.
