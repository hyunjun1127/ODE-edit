# P2R4 Phase-B Clamp ON/OFF × P2R2 V2 Writer — 터미널 사실 보고

- instruction_id: `ODEEDIT-S05-P2R4-P2R1-CLAMP-ON-OFF-CAUSAL-ABLATION-V1-PHASE-B`
- method_id: `P2R4-P2R1-CLAMP-ON-OFF-P2R2-V2-WRITER-V1`
- source head: `96f23e804be66569ad3fcf98bfcdd8a02ff3f7fb`; sealed P2R2 V2: `c97e8619b42da7954ce0e824c215a8a82d70589a`.
- 4 job roots × 2 arms × 10 independent B10 = 80 completed case-arm attempts; 800 request attempts; no imputation.
- `scientific_promotion=false`. 본 문서는 수치·식별자·기계적 수신증만 기록한다.

## 터미널 무결성

| model / clamp | completed / failed | request attempts | W0 restored | job wall s | job model F | job tokens | terminal SHA |
|---|---:|---:|---|---:|---:|---:|---|
| llama3-8b-inst / OFF | 20 / 0 | 200 | True | 2009.508 | 7932 | 1204799 | `b3eec43dbb3a57dec9145ad52996d159b658afb87af6324316122a336f027cb7` |
| llama3-8b-inst / ON | 20 / 0 | 200 | True | 2124.702 | 7932 | 1204799 | `96f93df5767d0ca51b4695cf84f1a16ef7d7ef5f813ea336e2cbb54f7e266ec4` |
| qwen2.5-7b-inst / OFF | 20 / 0 | 200 | True | 2211.001 | 7934 | 1105689 | `7d28dab4b17f5d7b074ad1c2d83425ad472e6211ba163eaba83309cd3f633547` |
| qwen2.5-7b-inst / ON | 20 / 0 | 200 | True | 2461.537 | 7934 | 1105689 | `ebc248d63389a5144abdd9864bab31450a082c550c49ec1a5bcf60c3c7fa3f94` |

모든 case manifest→terminal SHA 및 최상위 manifest→terminal SHA를 재해시해 일치시켰다. 모든 case에서 byte/pointer W0 restore 및 action freeze=true, heldout controller access=0이다.

## W-only 및 z-inject endpoint

| model / clamp / arm | W Eff | W Gen | W Loc | z Eff | z Gen | z Loc | W Eff new NLL / margin | z Eff new NLL / margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst / OFF / NEUTRAL | 100/100 | 193/200 | 741/1000 | 100/100 | 194/200 | 741/1000 | 0.085430 / 16.925206 | 0.000449 / 17.050810 |
| llama3-8b-inst / OFF / SOFTP | 100/100 | 193/200 | 738/1000 | 100/100 | 194/200 | 738/1000 | 0.086057 / 16.913953 | 0.000452 / 17.053934 |
| llama3-8b-inst / ON / NEUTRAL | 99/100 | 188/200 | 841/1000 | 100/100 | 189/200 | 841/1000 | 0.171467 / 16.697324 | 0.091897 / 16.903768 |
| llama3-8b-inst / ON / SOFTP | 99/100 | 188/200 | 840/1000 | 100/100 | 189/200 | 840/1000 | 0.223809 / 16.586852 | 0.163067 / 16.723224 |
| qwen2.5-7b-inst / OFF / NEUTRAL | 100/100 | 185/200 | 839/1000 | 100/100 | 185/200 | 839/1000 | 0.001141 / 17.268259 | 0.001150 / 17.248876 |
| qwen2.5-7b-inst / OFF / SOFTP | 100/100 | 185/200 | 839/1000 | 100/100 | 185/200 | 839/1000 | 0.001146 / 17.267003 | 0.001157 / 17.255119 |
| qwen2.5-7b-inst / ON / NEUTRAL | 100/100 | 185/200 | 839/1000 | 100/100 | 185/200 | 839/1000 | 0.001141 / 17.268259 | 0.001150 / 17.248876 |
| qwen2.5-7b-inst / ON / SOFTP | 100/100 | 185/200 | 839/1000 | 100/100 | 185/200 | 839/1000 | 0.001146 / 17.267003 | 0.001157 / 17.255119 |

## z→W / writer / clamp 수신증

| model / clamp / arm | z→W Eff NLL gap | z→W Gen NLL gap | terminal full-six target-new NLL | realization mean / p90 | cumulative Structural-P mean | final BF16 capacity sum mean | BF16 step-energy path sum mean | negative actual | unreachable events | target event occurrences / requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst / OFF / NEUTRAL | 0.084981 | 0.087175 | 0.000867 | 0.678694 / 0.945410 | 0.224196 | 119.741551 | 117.954810 | 6 | 1 | 2400 / 100 (clamp_would_hit) |
| llama3-8b-inst / OFF / SOFTP | 0.085606 | 0.083673 | 0.000867 | 0.653480 / 0.951485 | 0.224265 | 119.773769 | 117.960578 | 6 | 1 | 2400 / 100 (clamp_would_hit) |
| llama3-8b-inst / ON / NEUTRAL | 0.079570 | 0.031576 | 0.064214 | 0.586866 / 1.127004 | 0.072068 | 44.354241 | 52.943545 | 40 | 2 | 1067 / 100 (clamp_hit) |
| llama3-8b-inst / ON / SOFTP | 0.060742 | 0.036448 | 0.129733 | 0.723108 / 1.127797 | 0.070904 | 43.529577 | 52.729505 | 40 | 2 | 1077 / 100 (clamp_hit) |
| qwen2.5-7b-inst / OFF / NEUTRAL | -0.000009 | -0.004662 | 0.000684 | 1.139828 / 1.017783 | 2.186900 | 188.790180 | 191.046709 | 9 | 3 | 0 / 0 (clamp_would_hit) |
| qwen2.5-7b-inst / OFF / SOFTP | -0.000011 | -0.005094 | 0.000686 | 0.760933 / 1.007155 | 2.185568 | 188.680925 | 191.209003 | 9 | 2 | 0 / 0 (clamp_would_hit) |
| qwen2.5-7b-inst / ON / NEUTRAL | -0.000009 | -0.004662 | 0.000684 | 1.139828 / 1.017783 | 2.186900 | 188.790180 | 191.046709 | 9 | 3 | 0 / 0 (clamp_hit) |
| qwen2.5-7b-inst / ON / SOFTP | -0.000011 | -0.005094 | 0.000686 | 0.760933 / 1.007155 | 2.185568 | 188.680925 | 191.209003 | 9 | 2 | 0 / 0 (clamp_hit) |

writer update/path/net norm은 수신증에 독립 scalar norm으로 기록되지 않아 `NOT_RECORDED`이다. target path displacement norm은 표 파일에 있으며, ON target origin-net norm은 수신증에 없다. BF16 energy/capacity는 위 표의 명시된 원시 receipt 필드 합산값이다.

## 라우팅 및 receipt-adapter

| model / clamp / arm | route status counts | routing DOF | layer coefficient shares L4/L5/L6/L7/L8 | soft no-weaker max violation | simplex residual max | outer-h | applied-coordinate | numeric-h / physical-h / second-h | pre-split / remaining / debt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst / OFF / NEUTRAL | `{"NEUTRAL_CERTIFIED": 80}` | NOT_RECORDED | 0.043845/0.015016/0.008766/0.016266/0.916108 | 0.000000000000 | 0.000000000000 | 80 | 80 | 0 / 0 / 0 | 0 / 0 / 0 |
| llama3-8b-inst / OFF / SOFTP | `{"SOFT_CERTIFIED": 25, "SOFT_NEUTRAL_FALLBACK": 55}` | NOT_RECORDED | 0.043852/0.015017/0.008795/0.012494/0.919842 | 0.000000010000 | 0.000000000000 | 80 | 80 | 0 / 0 / 0 | 0 / 0 / 0 |
| llama3-8b-inst / ON / NEUTRAL | `{"NEUTRAL_CERTIFIED": 80}` | NOT_RECORDED | 0.054922/0.031345/0.038906/0.036662/0.838164 | 0.000000000000 | 0.000000000000 | 80 | 80 | 0 / 0 / 0 | 0 / 0 / 0 |
| llama3-8b-inst / ON / SOFTP | `{"SOFT_CERTIFIED": 9, "SOFT_NEUTRAL_FALLBACK": 71}` | NOT_RECORDED | 0.054924/0.030081/0.035145/0.035459/0.844391 | 0.000000010000 | 0.000000000000 | 80 | 80 | 0 / 0 / 0 | 0 / 0 / 0 |
| qwen2.5-7b-inst / OFF / NEUTRAL | `{"NEUTRAL_CERTIFIED": 80}` | NOT_RECORDED | 0.008705/0.008488/0.013974/0.046408/0.922425 | 0.000000000000 | 0.000000000000 | 80 | 80 | 0 / 0 / 0 | 0 / 0 / 0 |
| qwen2.5-7b-inst / OFF / SOFTP | `{"SOFT_CERTIFIED": 7, "SOFT_NEUTRAL_FALLBACK": 73}` | NOT_RECORDED | 0.008689/0.007222/0.020185/0.046303/0.917601 | 0.000000010000 | 0.000000000001 | 80 | 80 | 0 / 0 / 0 | 0 / 0 / 0 |
| qwen2.5-7b-inst / ON / NEUTRAL | `{"NEUTRAL_CERTIFIED": 80}` | NOT_RECORDED | 0.008705/0.008488/0.013974/0.046408/0.922425 | 0.000000000000 | 0.000000000000 | 80 | 80 | 0 / 0 / 0 | 0 / 0 / 0 |
| qwen2.5-7b-inst / ON / SOFTP | `{"SOFT_CERTIFIED": 7, "SOFT_NEUTRAL_FALLBACK": 73}` | NOT_RECORDED | 0.008689/0.007222/0.020185/0.046303/0.917601 | 0.000000010000 | 0.000000000001 | 80 | 80 | 0 / 0 / 0 | 0 / 0 / 0 |

`outer_h_application_count=1` 및 `applied_coordinate_update_count=1`은 committed joint writer transition의 논리적 ODE-transition receipt이다. `writer_h_numeric_multiplication_count=0`, `physical_h_application_count=0`, `second_h_application_count=0`은 추가 수치 h 곱셈이 없음을 기록한다.

## 정확한 case-paired OFF−ON 산술

| model / arm | matched cases | Δ W Eff correct | Δ W Gen correct | Δ W Loc correct | mean Δ W Eff new NLL | mean Δ z→W Eff NLL gap | mean Δ Structural-P | mean Δ BF16 capacity | mean Δ BF16 energy path |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst / NEUTRAL | 10 | 1 | 5 | -100 | -0.086037 | 0.005411 | 0.152128 | 75.387310 | 65.011265 |
| llama3-8b-inst / SOFTP | 10 | 1 | 5 | -102 | -0.137752 | 0.024863 | 0.153361 | 76.244193 | 65.231073 |
| qwen2.5-7b-inst / NEUTRAL | 10 | 0 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| qwen2.5-7b-inst / SOFTP | 10 | 0 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

OFF−ON은 동일 alias/arm/case_index, request order 및 request cardinality가 일치한 40개 case pair의 산술값이다. denominator 또는 대응 데이터가 없는 값은 표·JSON에서 `NOT_RECORDED`로 표시한다.

## Compute ledger

| model / clamp / arm | target F/B | KL F/B | response F/VJP | capture F | post-write F | writer materializations | target/writer/evaluator wall s (sum) |
|---|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst / OFF / NEUTRAL | 1250/1200 | 1250/1200 | 400/400 | 450 | 400 | 80 | 249.356/593.399/23.949 |
| llama3-8b-inst / OFF / SOFTP | 1250/1200 | 1250/1200 | 400/400 | 450 | 400 | 80 | 251.927/611.558/23.945 |
| llama3-8b-inst / ON / NEUTRAL | 1250/1200 | 1250/1200 | 400/400 | 450 | 400 | 80 | 264.414/631.600/23.591 |
| llama3-8b-inst / ON / SOFTP | 1250/1200 | 1250/1200 | 400/400 | 450 | 400 | 80 | 265.691/657.822/23.623 |
| qwen2.5-7b-inst / OFF / NEUTRAL | 1250/1200 | 1250/1200 | 400/400 | 450 | 400 | 80 | 198.959/734.088/22.189 |
| qwen2.5-7b-inst / OFF / SOFTP | 1250/1200 | 1250/1200 | 400/400 | 450 | 400 | 80 | 208.291/748.388/22.168 |
| qwen2.5-7b-inst / ON / NEUTRAL | 1250/1200 | 1250/1200 | 400/400 | 450 | 400 | 80 | 214.581/835.307/22.443 |
| qwen2.5-7b-inst / ON / SOFTP | 1250/1200 | 1250/1200 | 400/400 | 450 | 400 | 80 | 217.325/862.876/22.418 |

## 19965 technical-repair boundary

- initial B1 submission receipt: job `19965`, source `49490d23c6a156c6e9e1a1ce93ce2d85145f1659`, array `0-3%2`.
- replacement B1 receipt: job `19969`, source `96f23e804be66569ad3fcf98bfcdd8a02ff3f7fb`, attempt suffix `server2-node-r1`.
- production receipt: job `19973`, source `96f23e804be66569ad3fcf98bfcdd8a02ff3f7fb`, 80 case-arm attempts.
- source commit `96f23e804be66569ad3fcf98bfcdd8a02ff3f7fb` changes scheduler targeting in the P2R4 Phase-B sbatch/submitter from `devbox` to `server2` and updates its focused launcher test plus source manifest. No P2R2 writer equation file is modified by that commit.
- scheduler State/ExitCode for job `19965` is `NOT_RECORDED` in the permitted raw-free receipt set; no log was read.

## Artifact scope and root

- analysis root: `d0cdd0f5dfd58f0116dc71a6c766ac8dae02c6e73a812c22f04d1ef9b4f2619c`
- tables contain scalar receipt fields, endpoint counts/continuous metrics, hashes and typed status only; no prompts, targets, generations, tensors, weights, model/data/cache, or runtime logs are copied.
