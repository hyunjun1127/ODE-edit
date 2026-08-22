# P3R1 Two-Timescale Fast-Z FH C0/C1/C3 FULL-FP32 terminal factual report

- 작성 시각: 2026-08-23 00:40:03 KST
- instruction: `ODEEDIT-S05-P3R1-TWO-TIMESCALE-FASTZ-FH-C013-FP32-V1`
- 범위: Llama3-8B-Instruct 및 Qwen2.5-7B-Instruct, 모델별 독립 1×B100, target-only M5/M25, C0-FH/C1-FH/C3-FH K8, Official AlphaEdit/MEMIT
- scheduler valid endpoints: Llama job `22764`, Qwen job `22815`
- 유효 분모: aliases 2/2, requests 200/200; typed scientific failures 0; endpoint imputation 0
- scheduler technical history: attempts 6, completed valid 2, failed technical 4
- scientific_promotion: `false`

## 1. 한눈에 보는 절대값과 결론

| model | endpoint | rewrite NLL | rewrite success/acc | rephrase NLL | rephrase success/strict | rephrase acc/strict | LOC | W−z gap rw/rp | FP32 energy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama3-8B-Instruct | PRE_EDIT_W0 | 11.102700 | 13/100 · 0/100 | 9.821012 | 28/200 · 8/100 | 1/200 · 0/100 | 893/1000 | NOT_RECORDED/NOT_RECORDED | NOT_RECORDED |
| Llama3-8B-Instruct | TARGET_ONLY_M5_Z | 0.228546 | 100/100 · 96/100 | 1.516557 | 196/200 · 97/100 | 130/200 · 50/100 | 893/1000 | NOT_RECORDED/NOT_RECORDED | NOT_RECORDED |
| Llama3-8B-Instruct | TARGET_ONLY_M25_Z | 0.000807 | 100/100 · 100/100 | 1.097755 | 197/200 · 98/100 | 150/200 · 62/100 | 893/1000 | NOT_RECORDED/NOT_RECORDED | NOT_RECORDED |
| Llama3-8B-Instruct | C0-FH_Z | 13.980246 | 45/100 · 0/100 | 13.916630 | 103/200 · 39/100 | 0/200 · 0/100 | 498/1000 | NOT_RECORDED/NOT_RECORDED | NOT_RECORDED |
| Llama3-8B-Instruct | C0-FH_W | 13.980632 | 45/100 · 0/100 | 13.916843 | 103/200 · 39/100 | 0/200 · 0/100 | 498/1000 | 0.000386/0.000212 | 9.315118e+10 |
| Llama3-8B-Instruct | C1-FH_Z | 14.408823 | 53/100 · 0/100 | 14.001626 | 98/200 · 26/100 | 0/200 · 0/100 | 525/1000 | NOT_RECORDED/NOT_RECORDED | NOT_RECORDED |
| Llama3-8B-Instruct | C1-FH_W | 14.408825 | 53/100 · 0/100 | 14.001627 | 98/200 · 26/100 | 0/200 · 0/100 | 525/1000 | 1.926422e-06/1.001358e-06 | 1.006924e+15 |
| Llama3-8B-Instruct | C3-FH_Z | 0.148677 | 100/100 · 98/100 | 1.572331 | 193/200 · 94/100 | 139/200 · 57/100 | 867/1000 | NOT_RECORDED/NOT_RECORDED | NOT_RECORDED |
| Llama3-8B-Instruct | C3-FH_W | 0.148246 | 100/100 · 98/100 | 2.228247 | 184/200 · 87/100 | 117/200 · 42/100 | 867/1000 | -0.000431/0.655917 | 123.918069 |
| Llama3-8B-Instruct | Official-AlphaEdit_Z | 0.001156 | 100/100 · 100/100 | 1.099024 | 197/200 · 98/100 | 151/200 · 63/100 | 871/1000 | NOT_RECORDED/NOT_RECORDED | NOT_RECORDED |
| Llama3-8B-Instruct | Official-AlphaEdit_W | 0.001398 | 100/100 · 100/100 | 1.920829 | 186/200 · 88/100 | 122/200 · 44/100 | 871/1000 | 0.000242/0.821805 | 99.173529 |
| Llama3-8B-Instruct | Official-MEMIT_Z | 0.000870 | 100/100 · 100/100 | 1.092113 | 197/200 · 98/100 | 150/200 · 62/100 | 886/1000 | NOT_RECORDED/NOT_RECORDED | NOT_RECORDED |
| Llama3-8B-Instruct | Official-MEMIT_W | 0.346855 | 98/100 · 96/100 | 2.975614 | 164/200 · 76/100 | 100/200 · 36/100 | 886/1000 | 0.345985/1.883501 | 75.746522 |
| Qwen2.5-7B-Instruct | PRE_EDIT_W0 | 10.371340 | 11/100 · 0/100 | 10.019638 | 39/200 · 10/100 | 1/200 · 0/100 | 849/1000 | NOT_RECORDED/NOT_RECORDED | NOT_RECORDED |
| Qwen2.5-7B-Instruct | TARGET_ONLY_M5_Z | 0.688823 | 97/100 · 86/100 | 2.439692 | 183/200 · 87/100 | 114/200 · 46/100 | 849/1000 | NOT_RECORDED/NOT_RECORDED | NOT_RECORDED |
| Qwen2.5-7B-Instruct | TARGET_ONLY_M25_Z | 0.001963 | 100/100 · 100/100 | 1.436343 | 195/200 · 97/100 | 146/200 · 62/100 | 849/1000 | NOT_RECORDED/NOT_RECORDED | NOT_RECORDED |
| Qwen2.5-7B-Instruct | C0-FH_Z | 15.680719 | 55/100 · 0/100 | 15.909144 | 93/200 · 28/100 | 0/200 · 0/100 | 514/1000 | NOT_RECORDED/NOT_RECORDED | NOT_RECORDED |
| Qwen2.5-7B-Instruct | C0-FH_W | 15.680800 | 55/100 · 0/100 | 15.909220 | 93/200 · 28/100 | 0/200 · 0/100 | 514/1000 | 0.000081/0.000075 | 3.856717e+15 |
| Qwen2.5-7B-Instruct | C1-FH_Z | 16.475022 | 44/100 · 0/100 | 16.449311 | 101/200 · 31/100 | 0/200 · 0/100 | 507/1000 | NOT_RECORDED/NOT_RECORDED | NOT_RECORDED |
| Qwen2.5-7B-Instruct | C1-FH_W | 16.475077 | 44/100 · 0/100 | 16.449360 | 101/200 · 31/100 | 0/200 · 0/100 | 507/1000 | 0.000055/0.000049 | 2.704890e+15 |
| Qwen2.5-7B-Instruct | C3-FH_Z | 0.000085 | 100/100 · 100/100 | 2.076052 | 181/200 · 86/100 | 128/200 · 48/100 | 796/1000 | NOT_RECORDED/NOT_RECORDED | NOT_RECORDED |
| Qwen2.5-7B-Instruct | C3-FH_W | 0.000085 | 100/100 · 100/100 | 2.133684 | 177/200 · 85/100 | 126/200 · 48/100 | 796/1000 | 4.138086e-08/0.057631 | 5706.612613 |
| Qwen2.5-7B-Instruct | Official-AlphaEdit_Z | 0.052518 | 100/100 · 99/100 | 1.523862 | 195/200 · 97/100 | 144/200 · 59/100 | 826/1000 | NOT_RECORDED/NOT_RECORDED | NOT_RECORDED |
| Qwen2.5-7B-Instruct | Official-AlphaEdit_W | 0.052830 | 100/100 · 99/100 | 1.754300 | 193/200 · 96/100 | 135/200 · 53/100 | 826/1000 | 0.000312/0.230438 | 2376.313157 |
| Qwen2.5-7B-Instruct | Official-MEMIT_Z | 0.021766 | 100/100 · 100/100 | 1.524761 | 195/200 · 97/100 | 145/200 · 61/100 | 830/1000 | NOT_RECORDED/NOT_RECORDED | NOT_RECORDED |
| Qwen2.5-7B-Instruct | Official-MEMIT_W | 0.019718 | 100/100 · 100/100 | 1.757920 | 193/200 · 96/100 | 133/200 · 52/100 | 830/1000 | -0.002048/0.233159 | 5273.517699 |

M25 target-only의 M5 대비 rewrite/rephrase NLL delta는 Llama `-0.227739/-0.418802`, Qwen `-0.686860/-1.003349`이다. C1−C0 W rephrase NLL delta는 Llama `+0.084785`, Qwen `+0.540141`이고, C1−C0 FP32 energy ratio는 Llama 약 `10809.5×`, Qwen 약 `0.7014×`이다. C3와 Official AlphaEdit의 W rephrase NLL delta는 Llama `+0.307419`, Qwen `+0.379384`이다.

`success`는 target-new NLL < target-true NLL인 prompt count이며 `accuracy`는 target-new suffix teacher-forced exact-token count다. strict rephrase는 두 rephrase prompt가 모두 통과한 request count다. z와 W는 별도 endpoint다.

## 2. 실행·계약 무결성

- contract: `/mnt/raid5/janghj/ODE-edit/local/state/p3r1-two-timescale-fastz-fh-c013-fp32-v1/authoritative-contract.txt`, SHA256 `d1a4216bad9ab5330794c13c0f3aa59bbc2f0e32ca3f901593ed24a262f14e08`, bytes `18602`, lines `697`, mode `0600`.
- stream root/order: `467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a` / `018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3`. 두 모델 request-order digest는 모델별 tokenizer/model 입력 기준으로 별도 기록됐으며 request count/order gate는 각각 PASS다.
- Llama3-8B-Instruct: source HEAD/tree `05a804b0bf554a1ee3af0067dfc41fb1db8b21f8` / `465dfeeb4d69b50ebedf0f6d378585f4aa3e42e2`, case-terminal SHA `80a7d3d35d456b9302c0c7cc933f2fe3f29f9bd0acbda2f1575a43883d606e76`, result manifest SHA `2a3ee94993e5b98aefe02b271b6c1c2e7b140e41e5dc16c6f1b3c6490324d555`, status `TERMINAL_VALID`, 100/100, W0 restore PASS.
- Qwen2.5-7B-Instruct: source HEAD/tree `f15ed227e62f7187fede2a631c157d2ac31ae824` / `3bed9a6b4927b8c76f002786b559cdce0a03ea6b`, case-terminal SHA `3641520e3eceb990452737028186582903e90ded14ce8822370cb1e061f34868`, result manifest SHA `7835ad108dcbe2429227040b4645cacab09106e0515c29e7a4d55528c5333ace`, status `TERMINAL_VALID`, 100/100, W0 restore PASS.
- 모든 target/controller/writer/update storage는 `torch.float32`; autocast/BF16/FP16 conversion/materializer count는 0이다.
- fixed M5, K8, `FIXED_M_FINAL_ITERATE`; PRIMARY/RESCUE/CURRENT/early-stop/first-hit/retry/backtracking/best-iterate/heldout decision influence는 0이다.
- 각 arm outer 8개, C0/C1 fallback 0, C3 native compute_z 0, C3 P+C/barrier decision influence 0, pre-K8 current-batch cache append 0이다.
- Llama/Qwen은 서로 다른 모델별 W0를 사용한다. 동일 W0 보장은 각 모델 내부 arm/baseline 격리와 각 arm 종료 후 restore에 적용된다.

## 3. Target-only M5 대 M25

| model | depth | rw NLL/success/acc | rp NLL/success/strict | rp acc/strict | final objective mean | final-vs-best gap mean |
|---|---|---:|---:|---:|---:|---:|
| Llama3-8B-Instruct | M5 | 0.228546 · 100/100 · 96/100 | 1.516557 · 196/200 · 97/100 | 130/200 · 50/100 | 0.342548 | 0.013746 |
| Llama3-8B-Instruct | M25 | 0.000807 · 100/100 · 100/100 | 1.097755 · 197/200 · 98/100 | 150/200 · 62/100 | 0.071719 | 0.000068 |
| Qwen2.5-7B-Instruct | M5 | 0.688823 · 97/100 · 86/100 | 2.439692 · 183/200 · 87/100 | 114/200 · 46/100 | 0.576161 | 3.114675e-06 |
| Qwen2.5-7B-Instruct | M25 | 0.001963 · 100/100 · 100/100 | 1.436343 · 195/200 · 97/100 | 146/200 · 62/100 | 0.030676 | 0.000076 |

M5는 초기 observation m0과 5회 update를, M25는 m0과 25회 update를 기록한다. 두 target-only arm의 writer/materialization/history/factor mutation/heldout decision count는 0이다. per-inner 값은 `per-inner.csv/json`에 전부 보존했다.

## 4. K8 동적 target과 writer transfer

| model | writer | rw z→W fail req | rp z→W fail prompts | strict z→W fail req | W−z rw/rp | W LOC | energy |
|---|---|---:|---:|---:|---:|---:|---:|
| Llama3-8B-Instruct | C0-FH | 0 | 0 | 0 | 0.000386/0.000212 | 498/1000 | 9.315118e+10 |
| Llama3-8B-Instruct | C1-FH | 0 | 0 | 0 | 1.926422e-06/1.001358e-06 | 525/1000 | 1.006924e+15 |
| Llama3-8B-Instruct | C3-FH | 0 | 2 | 8 | -0.000431/0.655917 | 867/1000 | 123.918069 |
| Llama3-8B-Instruct | Official-AlphaEdit | 0 | 1 | 10 | 0.000242/0.821805 | 871/1000 | 99.173529 |
| Llama3-8B-Instruct | Official-MEMIT | 2 | 11 | 22 | 0.345985/1.883501 | 886/1000 | 75.746522 |
| Qwen2.5-7B-Instruct | C0-FH | 0 | 0 | 0 | 0.000081/0.000075 | 514/1000 | 3.856717e+15 |
| Qwen2.5-7B-Instruct | C1-FH | 0 | 0 | 0 | 0.000055/0.000049 | 507/1000 | 2.704890e+15 |
| Qwen2.5-7B-Instruct | C3-FH | 0 | 4 | 1 | 4.138086e-08/0.057631 | 796/1000 | 5706.612613 |
| Qwen2.5-7B-Instruct | Official-AlphaEdit | 0 | 1 | 1 | 0.000312/0.230438 | 826/1000 | 2376.313157 |
| Qwen2.5-7B-Instruct | Official-MEMIT | 0 | 1 | 1 | -0.002048/0.233159 | 830/1000 | 5273.517699 |

C0/C1은 각 outer의 arm-local M5 target과 finite-horizon waypoint를 suffix-current residual writer에 연결한다. C1은 Joint P+C minimax pi를 사용한다. C3는 arm-local target/waypoint를 direct Official AlphaEdit writer에 연결하고 native compute_z를 호출하지 않는다. Official baseline 행은 각 baseline 자체 native-z→W 전송이다.

## 5. C0/C1 routing과 층 집중도

| model | endpoint | top-energy layer/share | total energy | rewrite W NLL | rephrase W NLL | LOC |
|---|---|---:|---:|---:|---:|---:|
| Llama3-8B-Instruct | C0-FH | L8 / 91.5185% | 9.315118e+10 | 13.980632 | 13.916843 | 498/1000 |
| Llama3-8B-Instruct | C1-FH | L8 / 92.5258% | 1.006924e+15 | 14.408825 | 14.001627 | 525/1000 |
| Llama3-8B-Instruct | C3-FH | L8 / 49.4074% | 123.918069 | 0.148246 | 2.228247 | 867/1000 |
| Llama3-8B-Instruct | Official-AlphaEdit | L8 / 45.3320% | 99.173529 | 0.001398 | 1.920829 | 871/1000 |
| Llama3-8B-Instruct | Official-MEMIT | L8 / 49.6121% | 75.746522 | 0.346855 | 2.975614 | 886/1000 |
| Qwen2.5-7B-Instruct | C0-FH | L7 / 96.6117% | 3.856717e+15 | 15.680800 | 15.909220 | 514/1000 |
| Qwen2.5-7B-Instruct | C1-FH | L8 / 96.0403% | 2.704890e+15 | 16.475077 | 16.449360 | 507/1000 |
| Qwen2.5-7B-Instruct | C3-FH | L5 / 25.7801% | 5706.612613 | 0.000085 | 2.133684 | 796/1000 |
| Qwen2.5-7B-Instruct | Official-AlphaEdit | L5 / 31.2053% | 2376.313157 | 0.052830 | 1.754300 | 826/1000 |
| Qwen2.5-7B-Instruct | Official-MEMIT | L4 / 60.0053% | 5273.517699 | 0.019718 | 1.757920 | 830/1000 |

- Llama3-8B-Instruct C0-FH outer0 pi=`[0.199401, 0.225414, 0.213039, 0.176660, 0.185486]`, beta=`[0.199401, 0.281557, 0.370383, 0.487815, 1.000000]`, outer0 residual `5.673822→894.962972`.
- Llama3-8B-Instruct C1-FH outer0 pi=`[0.183233, 0.215782, 0.220903, 0.201008, 0.179074]`, beta=`[0.183233, 0.264191, 0.367568, 0.528855, 1.000000]`, outer0 residual `5.673822→937.079504`.
- Qwen2.5-7B-Instruct C0-FH outer0 pi=`[0.070292, 0.082144, 0.299529, 0.268859, 0.279175]`, beta=`[0.070292, 0.088355, 0.353400, 0.490588, 1.000000]`, outer0 residual `105.216965→9647.190973`.
- Qwen2.5-7B-Instruct C1-FH outer0 pi=`[0.137478, 0.126820, 0.322734, 0.223318, 0.189649]`, beta=`[0.137478, 0.147034, 0.438675, 0.540765, 1.000000]`, outer0 residual `105.216965→16179.548774`.
- 모든 outer/layer의 pi, beta, suffix mass, key/q norm/hash, residual, update norm/energy/share, P/C/KKT receipt는 `per-route` 및 `per-layer` 표에 있다.

## 6. 공식 baseline과 C3 구분

- `Official-AlphaEdit`는 native compute_z와 native writer를 포함한 full baseline이다. `C3-FH`는 P3R1 arm-local fixed-M target/finite-horizon waypoint를 같은 Official writer entrypoint에 연결한다.
- `Official-MEMIT`는 full native method다. C3를 Official AlphaEdit baseline으로 표기하지 않았다.
- Llama3-8B-Instruct: C3−AlphaEdit W rephrase NLL `0.307419`, Gen `-2/200`, strict `-1/100`, LOC `-4/1000`, energy delta `24.744540`; C3−MEMIT W rephrase NLL `-0.747367`.
- Qwen2.5-7B-Instruct: C3−AlphaEdit W rephrase NLL `0.379384`, Gen `-16/200`, strict `-11/100`, LOC `-30/1000`, energy delta `3330.299456`; C3−MEMIT W rephrase NLL `0.375763`.

## 7. outer trajectory

| model | arm | negative-progress outers | alpha=0 outers | first→last oracle residual | first→last W NLL post | final residual |
|---|---|---:|---:|---:|---:|---:|
| Llama3-8B-Instruct | C0-FH | 3/8 | 2/8 | 45.390577→52013.165800 | 14.949615→13.928117 | 1241347.145330 |
| Llama3-8B-Instruct | C1-FH | 2/8 | 3/8 | 45.390577→7177729.599122 | 14.796018→14.286704 | 1.059991e+08 |
| Llama3-8B-Instruct | C3-FH | 0/8 | 0/8 | 45.390577→41.851966 | 8.564867→0.256098 | NOT_RECORDED |
| Qwen2.5-7B-Instruct | C0-FH | 2/8 | 3/8 | 841.735722→5.748339e+07 | 9.090391→15.652215 | 4.977412e+08 |
| Qwen2.5-7B-Instruct | C1-FH | 3/8 | 4/8 | 841.735722→4465669.329797 | 10.830820→16.103371 | 1.087216e+09 |
| Qwen2.5-7B-Instruct | C3-FH | 1/8 | 0/8 | 841.735722→650.994635 | 5.795999→0.000894 | NOT_RECORDED |

`per-outer.csv/json`에는 lambda, oracle/waypoint residual, injection NLL, W NLL pre/post, predicted/actual progress, negative-progress, target objective, hashes와 commit counters가 있다.

## 8. compute 및 자원

| model/job | state | elapsed | batch MaxRSS KiB | model_forward receipt | processed tokens | target/writer/evaluator wall |
|---|---|---:|---:|---:|---:|---:|
| Llama3-8B-Instruct/22764 | COMPLETED | 02:52:15 | 23638916 | 5725 | 867432 | NOT_RECORDED/NOT_RECORDED/NOT_RECORDED |
| Qwen2.5-7B-Instruct/22815 | COMPLETED | 02:44:21 | 31945344 | 3404 | 571681 | NOT_RECORDED/NOT_RECORDED/NOT_RECORDED |

job-level target/writer/evaluator wall 분해, evaluator forward/token, GPU peak receipt는 `NOT_RECORDED`다. scheduler elapsed/MaxRSS와 runtime model_forward/processed_tokens receipt를 혼합하지 않았다. arm/baseline별 wall은 `compute.json`에 원문 receipt 값으로 보존했다.

## 9. 기술 이력과 endpoint 분리

| job | model | state | endpoint use | direct technical cause |
|---:|---|---|---|---|
| 22600 | Llama | FAILED | excluded | output-parent namespace mismatch; model/science step 0 |
| 22601 | Qwen | FAILED | excluded | output-parent namespace mismatch; model/science step 0 |
| 22764 | Llama | COMPLETED | valid | TECH-R1 terminal valid |
| 22765 | Qwen | FAILED | excluded | C0 path cross-called non-authoritative C1 Joint-PC certificate |
| 22784 | Qwen | FAILED | excluded | inactive inequality included in KKT multiplier certificate |
| 22815 | Qwen | COMPLETED | valid | TECH-R3 certificate-only repair terminal valid |

TECH-R1~R3 실패 prefix/root는 endpoint 지표에 포함하지 않았고 imputation하지 않았다. Qwen TECH-R3 repair는 기존 primal tolerance보다 slack이 큰 inactive inequality를 multiplier certificate에서 제외하는 검산-only 변경이다. solver/pi/objective/tolerance/science/stream/evaluator는 변경되지 않았다.

## 10. 판정 경계

- mechanical terminal gate: `PASS` (Llama/Qwen 모두 TERMINAL_VALID, 100/100, W0 restore, FP32/M5/K8/influence/fallback gates PASS).
- Target-only M25−M5: 두 모델 모두 rewrite/rephrase NLL이 음의 delta이며, success/accuracy 절대 count는 표와 machine table에 기록했다.
- C1−C0: 두 모델 모두 terminal rewrite/rephrase NLL delta가 양수다. energy delta는 Llama 양수, Qwen 음수다.
- C3−Official AlphaEdit: 두 모델 모두 W rephrase NLL delta가 양수다. endpoint별 Gen/strict/LOC/energy delta는 §6에 있다.
- failure labels: `TargetWeak`, `TrackingGap`, `BarrierInert`, `BarrierShortcut`은 이 실행에서 별도 typed terminal label로 기록되지 않아 `NOT_RECORDED`다.
- independent 10×B100 expansion은 실행하지 않았다. scientific_promotion=`false`.

## 11. machine-readable 산출물

- `endpoint-summary.*`: 모든 z/W absolute endpoint metrics와 denominator
- `per-request.*`: request/prompt별 NLL, success, accuracy, strict bits
- `per-inner.*`: target-only와 동적 K8의 inner observation 352행
- `per-outer.*`: model×arm×K8 48행
- `per-layer.*`: C0/C1 nonzero-write layer trajectory 100행
- `energy-layer.*`: endpoint final FP32 layer energy 50행
- `per-route.*`: model×arm×K8 route/certificate 48행
- `paired-deltas.*`: source-backed arithmetic deltas
- `technical-history.*`, `compute.json`, `identities.json`, `analysis-manifest-v2.json`, `rooted-analysis-receipt-v2.json`, `independent-rehash-review.json`
