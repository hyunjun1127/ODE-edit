# Realization Debt Phase A — lifelong v6 추가 분석 사실 보고서

## 0. 판정과 해석 경계

- 상태: `ANALYSIS_ONLY_TERMINAL_PASS`
- 새 editing run/checkpoint evaluation/model load/GPU/Slurm action: `0`.
- 이 보고서는 동일 B100 write의 request-layer realization debt와 직후 current-B100 rewrite 관측 간의 **동시점 기술적 연관**만 기술한다.
- 인과, 미래 forgetting, global locality, 자동 promotion을 주장하지 않는다. `scientific_promotion=false`.
- `absolute allocation-energy weighted debt`와 exact `G`-metric action은 raw schema가 없어 `NOT_COMPUTABLE_FROM_SCHEMA`이다.
- 기존 `d_parallel/d_perp`는 inherited L8 trajectory debt이므로 본 보고서의 per-write debt 명칭으로 재사용하지 않았다.

## 1. 한눈에 보는 arm별 사실값

| model | method | debt mean | median | p90 | CVaR90 | max | RS current % | NLL advantage mean | target-new strict % | Σ Frobenius action² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3-8b-inst | alphaedit | 172582.298052 | 0.709769 | 11.419356 | 1725808.115049 | 1276941999.348429 | 99.970000 | 13.841589 | 98.880000 | 130827.243031 |
| llama3-8b-inst | memit | 78510.621033 | 1.091928 | 12.582245 | 785084.827136 | 1992339601.207516 | 95.970000 | 8.507235 | 64.060000 | 256539.637069 |
| qwen2.5-7b-inst | alphaedit | 25234.721528 | 0.620265 | 8.831050 | 252336.111157 | 1261582300.262973 | 99.010000 | 12.389334 | 89.740000 | 2773881.554579 |
| qwen2.5-7b-inst | memit | 8.590358 | 0.705225 | 23.239285 | 62.090128 | 1524.742254 | 68.300000 | 3.580764 | 25.710000 | 7928958.171809 |

평균과 CVaR90은 일부 L4 극단치에 매우 민감하다. 따라서 arm 비교에서는 mean 단독 순위를 사용하지 않고 median/p90/p99/max와 아래 극단-tail provenance를 함께 읽어야 한다.

## 2. 정의와 산출 순서

각 raw request-layer 행에서 먼저 다음을 계산했다.

- `debt_native=(1-rho)^2+tau^2`
- `debt_parallel_native=debt_under+debt_over+debt_opposite=(1-rho)^2`
- `debt_under=1[0<=rho<1](1-rho)^2`
- `debt_over=1[rho>1](rho-1)^2`
- `debt_opposite=1[rho<0](1-rho)^2`
- `debt_orthogonal=tau^2`
- `normalized_potential_reduction=0.5*(q_pre^2-q_post^2)`

전 행에서 `debt_parallel_native == debt_under+debt_over+debt_opposite`와 `debt_native == debt_parallel_native+debt_orthogonal`을 별도로 검증했다. `debt_parallel_native`는 v3 inherited-debt의 `d_parallel`과 다른 per-write 값이다.

그 뒤 arm×batch×layer의 정확히 100개 행에서 `debt_parallel_native`를 포함한 모든 row metric의 mean/median/p90/p99/top-decile CVaR0.9/max를 계산했다. percentile은 NumPy linear quantile, CVaR0.9는 정렬된 상위 10개 행의 산술평균이다. 요약 rho/tau에 debt 식을 다시 적용하지 않았다.

weight action은 request별로 복제하지 않았다. 먼저 100개 request를 batch×layer로 집계한 뒤 한 개의 `frobenius_squared_telemetry`와 1:1 결합했다. 분모 `+1e-12`를 사용한 두 proxy만 `progress_eff_frobenius_proxy`, `debt_per_action_frobenius_proxy`로 명명했다.

request endpoint 결합은 `(model,method,batch_index,request_sha256)` 1:1이며 `RS_current=1[target_new_nll<target_true_nll]`; tie는 failure이다. `nll_advantage=target_true_nll-target_new_nll`이다. PS/NS는 만들지 않았다.

## 3. 정확한 분모와 hard gate

| gate | value |
| --- | --- |
| absolute_allocation_energy_weighted_debt | NOT_COMPUTABLE_FROM_SCHEMA |
| action_key_duplicate_rows | 0 |
| action_rows | 2000 |
| arm_count | 4 |
| batch_action_request_replication_count | 0 |
| batch_count_per_arm | 100 |
| core_nonfinite_count | 0 |
| debt_decomposition_identity_failure_count | 0 |
| debt_decomposition_max_abs_error | 0.000000 |
| debt_decomposition_tolerance_rule | 8*eps_fp64*max(1,abs(identity_lhs)) row-wise for each identity |
| debt_native_identity_failure_count | 0 |
| debt_native_identity_max_abs_residual | 0.000000 |
| debt_parallel_native_identity_failure_count | 0 |
| debt_parallel_native_identity_max_abs_residual | 0.000000 |
| exact_g_metric_action | NOT_COMPUTABLE_FROM_SCHEMA |
| imputation_count | 0 |
| input_sha_before_after_unchanged | True |
| interpolation_count | 0 |
| layer_key_duplicate_rows | 0 |
| layer_rows | 200000 |
| layer_to_action_join_missing | 0 |
| layer_to_endpoint_join_missing | 0 |
| layers_per_request | 5 |
| plot_byte_reproduction_failure_count | 0 |
| plot_count | 5 |
| reconstruction_count | 0 |
| request_key_duplicate_rows | 0 |
| request_rows | 40000 |
| requests_per_arm_batch | 100 |
| scientific_promotion | False |

모든 composite key는 unique이고 layer→endpoint, layer→action join 누락은 0이다. core/derived nonfinite, debt decomposition failure, interpolation, imputation은 모두 0이다.

## 4. Layer별 tail

| model | method | layer | median | p90 | p99 | CVaR90 | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama3-8b-inst | alphaedit | 4 | 9.615305 | 36.252570 | 111.372511 | 8628963.281931 | 1276941999.348429 |
| llama3-8b-inst | alphaedit | 5 | 2.229812 | 9.039129 | 20.893036 | 14.460485 | 82.231871 |
| llama3-8b-inst | alphaedit | 6 | 0.698351 | 1.546782 | 7.295189 | 3.610906 | 17.569346 |
| llama3-8b-inst | alphaedit | 7 | 0.324105 | 0.525663 | 0.902272 | 0.682491 | 2.704991 |
| llama3-8b-inst | alphaedit | 8 | 0.030505 | 0.261235 | 0.445327 | 0.351011 | 1.406072 |
| llama3-8b-inst | memit | 4 | 10.426256 | 22.041521 | 61.545770 | 3925351.983284 | 1992339601.207516 |
| llama3-8b-inst | memit | 5 | 4.241338 | 13.750359 | 34.264512 | 24.322787 | 476.710871 |
| llama3-8b-inst | memit | 6 | 1.187523 | 2.928080 | 7.794263 | 4.857841 | 31.298444 |
| llama3-8b-inst | memit | 7 | 0.541540 | 0.795714 | 1.646089 | 1.124633 | 7.159720 |
| llama3-8b-inst | memit | 8 | 0.421061 | 0.567143 | 0.837645 | 0.667194 | 2.065591 |
| qwen2.5-7b-inst | alphaedit | 4 | 7.350136 | 22.396823 | 55.589854 | 1261620.968964 | 1261582300.262973 |
| qwen2.5-7b-inst | alphaedit | 5 | 1.412991 | 6.093266 | 24.260204 | 14.675642 | 348.933903 |
| qwen2.5-7b-inst | alphaedit | 6 | 0.573966 | 1.612494 | 10.870618 | 4.707881 | 65.037370 |
| qwen2.5-7b-inst | alphaedit | 7 | 0.273980 | 0.686664 | 2.484194 | 1.349671 | 9.822798 |
| qwen2.5-7b-inst | alphaedit | 8 | 0.003508 | 0.050917 | 0.523433 | 0.223149 | 2.321788 |
| qwen2.5-7b-inst | memit | 4 | 21.938675 | 75.445023 | 236.617582 | 148.036967 | 1524.742254 |
| qwen2.5-7b-inst | memit | 5 | 3.766028 | 12.455608 | 31.700183 | 20.832492 | 160.248735 |
| qwen2.5-7b-inst | memit | 6 | 0.546297 | 2.243466 | 8.570730 | 4.965841 | 79.303639 |
| qwen2.5-7b-inst | memit | 7 | 0.250418 | 0.542855 | 1.118771 | 0.809740 | 4.775360 |
| qwen2.5-7b-inst | memit | 8 | 0.342114 | 0.689620 | 1.082571 | 0.867524 | 1.972690 |

전체 batch×layer 세부값은 `batch-layer-debt-summary.csv`, action 결합은 `batch-layer-action-proxy.csv`에 있다.

## 5. 구성요소와 극단 tail provenance

| model | method | debt>1e6 rows | negative ΔV rows | under mean | over mean | opposite mean | orthogonal mean | max batch | max layer | max debt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3-8b-inst | alphaedit | 16 | 3101 | 0.077800 | 950.747343 | 1369.681895 | 170261.791015 | 14 | 4 | 1276941999.348429 |
| llama3-8b-inst | memit | 2 | 2480 | 0.224533 | 739.922091 | 757.995838 | 77012.478570 | 30 | 4 | 1992339601.207516 |
| qwen2.5-7b-inst | alphaedit | 1 | 1732 | 0.016344 | 24.002128 | 0.004718 | 25210.698337 | 27 | 4 | 1261582300.262973 |
| qwen2.5-7b-inst | memit | 0 | 5310 | 0.109021 | 1.489213 | 0.003225 | 6.988900 | 36 | 4 | 1524.742254 |

`debt>1e6`은 결과 선택이나 pass/fail에 쓰지 않은 고정 observation bin이다. 상위 5개 request-layer 행의 request SHA, rho/tau, 네 debt 구성요소는 `debt-tail-hotspots.csv`에 보존했다. 세 arm의 매우 큰 전체 mean은 소수 L4 행과 큰 orthogonal 성분에 지배되며, qwen/MEMIT은 million-scale 행이 없지만 p90과 p99가 더 넓은 별도 tail 형태를 보인다.

`normalized_potential_reduction<0`은 같은 write에서 q_post²가 q_pre²보다 커진 관측 행 수다. 이를 미래 forgetting이나 인과 효과로 해석하지 않는다.

## 6. Immediate rewrite와의 동시점 연관

| model | method | outcome | Pearson r | Spearman rho |
| --- | --- | --- | --- | --- |
| llama3-8b-inst | alphaedit | rs_current_rate | 0.054569 | -0.145202 |
| llama3-8b-inst | alphaedit | nll_advantage_mean | 0.011901 | -0.344938 |
| llama3-8b-inst | alphaedit | target_new_strict_rate | 0.246628 | -0.396644 |
| llama3-8b-inst | memit | rs_current_rate | -0.059319 | 0.060949 |
| llama3-8b-inst | memit | nll_advantage_mean | -0.051094 | -0.008113 |
| llama3-8b-inst | memit | target_new_strict_rate | -0.010678 | -0.092556 |
| qwen2.5-7b-inst | alphaedit | rs_current_rate | 0.063564 | -0.650474 |
| qwen2.5-7b-inst | alphaedit | nll_advantage_mean | 0.070854 | -0.866631 |
| qwen2.5-7b-inst | alphaedit | target_new_strict_rate | 0.090373 | -0.868372 |
| qwen2.5-7b-inst | memit | rs_current_rate | -0.817735 | -0.688492 |
| qwen2.5-7b-inst | memit | nll_advantage_mean | -0.854829 | -0.706535 |
| qwen2.5-7b-inst | memit | target_new_strict_rate | -0.838022 | -0.768354 |

상관계수의 관측 단위는 arm별 100개 batch이다. 같은 batch의 debt와 바로 뒤 current-B100 endpoint를 연결했을 뿐, 미래 forgetting이나 causal mediation으로 해석하지 않는다.

## 7. Figure

- `median-rho-batch-layer-heatmap.png`: arm별 batch1–100×L4–L8 median rho
- `median-tau-batch-layer-heatmap.png`: arm별 batch1–100×L4–L8 median tau
- `median-debt-native-batch-layer-heatmap.png`: arm별 row-wise debt_native median
- `debt-frobenius-action-joint-trajectory.png`: batch×layer debt/action joint trajectory
- `debt-immediate-rewrite-relationship.png`: batch debt와 RS/NLL advantage/strict rate

모든 PNG는 저장소 Python CLI로 동일 CSV를 두 번 실제 실행해 byte SHA가 같은지 검증했다. 정확한 명령·입력/출력 SHA·Python/NumPy/pandas/Matplotlib/Pillow 버전은 `plot-reproduction.json`에 있다.

## 8. 입력 identity

| kind | path | bytes | sha256 |
| --- | --- | --- | --- |
| primary | experiment-reports/servers/server4/official-layer-realization-debt-lifelong-b100x100-2026-09-03-v3/production-layer-complete.csv.gz | 23850101 | 2fa6450eafa2cc9a095b25f1f8009bb2da16084a5eeefdd0287a7b568814359e |
| primary | experiment-reports/servers/server4/official-layer-realization-debt-lifelong-b100x100-2026-09-03-v3/production-weight-batch-unit-complete.csv.gz | 198735 | d3333403c9fd5249d2163e51d3221e356c33e5c3617c1150de729e7294d967a8 |
| primary | experiment-reports/servers/server4/official-layer-realization-debt-lifelong-b100x100-2026-09-03-v3/production-request-complete.csv.gz | 11065700 | 4e0d6020e97f8c45f6ad246224a2f581097991fc6ca8d08c621c460e1492dff5 |
| context_v3 | experiment-reports/servers/server4/official-layer-realization-debt-lifelong-b100x100-2026-09-03-v3/analysis-manifest.json | 46003 | 47398d3e497edbbf8f2ace5be4f315134653f16ee2698d47945d4769fe0deb15 |
| context_v3 | experiment-reports/servers/server4/official-layer-realization-debt-lifelong-b100x100-2026-09-03-v3/rooted-analysis-receipt.json | 2145 | 9882b8bca524f2a34d2a3c1c4fea803aea834552c675a19be2b1e69ab7224af9 |
| context_v3 | experiment-reports/servers/server4/official-layer-realization-debt-lifelong-b100x100-2026-09-03-v3/official-layer-realization-debt-lifelong-fourarm-exhaustive-factual-ko.md | 69433 | c9daf397b8c9fa679fa89e4f6407d7e03761a11f9ead177e998a97e726e24a8f |
| context_v4-r2 | experiment-reports/servers/server4/official-layer-realization-debt-lifelong-b100x100-2026-09-03-v4-r2/analysis-manifest.json | 8318 | 848dd78ea44d26105eb2f75344aa252413d249fddf2b03dd104e3a78f9cea0a9 |
| context_v4-r2 | experiment-reports/servers/server4/official-layer-realization-debt-lifelong-b100x100-2026-09-03-v4-r2/rooted-analysis-receipt.json | 1344 | bffff6260cae5f4679f0c6ac7b3b5020ac2c286955caebb7e1a4683454561b04 |
| context_v4-r2 | experiment-reports/servers/server4/official-layer-realization-debt-lifelong-b100x100-2026-09-03-v4-r2/official-layer-debt-lifelong-finalw-full10k-factual-ko.md | 79795 | 225fbe8cc7fea44a2e9e00b01a1e3648686e439cd6618a150165b3f33cc95c68 |
| context_v5 | experiment-reports/servers/server4/official-layer-realization-debt-lifelong-b100x100-2026-09-03-v5/analysis-manifest.json | 12364 | 5599b8a351e541f214d9aae8a2b645009edb0989001dfa37b1c36e2007ac7070 |
| context_v5 | experiment-reports/servers/server4/official-layer-realization-debt-lifelong-b100x100-2026-09-03-v5/rooted-analysis-receipt.json | 1005 | a5b7355ccbf819095cf4aa8c4a199f11ab394b41a9bc9793a1c2d8a68b44682d |
| context_v5 | experiment-reports/servers/server4/official-layer-realization-debt-lifelong-b100x100-2026-09-03-v5/official-layer-realization-debt-lifelong-v5-exhaustive-cumulative-factual-ko.md | 264804 | 7efe0fb9df78cd00c9abc3abce81db411ce099a493cd1812582ff4cb89084de3 |
| context_v6 | experiment-reports/servers/server4/official-layer-realization-debt-lifelong-b100x100-2026-09-03-v6/analysis-manifest.json | 15407 | 4a9374bd9874a4ddaefae81fcec56deaf4b175ad3b553b46d3d6a1e25f21107f |
| context_v6 | experiment-reports/servers/server4/official-layer-realization-debt-lifelong-b100x100-2026-09-03-v6/rooted-analysis-receipt.json | 980 | fad7c32463583d6c5bb1a648ebc0265440fef1466483ddac13a72eacdc5e2565 |
| context_v6 | experiment-reports/servers/server4/official-layer-realization-debt-lifelong-b100x100-2026-09-03-v6/official-layer-debt-lifelong-counterfact-metrics-v6-factual-ko.md | 312488 | 60b70b81747068f84ae7904376995882695a12b4dd90214b8902ec601ca5eb23 |

세 primary gzip SHA는 산출 전후 재검증해 동일했다. v3–v6 manifest/receipt/report는 context binding에만 사용했고 원본 bytes를 수정하지 않았다.

## 9. 출력 identity (report/manifest/receipt 이전 산출물)

| path | bytes | sha256 |
| --- | --- | --- |
| analysis-audit.json | 11796 | bb501a81730fb6463a4e43d9673c0c72ae6383a2ff69c98357809050430e124e |
| arm-debt-summary.csv | 8310 | 192e1035b532b56e7e3aa058a38906e9a2d7b6611178aa49bf7ff35cb0359857 |
| arm-layer-tail-summary.csv | 21125 | c8f15ad69b7ba5d8ef7f8e2dc0639478f0d99b7828c363a6fb67d8ee32c2fbc4 |
| arm-tail-diagnostics.csv | 1393 | 444478ca2082c61e9fa37ac0f4ac3d17747c4a5ed7bd8ed9b0d704e8e72e2548 |
| association-summary.csv | 1918 | e20416336c2e4c85c4476cc077481a4625e9fdebfa1a0c9e4edc7a78875f3813 |
| batch-endpoint-relationship.csv | 120822 | 86518e41989aaa5053dc755433ea5a0afc69b500407d9e06186b2cf658eed076 |
| batch-layer-action-proxy.csv | 2019054 | 20077c3d632373948437c052266d392052079435ae6564fc22ea7be863c1564f |
| batch-layer-debt-summary.csv | 1891766 | bd77f87140248947229d4ecbb1c5500c7465525e00a189c6632e19e147b8c589 |
| debt-frobenius-action-joint-trajectory.png | 140516 | 4e36a7e8d1fe7af2c9b20c3c18e99a3bce762ee463ab8871ea652353085f9933 |
| debt-immediate-rewrite-relationship.png | 246571 | 984b24aeab81cb19848fc73fbb04a373093fce25b8beaabe67d41d15d438c68e |
| debt-tail-hotspots.csv | 6646 | 35f556cae2eabd4582c633cec087cc2e819041dc0966a76d3a270eff8590f926 |
| median-debt-native-batch-layer-heatmap.png | 47154 | be032415e2f5ebd5cc2a1888cb93ce837ba2b177e4aed6aa30dd118473ab8149 |
| median-rho-batch-layer-heatmap.png | 49009 | e13202cae5f786d7177e35f6e9efe3e28a441b99b9dacb62d5dfe87f30c316a8 |
| median-tau-batch-layer-heatmap.png | 46131 | 4e25c2e3e3821e96dc1917750a8ba766abf57ea297164fd3efed5e5946e1897e |
| output-member-inventory.csv | 2209 | 7496196d548c156620495c32054897db0b138637d18978197dad9cae9ee9d677 |
| paired-method-batch-deltas.csv | 30101 | e8fb892acf363908fd01c55dc9436f8624520da8ae44049bf4290d1bd35340f7 |
| plot-reproduction.json | 3663 | c15ee5d3c9597002846defbc6269557b6f57df8b95d7e0f6a37f7d30bc1f889c |
| request-debt-endpoint.csv.gz | 18561072 | cd18d2339aaa788132f188d496bab07f217204f81dacfc31af3e576783a2ed38 |
| request-layer-debt.csv.gz | 19855352 | aa737302dcada6632ac10407edba4c67280a0f867168132a1cf099be6b7d10cc |

최종 전체 member root와 report/manifest/receipt SHA는 `analysis-manifest.json` 및 `rooted-analysis-receipt.json`이 결속한다.

## 10. 한계

- raw schema에는 `R_entry_norm_sq`, `A_norm_sq`, exact allocation-energy weight, `DeltaW^T G DeltaW`가 없다.
- Frobenius proxy를 exact G-action 또는 request-level action으로 해석하지 않는다.
- batch action을 100 request에 복제하지 않았다.
- current-B100 endpoint만 사용했으며 canonical PS/NS 또는 미래 checkpoint 결과를 재구성하지 않았다.
- interpolation/imputation/reconstruction은 모두 0이다.
