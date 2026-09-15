## 9. 이번 recall의 설계-실행 상세 보충

이번 review-only recall에서 원문·pause를 다시 결속했다. 2026-09-15 사용자 pause를 보존했고 이번에는 정확 네 job의 scheduler만 한 번 확인했다. CAKE도 별도 완료 리뷰했지만 10k 최종값과 아래1k를 직접 합치지 않는다.

실행 lock의 `numerical_rationale.alpha_cap=1`은 CAP1에서 복사된 **구형 설명 metadata**다. 실제 runner가 읽은 `numerical_policy.alpha_cap_mode/alpha_cap` 및 저장 correction receipt는 각각10/100/disabled-null로 일치한다. 이 metadata 차이를 숨기거나 lock을 수정하지 않았다. `after_gate`의 과거 RUN_THROUGH 문자열도 이후 user pause/현재 review-only 권한을 대체하지 않는다.

| Arm | cap-bound/norm-bound | halfspace active | ball hit | trust retractions | post-ball <gE,C> >0 | CPU W diff max |
| --- | --- | --- | --- | --- | --- | --- |
| CAP10 | 10/0 | 9 | 112 | 0 | 7 | 0 |
| CAP100 | 10/0 | 9 | 156 | 0 | 7 | 0 |
| NORM_ONLY | 0/10 | 9 | 980 | 0 | 9 | 0 |

30개 batch의 q/gradient norm/cos/coefficient·KKT·direction/C inner product·pre/post ball·actual selected geometry는 mechanism-observed-details.csv에 전부 있다. 반공간의 1차 조건은 ball/FP32 materialization 이후 개별 요청/PS/old 보존 보장이 아니다. 양의 post-ball 내적도 제외하지 않았다. CPU 재구성 일치는 해당 저장 selected W4 범위이며 GPU derivative/전체 모델 replay가 아니다.

### 실제 동작 사례 — 모든 batch 자료는 별도 CSV로 유지

#### CAP10 B001

자기 batch-entry→native Vp의 action norm=7.6101871. q=-2.1026344e-07, alpha_norm=2904.1925, used alpha=10. Ball hit=0, trust scale=1.0. 최대 C1/native=0.0860825%, 실제 selected=C05 / 0.0430412%다.

| 후보 | E−RAW | D−RAW | strict lost | feasible | 선택 | 사유 |
| --- | --- | --- | --- | --- | --- | --- |
| RAW | 0.0 | 0.0 | 0 | True | False | FINITE_EXACT_QUALITY_FEASIBLE |
| C1 | 1.6535341273299364e-09 | -3.945859043597011e-05 | 0 | False | False | E_EXCEEDS_OWN_RAW_NO_ALLOWANCE |
| C05 | -9.478389983910013e-09 | -2.000844040139782e-05 | 0 | True | True | FINITE_EXACT_QUALITY_FEASIBLE |
| C025 | 8.541101124102946e-09 | -1.0081025152430811e-05 | 0 | False | False | E_EXCEEDS_OWN_RAW_NO_ALLOWANCE |

선택 후 E 0.00228766615→0.00228765667, D 0.00157817869→0.00155817025. 선택된 실제 weight에서 finalizer1회, history_count=1, next_ordinal=100. 다음 batch가 있으면 저장 W/M/RNG/ledger 일치가 연결된다. RAW도 native write를 commit한다.

#### CAP10 B002

자기 batch-entry→native Vp의 action norm=7.7166246. q=-8.4953136e-06, alpha_norm=1868.6117, used alpha=10. Ball hit=7, trust scale=1.0. 최대 C1/native=0.133789%, 실제 selected=RAW / 0%다.

| 후보 | E−RAW | D−RAW | strict lost | feasible | 선택 | 사유 |
| --- | --- | --- | --- | --- | --- | --- |
| RAW | 0.0 | 0.0 | 0 | True | True | FINITE_EXACT_QUALITY_FEASIBLE |
| C1 | 7.077650661813789e-07 | -9.216162590064414e-05 | 0 | False | False | E_EXCEEDS_OWN_RAW_NO_ALLOWANCE |
| C05 | 2.202715404563474e-07 | -4.6619110975143485e-05 | 0 | False | False | E_EXCEEDS_OWN_RAW_NO_ALLOWANCE |
| C025 | 1.6247904568479632e-07 | -2.3447515104635386e-05 | 0 | False | False | E_EXCEEDS_OWN_RAW_NO_ALLOWANCE |

선택 후 E 0.00647821125→0.00647821125, D 0.00359200933→0.00359200933. 선택된 실제 weight에서 finalizer1회, history_count=2, next_ordinal=200. 다음 batch가 있으면 저장 W/M/RNG/ledger 일치가 연결된다. RAW도 native write를 commit한다.

#### CAP100 B001

자기 batch-entry→native Vp의 action norm=7.6101871. q=-2.1026344e-07, alpha_norm=2904.1925, used alpha=100. Ball hit=0, trust scale=1.0. 최대 C1/native=0.860824%, 실제 selected=RAW / 0%다.

| 후보 | E−RAW | D−RAW | strict lost | feasible | 선택 | 사유 |
| --- | --- | --- | --- | --- | --- | --- |
| RAW | 0.0 | 0.0 | 0 | True | True | FINITE_EXACT_QUALITY_FEASIBLE |
| C1 | 2.3970894108059186e-06 | -0.0002940843290275552 | 0 | False | False | E_EXCEEDS_OWN_RAW_NO_ALLOWANCE |
| C05 | 6.013517122481146e-07 | -0.00017481506762351273 | 0 | False | False | E_EXCEEDS_OWN_RAW_NO_ALLOWANCE |
| C025 | 1.3301854778552044e-07 | -9.444457793961192e-05 | 0 | False | False | E_EXCEEDS_OWN_RAW_NO_ALLOWANCE |

선택 후 E 0.00228766615→0.00228766615, D 0.00157817869→0.00157817869. 선택된 실제 weight에서 finalizer1회, history_count=1, next_ordinal=100. 다음 batch가 있으면 저장 W/M/RNG/ledger 일치가 연결된다. RAW도 native write를 commit한다.

#### CAP100 B006

자기 batch-entry→native Vp의 action norm=8.2248681. q=3.2415373e-06, alpha_norm=1163.2412, used alpha=100. Ball hit=23, trust scale=1.0. 최대 C1/native=2.14892%, 실제 selected=C1 / 2.14892%다.

| 후보 | E−RAW | D−RAW | strict lost | feasible | 선택 | 사유 |
| --- | --- | --- | --- | --- | --- | --- |
| RAW | 0.0 | 0.0 | 0 | True | False | FINITE_EXACT_QUALITY_FEASIBLE |
| C1 | -0.00029184903985879015 | -0.0017483460464973177 | 0 | True | True | FINITE_EXACT_QUALITY_FEASIBLE |
| C05 | -0.00015369911459856694 | -0.0010451435015284005 | 0 | True | False | FINITE_EXACT_QUALITY_FEASIBLE |
| C025 | -7.860029763833223e-05 | -0.0005809785318433569 | 0 | True | False | FINITE_EXACT_QUALITY_FEASIBLE |

선택 후 E 0.0108463245→0.0105544755, D 0.0119846758→0.0102363297. 선택된 실제 weight에서 finalizer1회, history_count=6, next_ordinal=600. 다음 batch가 있으면 저장 W/M/RNG/ledger 일치가 연결된다. RAW도 native write를 commit한다.

#### NORM_ONLY B001

자기 batch-entry→native Vp의 action norm=7.6101871. q=-2.1026344e-07, alpha_norm=2904.1925, used alpha=2904.1925. Ball hit=87, trust scale=1.0. 최대 C1/native=24.6277%, 실제 selected=RAW / 0%다.

| 후보 | E−RAW | D−RAW | strict lost | feasible | 선택 | 사유 |
| --- | --- | --- | --- | --- | --- | --- |
| RAW | 0.0 | 0.0 | 0 | True | True | FINITE_EXACT_QUALITY_FEASIBLE |
| C1 | 0.005133288893994177 | 0.013120430364040203 | 0 | False | False | E_EXCEEDS_OWN_RAW_NO_ALLOWANCE |
| C05 | 0.0006519769231817917 | 0.004028800840387703 | 0 | False | False | E_EXCEEDS_OWN_RAW_NO_ALLOWANCE |
| C025 | 0.00019013664270460138 | 0.0007232002194541565 | 0 | False | False | E_EXCEEDS_OWN_RAW_NO_ALLOWANCE |

선택 후 E 0.00228766615→0.00228766615, D 0.00157817869→0.00157817869. 선택된 실제 weight에서 finalizer1회, history_count=1, next_ordinal=100. 다음 batch가 있으면 저장 W/M/RNG/ledger 일치가 연결된다. RAW도 native write를 commit한다.

#### NORM_ONLY B006

자기 batch-entry→native Vp의 action norm=8.2248681. q=3.2415373e-06, alpha_norm=1163.2412, used alpha=1163.2412. Ball hit=99, trust scale=1.0. 최대 C1/native=24.3467%, 실제 selected=C025 / 6.08666%다.

| 후보 | E−RAW | D−RAW | strict lost | feasible | 선택 | 사유 |
| --- | --- | --- | --- | --- | --- | --- |
| RAW | 0.0 | 0.0 | 0 | True | False | FINITE_EXACT_QUALITY_FEASIBLE |
| C1 | 0.03958804184898326 | -0.00046222184329280935 | 1 | False | False | E_EXCEEDS_OWN_RAW_NO_ALLOWANCE\|RAW_STRICT_IDS_LOST |
| C05 | 0.017125567936964216 | -0.003915852488717064 | 1 | False | False | E_EXCEEDS_OWN_RAW_NO_ALLOWANCE\|RAW_STRICT_IDS_LOST |
| C025 | -2.22026566916618e-06 | -0.0034329745079730856 | 0 | True | True | FINITE_EXACT_QUALITY_FEASIBLE |

선택 후 E 0.0108463245→0.0108441042, D 0.0119846758→0.00855170125. 선택된 실제 weight에서 finalizer1회, history_count=6, next_ordinal=600. 다음 batch가 있으면 저장 W/M/RNG/ledger 일치가 연결된다. RAW도 native write를 commit한다.

### Source-backed 매핑

| 요구 | 함수 source:line | 실물 근거 | 확인수준 |
| --- | --- | --- | --- |
| actual RAW Vp; own native fit1 | runner.py:194 | 30 native-targets-map; commit native100/solve1 | SOURCE_AND_CPU_TENSOR_BINDING |
| C0 copies Vp; gC=gW A^T | model_adapter.py:166 | gE/gD/RAW header bridge; derivative FD skipped | SOURCE_CONFIRMED_NUMERICAL_NOT_ESTABLISHED |
| E desired token mean then request mean | model_adapter.py:388 | current rows denominator100;7 microbatches | SOURCE_AND_STORED_ROWS |
| D W0 full-vocab KL S64 | model_adapter.py:432 | 64doc128positions; S64 teacher ID | SOURCE_AND_STORED_ROWS |
| E/D separate sweep, only S64 gradient | model_adapter.py:411 | 70current+640S64 backward/arm | SOURCE_AND_COUNTERS |
| halfspace d and zero branch | policy.py:159 | per-batch-actions CPU q/dot vs stored | SOURCE_AND_STORED_TENSORS |
| bounded/disabled cap only | policy.py:263 | alpha norm/used/mode/cap receipt | SOURCE_AND_CPU_SCALARS |
| ball then C-only trust | policy.py:277 | ball hit/trust/inner products | SOURCE_AND_CPU_SCALARS |
| RAW/C1/C05/C025 actual materialization | policy.py:349 | 160 candidate receipts incl reused10batch | SOURCE_AND_CPU_TENSORS |
| exact E and strict ID set; minD tieRAW | policy.py:424 | candidate-details arithmetic; no tolerance relaxation | SOURCE_AND_STORED_ROWS |
| final history1 and accepted-only ledger | runner.py:278 | 30CP/27links/ledger summaries | SOURCE_AND_CPU_TENSORS |
| FD/ULP/selfKL/direct diagnostic skip | runner.py:219 | technical JSON SKIPPED_USER_DIRECTED | SKIPPED_USER_DIRECTED |

원 수치 검증은 SKIPPED_USER_DIRECTED/NOT_ESTABLISHED. 이번 source/CPU 산술·actual tensor 확인은 FD/direct/selfKL 재검증이 아니다. Strict lost ID의 숫자/문자열 정렬 차이는 reviewer만 exact set으로 보완했고, runtime의 [12179,18415,9763]은 동일3개 ID였다. 과학적 screen 오류로 오인하지 않았다.

CAKE의 actual B10 first1000 참고값은 CAKE-B10-first1000.csv에만 별도 제공한다(RS989/1000, PS1703/2000, NS8118/10000). Layer/L2/decay/clamp/seed가 달라 단일요인 효과로 해석하지 않는다. CAKE W100/full10k를 이 표에 넣지 않는다.
