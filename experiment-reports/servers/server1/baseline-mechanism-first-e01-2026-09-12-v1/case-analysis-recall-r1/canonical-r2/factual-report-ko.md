# E1-A singleton per-case 연결: CPU 사실 보고

기존 AlphaEdit/MEMIT 각 L4–L8의 100개 current 및 12개 seen-full 파일을 재사용했다. 새 편집·모델 평가·토크나이저 실행은 없다.

## 정의·분모

각 arm의 최종 RS 10,000, PS 20,000, NS 100,000 prompt-pair를 같은 at-write prompt/target SHA와 연결했다. 10개 arm은 동일 10,000 요청을 반복 관측한 것이며 독립 100,000 요청이 아니다. RS/PS는 true−new NLL, NS는 new−true NLL > 0을 성공으로 한다. Tie는 실패이고 원 raw margin(true−new)은 별도 보존했다.

전체 원분모가 주표다. common-at-write-success는 동일 family 5개 layer 모두 at-write 성공한 정확히 같은 prompt 부분집합이다. Common-support는 관계 × at-write margin × age × new target token count의 공동 점유 bin에 해당하는 각 arm 행이며, 동일 사례 매칭/가중 표준화와 다르다. 빈 bin과 제외율을 CSV에 남겼다. True token count도 local 원장에 보존했다. Bin은 분석 코드에 고정하고 raw outcome 파일 읽기 전 lock을 만들었지만 과거 공개 결과가 있는 탐색 분석이며 사전 실험 gate가 아니다.

## 최종 전체 원분모 연결

| Family | Layer | Metric | At-write n/d | Final n/d | Lost / entry-success | Recovery |
|---|---:|---|---:|---:|---:|---:|
| AlphaEdit | 4 | RS | 9993/10000 | 9939/10000 | 55/9993 | 1 |
| AlphaEdit | 5 | RS | 9989/10000 | 9934/10000 | 57/9989 | 2 |
| AlphaEdit | 6 | RS | 9987/10000 | 9683/10000 | 307/9987 | 3 |
| AlphaEdit | 7 | RS | 9984/10000 | 9240/10000 | 746/9984 | 2 |
| AlphaEdit | 8 | RS | 9988/10000 | 9396/10000 | 595/9988 | 3 |
| AlphaEdit | 4 | PS | 19403/20000 | 19136/20000 | 423/19403 | 156 |
| AlphaEdit | 5 | PS | 19303/20000 | 18839/20000 | 630/19303 | 166 |
| AlphaEdit | 6 | PS | 18957/20000 | 17148/20000 | 2076/18957 | 267 |
| AlphaEdit | 7 | PS | 18525/20000 | 16194/20000 | 2726/18525 | 395 |
| AlphaEdit | 8 | PS | 17838/20000 | 15556/20000 | 2909/17838 | 627 |
| AlphaEdit | 4 | NS | 72505/100000 | 65348/100000 | 11547/72505 | 4390 |
| AlphaEdit | 5 | NS | 69452/100000 | 62822/100000 | 11922/69452 | 5292 |
| AlphaEdit | 6 | NS | 65726/100000 | 58396/100000 | 14069/65726 | 6739 |
| AlphaEdit | 7 | NS | 63523/100000 | 56277/100000 | 15869/63523 | 8623 |
| AlphaEdit | 8 | NS | 62104/100000 | 54703/100000 | 15764/62104 | 8363 |
| MEMIT | 4 | RS | 9934/10000 | 7722/10000 | 2233/9934 | 21 |
| MEMIT | 5 | RS | 9923/10000 | 8084/10000 | 1863/9923 | 24 |
| MEMIT | 6 | RS | 9836/10000 | 8277/10000 | 1612/9836 | 53 |
| MEMIT | 7 | RS | 9842/10000 | 8444/10000 | 1451/9842 | 53 |
| MEMIT | 8 | RS | 9676/10000 | 8177/10000 | 1642/9676 | 143 |
| MEMIT | 4 | PS | 18959/20000 | 14966/20000 | 4294/18959 | 301 |
| MEMIT | 5 | PS | 18724/20000 | 15008/20000 | 4148/18724 | 432 |
| MEMIT | 6 | PS | 17880/20000 | 14761/20000 | 3981/17880 | 862 |
| MEMIT | 7 | PS | 17840/20000 | 14941/20000 | 3772/17840 | 873 |
| MEMIT | 8 | PS | 17427/20000 | 14460/20000 | 3941/17427 | 974 |
| MEMIT | 4 | NS | 69438/100000 | 57848/100000 | 19823/69438 | 8233 |
| MEMIT | 5 | NS | 64858/100000 | 54285/100000 | 19220/64858 | 8647 |
| MEMIT | 6 | NS | 60468/100000 | 50488/100000 | 20004/60468 | 10024 |
| MEMIT | 7 | NS | 58058/100000 | 48547/100000 | 19536/58058 | 10025 |
| MEMIT | 8 | NS | 56990/100000 | 48425/100000 | 19642/56990 | 11077 |

## 범위·관측 한계

Active/superseded는 NFC·공백 정규화 subject + relation의 관측 prefix 내 latest ordinal로 구분한다. 뒤의 target이 같은 재발행과 다른 target overwrite 여부를 별도로 기록했다. Superseded target 실패는 자동으로 실제 active-history loss로 해석하지 않는다. 원분모를 제거하지 않았다.

실패·회복은 실제 at-write와 B001/B005/B010/B020…B100 seen-full 관측만 사용한다. 저장 간격 안의 최초 실패 시점 또는 숨은 실패·회복은 추정하지 않았다. At-write margin은 layer 선택 이후 변수이므로 matched gap을 인과 기여율로 해석할 수 없다. 동일 batch의 서로 다른 layer는 서로 다른 trajectory/state이므로 본 표는 관찰적 산술 비교다.

BLUE L4+L8/원본5-layer/W0는 본 selective raw 입력에 포함되지 않은 별도 reference다. 기존 global/blue-native-lifelong-independent-audit-2026-09-11-v1/independent-review-ko.md의 봉인 집계를 context로만 연결하며, 이 singleton per-case 표에 복제하거나 source/config 동등성을 주장하지 않는다. 새 GPU E0 복원 equivalence, E1-B target/write/derivative와 general 항목의 완료 여부는 별도 owner package에서 판정한다.

## 산출물·재현

`margin_age_matched_retention.csv`: 전체/공통성공/공통지지/active strata, NLL와 margin 요약. `matching-strata-with-empty-bins.csv`: 모든 점유 union bin과 layer별 빈 bin. `checkpoint-paired-retention.csv`: 12개 저장 시점별 같은 at-write row 연결. `layer-paired-differences.csv`: L4/L5 대비 더 깊은 layer의 exact-paired 산술 차이. `observed-failure-intervals.csv`: 관측된 실패 구간 상태. 대형 per-prompt 원장은 local-only이며 SHA/경로는 `paired_case_ledger_manifest.json`에 있다.

실행 명령은 아래 CLI이며 출력/로컬 디렉터리는 create-once다. PNG나 새 GPU 평가를 이 부분 보고의 선행조건으로 추가하지 않았다.

```bash
python3 -m project.run_scripts.baseline_mechanism_first.case_population --plan local/baseline-mechanism-first-e01/20260912-v1/imports/server4-allowlist-v1.json --dataset /mnt/raid5/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json --output <new-output-directory> --private <new-local-directory>
```

scientific_promotion=false. SH1은 사실·수치 연결을 제공하며 최종 원인 종합은 GH 소유다.
