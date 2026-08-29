# Cache-aware one-sided q-KL projected AlphaEdit — atomic B100 factual report

상태: **STOPPED_AFTER_ATOMIC_B100_USER_DIRECTED**
범위: Llama/Qwen atomic B100, PRE-EDIT W0, stock Official AlphaEdit N1, ours projected N2/N4.
Sequential scientific execution은 사용자 지시로 제외되었고 promotion=false이다.

## 핵심 비교: PRE-EDIT / BASELINE / OURS

NLL은 token-mean이며, Gen strict는 request의 모든 rephrase prompt가 exact target-new인 비율이다.
PRE-EDIT locality=1은 자기 자신과의 reference identity이며 편집 성능이 아니다.

| Model | Group | Method | Rewrite new NLL mean/med/p90 | Rewrite exact | Rephrase new NLL mean/med/p90 | Gen prompt / strict | Loc | terminal qKL | removed energy | active nodes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| llama | PRE_EDIT | PRE_EDIT_W0 | 10.465489/10.851508/14.949640 | 0.020000 | 9.854293/10.040214/14.373120 | 0.010000 / 0.000000 | 1.000000 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED |
| llama | BASELINE | OFFICIAL_ALPHAEDIT_N1 | 0.001144/0.000551/0.002057 | 1.000000 | 1.833039/0.222728/6.822834 | 0.640000 / 0.480000 | 0.820000 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED |
| llama | OURS | QKL_PROJECTED_ODE_N2 | 0.226380/0.003837/0.049405 | 0.960000 | 2.244301/0.492313/7.471985 | 0.595000 / 0.440000 | 0.829000 | 2.016115 | 0.012208 | 1.000000 |
| llama | OURS | QKL_PROJECTED_ODE_N4 | 0.574507/0.005644/0.768567 | 0.900000 | 2.550082/0.715521/7.707615 | 0.555000 / 0.410000 | 0.837000 | 1.469940 | 0.018411 | 1.000000 |
| qwen | PRE_EDIT | PRE_EDIT_W0 | 10.704443/10.421624/15.007283 | 0.000000 | 10.239913/10.265379/14.426141 | 0.005000 / 0.000000 | 1.000000 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED |
| qwen | BASELINE | OFFICIAL_ALPHAEDIT_N1 | 0.031709/0.013291/0.063181 | 1.000000 | 1.853892/0.461679/5.584766 | 0.640000 / 0.470000 | 0.812000 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED |
| qwen | OURS | QKL_PROJECTED_ODE_N2 | 0.060920/0.018990/0.095849 | 1.000000 | 1.917562/0.481277/5.608226 | 0.625000 / 0.460000 | 0.813000 | 3.022079 | 0.002387 | 1.000000 |
| qwen | OURS | QKL_PROJECTED_ODE_N4 | 0.138594/0.033986/0.225417 | 0.980000 | 2.080734/0.586474/6.153248 | 0.595000 / 0.410000 | 0.821000 | 2.325931 | 0.006839 | 1.000000 |

## Factual interpretation

- N2는 두 모델에서 predictive q-KL을 낮추고 locality를 비열세 또는 소폭 개선했다.
- Llama N2는 rewrite exact 0.96을 유지했으나 rephrase prompt success가 Official 0.64에서 0.595로 낮아졌다.
- Qwen N2는 rewrite exact 1.00을 유지했고 rephrase prompt success는 0.64에서 0.625였다.
- N4는 두 모델 모두 q-KL/locality 쪽 이동이 더 크지만 rewrite/rephrase strength tradeoff도 더 컸다.
- 이 결과는 atomic B100 한 번의 factual endpoint이며 scientific promotion 또는 sequential claim이 아니다.

## Rewrite/Rephrase 상세

아래 값은 각 target-new/target-true NLL의 mean/median/p90/max와 exact/token accuracy이다.

### llama

- **PRE_EDIT_W0**
  - rewrite_target_new: n=100, NLL=10.465489/10.851508/14.949640/18.232416; exact=2/100 (0.020000); token=3/101 (0.029703)
  - rewrite_target_true: n=100, NLL=4.551351/4.391057/9.034202/13.833670; exact=24/100 (0.240000); token=24/100 (0.240000)
  - rephrase_target_new: n=200, NLL=9.854293/10.040214/14.373120/19.066179; exact=2/200 (0.010000); token=4/202 (0.019802)
  - rephrase_target_true: n=200, NLL=5.018609/4.507538/10.091694/14.675686; exact=43/200 (0.215000); token=43/200 (0.215000)
  - rewrite new<true=0.100000; rephrase new<true=0.130000; rephrase strict exact=0.000000; locality=1000/1000 (1.000000)
- **OFFICIAL_ALPHAEDIT_N1**
  - rewrite_target_new: n=100, NLL=0.001144/0.000551/0.002057/0.017939; exact=100/100 (1.000000); token=101/101 (1.000000)
  - rewrite_target_true: n=100, NLL=14.566310/14.378094/19.601617/26.657900; exact=0/100 (0.000000); token=0/100 (0.000000)
  - rephrase_target_new: n=200, NLL=1.833039/0.222728/6.822834/12.291740; exact=128/200 (0.640000); token=130/202 (0.643564)
  - rephrase_target_true: n=200, NLL=9.274225/9.314996/14.740460/22.619349; exact=4/200 (0.020000); token=4/200 (0.020000)
  - rewrite new<true=1.000000; rephrase new<true=0.950000; rephrase strict exact=0.480000; locality=820/1000 (0.820000)
- **QKL_PROJECTED_ODE_N2**
  - rewrite_target_new: n=100, NLL=0.226380/0.003837/0.049405/8.827570; exact=96/100 (0.960000); token=97/101 (0.960396)
  - rewrite_target_true: n=100, NLL=10.781811/10.681119/15.341920/21.691807; exact=1/100 (0.010000); token=1/100 (0.010000)
  - rephrase_target_new: n=200, NLL=2.244301/0.492313/7.471985/12.772212; exact=119/200 (0.595000); token=121/202 (0.599010)
  - rephrase_target_true: n=200, NLL=8.197197/8.389377/12.804981/18.953291; exact=4/200 (0.020000); token=4/200 (0.020000)
  - rewrite new<true=0.980000; rephrase new<true=0.915000; rephrase strict exact=0.440000; locality=829/1000 (0.829000)
- **QKL_PROJECTED_ODE_N4**
  - rewrite_target_new: n=100, NLL=0.574507/0.005644/0.768567/13.174025; exact=90/100 (0.900000); token=91/101 (0.900990)
  - rewrite_target_true: n=100, NLL=9.563771/9.236450/14.418640/21.059254; exact=2/100 (0.020000); token=2/100 (0.020000)
  - rephrase_target_new: n=200, NLL=2.550082/0.715521/7.707615/13.002110; exact=111/200 (0.555000); token=113/202 (0.559406)
  - rephrase_target_true: n=200, NLL=7.806610/7.986707/12.224680/18.609533; exact=7/200 (0.035000); token=7/200 (0.035000)
  - rewrite new<true=0.930000; rephrase new<true=0.890000; rephrase strict exact=0.410000; locality=837/1000 (0.837000)

### qwen

- **PRE_EDIT_W0**
  - rewrite_target_new: n=100, NLL=10.704443/10.421624/15.007283/18.132652; exact=0/100 (0.000000); token=1/101 (0.009901)
  - rewrite_target_true: n=100, NLL=5.463334/4.787822/10.775709/18.370783; exact=19/100 (0.190000); token=19/100 (0.190000)
  - rephrase_target_new: n=200, NLL=10.239913/10.265379/14.426141/18.661018; exact=1/200 (0.005000); token=3/202 (0.014851)
  - rephrase_target_true: n=200, NLL=5.829665/5.491144/11.202452/17.853935; exact=31/200 (0.155000); token=31/200 (0.155000)
  - rewrite new<true=0.140000; rephrase new<true=0.150000; rephrase strict exact=0.000000; locality=1000/1000 (1.000000)
- **OFFICIAL_ALPHAEDIT_N1**
  - rewrite_target_new: n=100, NLL=0.031709/0.013291/0.063181/0.335821; exact=100/100 (1.000000); token=101/101 (1.000000)
  - rewrite_target_true: n=100, NLL=13.910183/14.172611/18.826527/22.704771; exact=0/100 (0.000000); token=0/100 (0.000000)
  - rephrase_target_new: n=200, NLL=1.853892/0.461679/5.584766/13.059733; exact=128/200 (0.640000); token=130/202 (0.643564)
  - rephrase_target_true: n=200, NLL=11.828972/11.948886/16.541853/21.752930; exact=0/200 (0.000000); token=0/200 (0.000000)
  - rewrite new<true=1.000000; rephrase new<true=0.985000; rephrase strict exact=0.470000; locality=812/1000 (0.812000)
- **QKL_PROJECTED_ODE_N2**
  - rewrite_target_new: n=100, NLL=0.060920/0.018990/0.095849/2.225948; exact=100/100 (1.000000); token=101/101 (1.000000)
  - rewrite_target_true: n=100, NLL=13.023006/13.204950/17.844669/23.027718; exact=0/100 (0.000000); token=0/100 (0.000000)
  - rephrase_target_new: n=200, NLL=1.917562/0.481277/5.608226/13.167984; exact=125/200 (0.625000); token=127/202 (0.628713)
  - rephrase_target_true: n=200, NLL=11.612630/11.709392/16.437471/21.772470; exact=0/200 (0.000000); token=0/200 (0.000000)
  - rewrite new<true=1.000000; rephrase new<true=0.985000; rephrase strict exact=0.460000; locality=813/1000 (0.813000)
- **QKL_PROJECTED_ODE_N4**
  - rewrite_target_new: n=100, NLL=0.138594/0.033986/0.225417/4.171723; exact=98/100 (0.980000); token=99/101 (0.980198)
  - rewrite_target_true: n=100, NLL=11.866651/11.942427/16.970636/22.881498; exact=0/100 (0.000000); token=0/100 (0.000000)
  - rephrase_target_new: n=200, NLL=2.080734/0.586474/6.153248/13.806213; exact=119/200 (0.595000); token=121/202 (0.599010)
  - rephrase_target_true: n=200, NLL=11.112872/11.091875/16.047032/21.295351; exact=0/200 (0.000000); token=0/200 (0.000000)
  - rewrite new<true=0.990000; rephrase new<true=0.980000; rephrase strict exact=0.410000; locality=821/1000 (0.821000)

## Geometry / compute / integrity

| Model | Method | nodes/active | native→projected qKL mean | terminal qKL | removed mean | violation max | edit-core s | total s | peak alloc GiB | cache append | W0 restore | failures/retry/fallback |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama | OFFICIAL_ALPHAEDIT_N1 | 0/NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | 492.477 | 574.160 | 38.123 | NOT_RECORDED | PASS | 0/0/0 |
| llama | STATIC_SPLIT_OFF_N2 | 10/0 | 1.605867→1.605867 | 4.155921 | 0.000000 | 0.000000 | 549.838 | 628.852 | 41.558 | 1 | PASS | 0/0/0 |
| llama | STATIC_SPLIT_OFF_N4 | 20/0 | 1.500600→1.500600 | 4.155921 | 0.000000 | 0.000000 | 569.509 | 647.478 | 41.558 | 1 | PASS | 0/0/0 |
| llama | QKL_PROJECTED_ODE_N2 | 10/10 | 0.986707→0.639258 | 2.016115 | 0.012208 | 0.000001 | 558.351 | 637.246 | 45.576 | 1 | PASS | 0/0/0 |
| llama | QKL_PROJECTED_ODE_N4 | 20/20 | 0.596617→0.412778 | 1.469940 | 0.018411 | 0.000000 | 609.676 | 687.062 | 45.576 | 1 | PASS | 0/0/0 |
| qwen | OFFICIAL_ALPHAEDIT_N1 | 0/NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | 271.155 | 351.803 | 40.803 | NOT_RECORDED | PASS | 0/0/0 |
| qwen | STATIC_SPLIT_OFF_N2 | 10/0 | 2.637693→2.637693 | 3.748558 | 0.000000 | 0.000000 | 339.000 | 424.916 | 46.803 | 1 | PASS | 0/0/0 |
| qwen | STATIC_SPLIT_OFF_N4 | 20/0 | 2.547012→2.547012 | 3.748572 | 0.000000 | 0.000000 | 353.502 | 433.024 | 46.803 | 1 | PASS | 0/0/0 |
| qwen | QKL_PROJECTED_ODE_N2 | 10/10 | 2.099719→1.696138 | 3.022079 | 0.002387 | 0.000000 | 340.898 | 426.094 | 47.914 | 1 | PASS | 0/0/0 |
| qwen | QKL_PROJECTED_ODE_N4 | 20/20 | 1.491811→1.208903 | 2.325931 | 0.006839 | 0.000000 | 359.699 | 442.368 | 47.914 | 1 | PASS | 0/0/0 |

FULL-FP32, autocast/TF32 off, W0 pointer/bytes restore, create-once results, controller target_true/rephrase/locality influence 0은 모든 terminal cell에서 검증됐다.
Official N1의 qKL/removed-energy/cache-append telemetry는 stock exact bypass 때문에 NOT_RECORDED이다.

## Implementation-control appendix

STATIC_SPLIT_OFF N2/N4는 scientific baseline이 아니라 stock endpoint factorization/splitting control이다. 상세 값은 `implementation-control.csv`와 `analysis.json`에 있다.

## Sequential exclusion

- jobs: 28081 llama sequential-b10x10, 28083 qwen sequential-b10x10
- running cells cancelled: 4; pending cells cancelled: 6
- partial roots/logs are immutable evidence; terminal result count=0; atomic-B100 denominator influence=0.

## Claim boundary

허용: cache-aware one-sided projected N2/N4의 atomic-B100 factual tradeoff.
금지: sequential/lifelong 효과, automatic promotion, continuous-time convergence, 또는 N4 우월성 주장.
