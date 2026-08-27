# BGODE-R3 G0 CPU 수학 폐쇄 사실 보고

상태는 `R3_G0_MATHEMATICAL_PASS_G1_NOT_RUN`이다. G0는 모델·GPU·Slurm을
사용하지 않았고 자연 request를 고르거나 성능 endpoint를 만들지 않았다.
`scientific_promotion=false`이다.

## 실행 경계

- authoritative contract: 25746 bytes, logical records 1089, SHA256
  `fa0a6923efe944d485dc49802772b4d08f7d02f7c7113dbee302e7d9744642dc`
- base main: `8e60f5c2d8de998f128a20e25c09f66aff770597`, tree
  `bc9aae754dec2ea6a55dcf23c2b23d4e644bb299`
- 지정된 R2 source/report 10개를 전부 읽고 R2 bytes는 변경하지 않았다.
- R3 focused tests: 28/28 PASS
- R2 legacy focused regression: 21/21 PASS
- Python compile, diff-check, session boundary: PASS
- model/GPU/Slurm/result action: 0/0/0/0

## 폐쇄된 수학 항목

1. unequal non-prefix, common-prefix divergence, source-prefix-target,
   target-prefix-source 네 topology가 같은 고정 event partition으로 정상화된다.
2. source-prefix에서는 source-exclusive token identity를 barrier/Fisher/q0에
   유지하고 equality `sS`에서만 집계한다. target-prefix에서는 target-exclusive
   집합만 하나의 target macro로 collapse한다.
3. streaming reducer와 tiny-vocabulary brute force가 FP64 backward error 내에서
   probability와 score 모두 일치한다. rare-event와 tokenizer/output-head vocab
   불일치 fixture도 log-space에서 유한하다.
4. W0 fine-event `q0` bytes가 고정되고 target-only exponential tilt, target-excluded
   KL decomposition, anchored excess identity가 일치한다.
5. `C=[sY-muN,sS-muN]`, `d=[1,0]`은 target 증가/source 감소와 연속시간 closed
   form을 재현한다. redundant equality는 허용하고 inconsistent system은 닫는다.
6. `alpha=||B||F`를 factor Gram FP64로 계산한 normalized basis가 raw
   `beta/alpha` write와 일치하고, block coefficient norm이 physical block-Frobenius
   norm과 대응한다.
7. solver는 `G` inverse/eigendecomposition 없이 equality particular/null basis와
   `XZ`를 직접 SVD한다. fixed rcond는 `max(m,n)*eps_FP32`이다.
8. factor-space solution은 dense KKT와 일치하고 equality/stationarity residual을
   통과한다. retained modes의 symmetric-difference/JVP projection도 통과한다.
9. t=0 Full=Fisher이며, 일반 node에서
   `gTuB=gTuF-deltaTGdelta` 국소 항등식이 성립한다.
10. affine one-step/split identity와 nonlinear first-order Euler convergence가
    성립한다. node factor retention과 prohibited production identifier/call path는 0이다.

## 아직 하지 않은 일

G1 tokenizer-only natural topology scan, live model JVP/FD, normalized physical
probe write/restore, G2 `N={4,8,16,32}` convergence, attribution 및 성능 평가는 모두
실행하지 않았다. 각 모델의 natural unequal-nonprefix G1이 통과하지 않으면 해당
모델 G2는 release하지 않는다. 미래 fixed-strength curve
`T={0.5,1,2,3,5}, N=32`는 predeclared일 뿐 실행하지 않았다.
