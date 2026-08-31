# FzCB completion-value fast-kill K0 사실 보고서

## 결론

**KILL_FZCB_COMPLETION_VALUE.** K1 Llama/Qwen 실행은 0이다. K0에서 대부분의 기계적 gate는 닫혔지만, case 3002의 seed-00 actual suffix rollout이 고정 range tolerance를 넘었다. 실패 candidate를 삭제하거나 tolerance를 바꾸지 않는 계약에 따라 K0에서 중단했다. 이는 output 보존/편집 성능 결론이 아니다.

| 항목 | 결과 |
|---|---:|
| K0 cases | 4/4 |
| candidates | 20/20 |
| valid equality-null directions | 16/16 |
| actual suffix rollout 성공 | 19/20 |
| actual suffix rollout 실패 | 1/20 |
| max full-model adjoint relative error | 2.29873808e-06 |
| max scaled null residual | 6.22855587e-05 |
| max finite terminal closure | 5.13584041e-07 |
| W0 exact restore | 4/4 |
| cache exact restore | 4/4 |
| direct-z compute/recompute | 4/0 |

## Scientific stop 근거

case 3002 seed-00의 actual suffix equality solve는 상대 range residual `0.000489372702`였고, 고정 tolerance는 `0.00048828125`였다. 초과량은 약 `1.09145e-6`이다. 이 candidate의 actual suffix action과 terminal closure는 `+Infinity`로 보존했다. 나머지 19개 candidate는 finite terminal이었다.

## K0 value spread 및 repeat floor

| case | candidate V_suf spread | repeated-solve noise | 3×noise 초과 |
|---:|---:|---:|:---:|
| 14148 | 6.41812221147e-08 | 0 | PASS |
| 16872 | 3.97776602767e-08 | 0 | PASS |
| 4164 | 3.72019712813e-08 | 0 | PASS |
| 3002 | 6.09652488492e-08 | 0 | PASS |

spread는 모든 case에서 exact repeat noise 0보다 컸지만, K0 actual rollout range gate 실패가 우선한다. 따라서 predictive ordering 가설을 K1 denominator에서 검정하지 않았다.

## 구현·불변식

- Llama3-8B-Instruct × Official MEMIT direct-z, FULL-FP32.
- explicit Kronecker 0, dense inverse 0, value-gradient/CBF/full-QCQP/AlphaEdit 0.
- actual full-model JVP/VJP adjoint, fixed-z corrector, exact-copy candidate reset, W0/cache rollback을 사용했다.
- K0 IDs: 14148, 16872, 4164, 3002. K1 sealed IDs는 소비하지 않았다.
- technical lineages 30449/30460/30472/30486은 denominator 0이며 immutable 보존했다.
- terminal science job: 30543, COMPLETED, elapsed 01:40:03.

## Identity

- execution source: `878552e3a7b94439949a5bb02ee0a3c3ed02373f`
- terminal result SHA256: `f1bbc3045bb18c15c63eed99b803a3f217a64cf7bea038868ce2c685fe4faa00`
- pre-GPU receipt SHA256: `01f7fb202f2909d97a1ee106ab30054ed8a3ce5d2eac97bf57db837845e29aa1`
- promotion: false
- origin/main integration/push: 0 (GH handoff only)
