# BGODE-R2 Stage C multi-token topology 사실 보고

Stage C는 outcome-independent tokenizer-token manifest로 Llama/Qwen 각각 unequal non-prefix, source-prefix, target-prefix를 검사했다. 여섯 scheduler cell은 모두 exit0 terminal을 게시했으며 raw result는 변경하지 않았다.

| 모델 | topology | event partition | FP64 two-equality | W0 |
|---|---|---:|---:|---:|
| Llama3-8B | unequal non-prefix | 정상화 PASS | NUMERICAL_IMPLEMENTATION_BOUNDARY | exact |
| Llama3-8B | source-prefix | 정상화 PASS | PASS | exact |
| Llama3-8B | target-prefix | 정상화 PASS | PASS | exact |
| Qwen2.5-7B | unequal non-prefix | 정상화 PASS | NUMERICAL_IMPLEMENTATION_BOUNDARY | exact |
| Qwen2.5-7B | source-prefix | 정상화 PASS | PASS | exact |
| Qwen2.5-7B | target-prefix | 정상화 PASS | PASS | exact |

두 unequal non-prefix boundary는 event/JVP 오류가 아니다. event normalization, score centering, serial JVP/FD, Official adapter fidelity, FULL-FP32 model과 W0 restore는 통과했고, 고정 FP64 `range(G)` backward-error gate만 거부했다. Llama direction residual은 1.282e-2/4.709e-5, bound는 1.560e-6/1.362e-6이다. Qwen은 2.401e-2/9.108e-2, bound는 7.456e-4/1.624e-4이다.

계약상 Stage D는 Stage B와 Stage C가 모두 통과한 뒤에만 release된다. 따라서 tolerance/pinv/ridge/damping/fallback을 바꾸거나 재제출하지 않고 Stage D를 HOLD한다. scientific promotion은 false이다. 상세 spectrum과 모든 terminal identity는 `stage-c-summary.json`에 있다.
