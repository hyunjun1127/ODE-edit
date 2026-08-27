# BGODE-R2 Stage B FP64 two-equality 기술 폐쇄 사실 보고

Stage B는 효능 실험이 아니라 Llama/Qwen의 동일 sealed request000에서 Official AlphaEdit adapter fidelity, termination-free prefix event, FP64 Fisher/two-equality 해와 write 전 원자성을 닫는 기술 gate이다. Dynamic writer action은 두 모델 모두 0이며 과학 promotion은 하지 않는다.

| 모델 | terminal | Official Δ relative error | G rank | equality rank | equality residual | event norm residual | W0 restore |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama3-8B | PASS | 0.0 | 2 | 2 | 6.727e-16 | 1.572e-15 | pointer+bytes exact |
| Qwen2.5-7B | PASS | 0.0 | 2 | 2 | 1.416e-16 | 5.704e-16 | pointer+bytes exact |

Qwen의 과거 R1 FP32 `range(G)` hold는 R2 FP64에서 재현되지 않았다. 방향 range residual은 고정 backward-error bound 안이며 Schur/equality rank는 2이다. 이는 Stage B 수치 구현 통과 사실이지 효능·수렴·barrier attribution 결과가 아니다.

두 모델 모두 fixed z compute1/recompute0, ordered validator call1, history append0, rho/root/localizer0, probability floor/ridge/damping/fallback0, FP16/BF16/TF32/autocast/quantization/storage-cast0이다. 상세 spectrum과 JVP/FD/메모리는 `stage-b-summary.json`에 기록했다.
