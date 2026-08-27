# BGODE-R3 G1 자연 multi-token 사실 보고서

## 판정

`R3_G1_NATURAL_UNEQUAL_NONPREFIX_TERMINAL_PASS`이다. sealed ordinal 26/case 17454의 자연 unequal-non-prefix 요청에서 Llama/Qwen 2/2가 terminal-valid이다. 다른 topology 6 cell은 `NATURAL_TOPOLOGY_UNAVAILABLE`이며 합성 대체는 0이다. scientific promotion은 false이다.

초기 job 26885의 두 task는 모든 과학 동작과 W0 restore 뒤 관찰 receipt JSON 변환에서만 실패했으며 endpoint denominator는 0/2이다. 과학식·요청·허용오차·writer를 바꾸지 않은 TECH-R1 job 26899가 2/2 terminal을 새 namespace에서 완성했다.

## 모델별 gate

| 모델 | event 수 | normalization residual | max FD abs / rel | Official adapter rel-F | t0 Full−Fisher | T_AE | wall(s) | W0 restore |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| llama3-8b-inst | 256511 | 6.661e-16 | 5.589e-03 / NOT_RECORDED | 0.000e+00 | 7.494e-15 | 21.593080 | 221.192 | PASS |
| qwen2.5-7b-inst | 304127 | 0.000e+00 | 1.715e-02 / NOT_RECORDED | 0.000e+00 | 1.585e-13 | 14.590098 | 268.994 | PASS |

두 모델 모두 fixed-z compute1/recompute0, node/terminal history append0, localizer/root0, retained factor0, FP32 model·FP64 event/controller·FP32 physical write 경계를 지켰다. Genuine P-inside ordered proposal의 Official dense endpoint fidelity, 전체 내부 prefix serial JVP/central FD, fine-event normalization/score-centering, equality residual, t=0 Full=Fisher, h_probe=T_AE/32 one-step write와 pointer+bytes exact W0 restore를 통과했다.

## G2 경계

두 모델의 자연 unequal-non-prefix G1 gate는 PASS했다. G2는 결과 전에 봉인하는 Cauchy numerical lock이 아직 필요하므로 이 package 자체는 실행을 release하지 않는다. 모델별로 동일 ordinal/request/W0/fixed-z/T_AE와 N={4,8,16,32}, arms={Plain,Fisher,Full}를 사용하며 unavailable topology를 대체하지 않는다.

## identity

- source HEAD/tree: `072a8addee4a331929db014752c0ee246a5f01b2` / `596d77414965f03bb5e1c18e8e094aab19b8b07a`
- TECH-R1 job: 26899, terminal 2/2
- initial technical exclusion: job26885, endpoint 0/2
- promotion: false
