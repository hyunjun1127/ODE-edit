# BGODE-R3 G1 pre-GPU 사실 보고서

## 판정

`R3_G1_PRE_GPU_PASS`; G2는 아직 HOLD이다. model/GPU/Slurm action은 0/0/0이다.

## 자연 요청 봉인

- sealed canonical stream/order에서 tokenizer-only scan을 수행했다.
- 두 모델 모두 자연 `unequal-non-prefix`만 존재한다.
- multi-token G1을 충족하는 최초 공통 요청은 ordinal 26, case 17454이다.
- Llama target/source token 길이는 2/1, Qwen은 2/1이다.
- 나머지 세 topology/model은 총 6 cell 모두 `NATURAL_TOPOLOGY_UNAVAILABLE`; 합성 대체는 0이다.

## 구현 경계

- Official AlphaEdit fixed-z와 genuine P-inside ordered proposal을 재사용한다.
- factor Gram FP64 Frobenius normalization을 JVP/controller/write에 동일 적용한다.
- 모든 내부 prefix에서 5 actuator serial forward-JVP와 central FD를 검사한다.
- 동일 fine partition의 W0 q0를 create-once raw artifact로 봉인한다.
- t=0 Full/Fisher identity를 검사한 뒤 `h_probe=T_AE/32` Fisher write 1회와 exact W0 restore만 수행한다.
- history append0, localizer/root/ridge/damping/floor/fallback0, promotion=false이다.

## gate

- focused CPU: 32/32 PASS
- py_compile/bash/session: PASS
- Llama/Qwen tokenizer·EasyEdit·HF·P/stats preflight: PASS
- implementation source: `cf8a2b4d05dd13cb5a4525bc7592b05b90810612` / `50c95b3d430a0681ca8701bb40708963b6439d53`
- source members root: `f60c76cbbbc3ae0312ab41ab41fedd5672b6ecb470e9eab4fc5e2cc54c4c8ac9`
- natural manifest identity: `7e512324adc789601df25c3767ba4ebc838a258452580db1aa7970937d563b40`

## release

G1 array mapping은 0=Llama, 1=Qwen, `%2`, project cap3이다. 제출 직전 unrelated/P1R55 포함 active GPU를 재계수한다. G1 natural unequal-nonprefix PASS 없이는 해당 모델 G2를 release하지 않는다.
