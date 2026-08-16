# P2R6 RED-R2 Phase-1 보고서 정정 부록

- 상태: `APPEND_ONLY_CORRECTION`
- 기존 보고서 파일은 변경하지 않았다.
- 기존 보고서 SHA256: `1c6861b58de2662ef0bad5cf766397b6455a8ef8aca278c3da19babf704973be`
- 기존 엄격 중단 분류: 변경 없음 (`PHASE1_TECHNICAL_INVALID_STRICT_STOP`)
- 과학 실패: `false`

## 정정 사실

기존 보고서 §5.2의 마지막 열은 `W full-six target-new NLL`로 표기됐으나, 해당 열의 원자료는 terminal z-intervention objective였다. 그 값은 W-only full-six target-new NLL로 사용하지 않는다.

W-only 값은 각 유효 endpoint의 마지막 writer step(`step-07.json`)에 기록된 `next_w_nll_by_request` 10개 값의 산술평균이다.

| 모델 | arm | 정정 W-only full-six target-new NLL | source receipt SHA256 | source identity SHA256 |
|---|---|---:|---|---|
| Llama | A0-CAP | 1.1113469866737433 | `a0411e01941cc40dbf59d735ea9698131944577e609a26c20c0df4de48abc0d5` | `dd9c89bc76d5bcc067b99f969ff2967888360a18bc05ea3cabfd92cf9bf66672` |
| Llama | AETA-CAP | 0.5101360967732035 | `6385a3e6330ff6764aebc8466742c6eb08c145f858bc4b48042c62c5ebec7d2a` | `7448d9cd09705640f27f77d1895daf32bad5d3fd8c2596584c049176fe00d8d2` |
| Qwen | A0-CAP | 0.0007309559659915976 | `ea56e11b67fe6ebf1fffa116e997fef94cb459c8e14528828a2bc9d353e0c78d` | `e647a0e122e20222308c704a039e2ee44a5c0b7b9a9f1cc41053e9fe953112c0` |
| Qwen | AETA-CAP | 0.0019331367017002775 | `8b92be26b776ee80b022c9b7b01c8d3c682460a79a01aa540ce056bf39b18599` | `6a18500909dac051c76ef0da20a5b9423391b76804177f11614b16b9b4f7d470` |

AR/AS endpoint는 기존 기술 실패로 `NOT_RECORDED`이며 이 부록에서 보간하지 않는다.
