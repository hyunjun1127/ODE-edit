# zsRE W&B 전용 페이지·지표 및 GPT-J 6종 제출

USER 요청에 따라 CF와 분리된 zsRE index와 모델별 saved view를 생성했다.
원격 spec/filter/panel 재조회 PASS이며 browser rendering 및 실제 scientific history는 별도 미관측이다.

- [zsRE index](https://forge.coreweave.com/wandb/wkdguswns2256/layer%20allocation?nw=zsreindex)
- [Llama3](https://forge.coreweave.com/wandb/wkdguswns2256/layer%20allocation?nw=zsrellama3)
- [Qwen2.5](https://forge.coreweave.com/wandb/wkdguswns2256/layer%20allocation?nw=zsreqwen25)
- [GPT-J](https://forge.coreweave.com/wandb/wkdguswns2256/layer%20allocation?nw=zsregptj)

dataset=zsre AND metric_schema=official-baselines-scalar-v1 AND 정확 model filter.
기존 CF/다른9 view와 run history/config는 불변. 원본 remote specs는 ignored local 보존.
`zsre/{W0_first2000,current/pre,current/post,all_seen/post}/`의 Efficacy,
Generalization, Specificity, Specificity_loc_ans, Score, requests를 edits 축에 표시한다.
E/G는 teacher-forced target/request macro, S는 같은 cohort W0 prediction agreement,
loc_ans는 별도 정답정확도, Score는 실측 E/G/S 조화평균. 결측은 생략하고 0으로 채우지 않는다.
현재 official runner가 측정하지 않는 current/pre 및 중간 current endpoint는 빈 패널이며
이를 채우기 위해 새 forward를 추가하지 않았다. W5/10/15/20 current100은 실측 subset이다.

GH 공통 source21a78e27/main a0408faa의 `official_zsre_metrics`를 읽기 전용 채택했다.
GH direct request/ACK turn `01a11cb4-d395-7360-9c73-7bfbac88b063`에서
공통구현·SH1/3/4 전달을 수락했고 READY nonce
`GH-SH2-ZSRE-WANDB-METRICS-READY-20261009-R1`을 수신했다.
서버별 수신 완료 여부는 GH 직접 전달 영수증과 구분한다. logger 복제/기존 job hotpatch0.

## 실제 등록

| Arm | Job | afterany |
|---|---:|---|
| MEMIT |61666|기존 CF frontier 61651,61653,61654,61655|
| FT |61667|61666|
| AlphaEdit |61668|61666|
| BLUE |61669|61666|
| FE |61670|61666|
| SPHERE |61671|61667|
| CPU collector |61672|새6개 전체|

전량 held owner/source/argv/assets/resources/dependency 확인 후 release. 초기 모두 PENDING.
기존 CF61650–61656 수정/취소0. cap4, 각GPU1/CPU6/59392MiB/48h,
collectorGPU0/CPU6/24576MiB/4h, exportNONE/Requeue0. wall은 ETA가 아니다.

첫 MEMIT allocation 안에서 새 zsRE W0 token-prediction reference를 한 번 만들고,
원 parent의 모델별 native B100 smoke를 실행한다. 실제 source/native/factual/checkpoint 및
same-call formula 검산을 통과한 receipt만 후속이 허용한다. independent original oracle로
과장하지 않는다. smoke 이후 각 chain은 별도 Python/model의 fresh cold 2k이며
기존 smoke RAM trajectory를 재개하지 않는다. No generation/FluCon, checkpoint latest1/finalW20 KEEP.

실행 source `ccc1f5d683349b75d1780c49bcf1116b370b91a2`, official tree
`7ff32ba330e661173713dcc20daaa5d0c40b052a`.
Manifest SHA `0a0ff645db896233b6a4f3589ff244324a4e1ffd5bad95e272dac0b9bb07b4c6`;
lock SHA `d0075387c52e2bd36202c8e0e1b8422bb862fb9e7a07e54e7f4e5c7e2c614b04`.
자산 identity `3cba137f6cee5fe0b82c6d46f9b7716d203a3fe383954f9f1ad5dab74bbfdb5e`.
Attempt/log/source/checkpoints:
`/mnt/raid5/janghj/ODE-edit/local/official-baselines-server2/20261008-r1/registration-zsre-r1/`.

최종 caller/tracking/controller/collector 관련 CPU97 PASS, source157/Python260/외부import0 PASS.
전체146 검사 중 old qualification_input compatibility fixture13은 source-member set 변경으로
실패한다. 이를 PASS로 표현하지 않으며 새 zsRE 경로는 그 old-proof bridge를 사용하지 않는다.
별도 독립 reviewer 없음. actual GPU smoke, 온라인 run startup/readback, W20 완료는 미관측이다.
추가 monitor/heartbeat/자동retry/대형전송/삭제0.
