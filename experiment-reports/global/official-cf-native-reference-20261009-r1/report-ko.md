# Official CF native reference 입력 보완

Nonce `SH2-GH-OFFICIAL-NATIVE-REFERENCE-20261009-R1` 및 `SH1-GH-OFFICIAL-NATIVE-ORACLE-W0-BINDING-20261009-R1`, parent `USER-OFFICIAL-BASELINES-20261008-R1`.

SH2가 지적한 입력 누락이 맞다. `official/hparams/sources.lock.json`에는 AlphaEdit 원본 evaluator의 commit/SHA가 이미 고정돼 있었지만 실행 배포에는 원본 oracle이 없었다. same-call logits에서 owner 수식을 다시 계산하는 검사는 독립 원본 평가 대조를 대신하지 않는다.

원본 commit `b84624f44dfe8fc6cd9e41df916c44124a0c46dc`의 `eval_utils_counterfact.py` 7,941B/SHA `25c3f49039e7bacb58de6d9ab8139f6f24f21532434a08690e9ec71ea939e145`를 exact bytes로 배포했다. 변경하지 않은 `test_batch_prediction` AST만 실행하므로 원 generation/import/fit 경로는 호출하지 않는다. 별도 reference forward의 NLL·strict booleans를 canonical 같은-state raw와 비교하며 canonical logits를 oracle에 재사용하지 않는다.

공통 API와 고정 tolerance·scope는 `official/evaluation/CF_NATIVE_REFERENCE.md` 및 `official/hparams/cf-native-reference.lock.json`에 기록한다. 기존 W0·qualification 모델에서 고정 first4 actual smoke를 수행한다. NLL abs1e-4 nats/rel1e-5, 산술 집계1e-10pp를 실제 비교 전에 고정하며 token/position/order/분모/strict/성공 bits는 정확히 같아야 한다. 원래 paper 재현 점검1pp나 generation KV tolerance를 새 parity에 대신 쓰지 않는다.

README의 Llama AlphaEdit full2k actual-weight 비교는 별도 완료 증거다. 작은 smoke로 해제하지 않으며, 기존 승인 trajectory의 실제 W20 weights와 동일 cohort canonical raw를 사용해 확인할 수 있다. full2k 완료 전 W20 비교가 끝나야 main을 시작한다는 순환 선행 gate로 해석하지 않는다. 추가 edit/fit/arm/generation은 없다.

검토 중 원본 표시용 Score의 NumPy `around`와 공통 Python `round` 차이를 확인했다. 독립 검토가 지적한 2k 반올림 경계 회귀도 수리하여 표시용 `Score_AlphaEdit_display`만 요청별·cohort NumPy 평균과 반올림 순서로 보완했다. E/G/S 및 원시 Score·optimizer·baseline source/hparams는 그대로다. 원본 summarize 사본은 최종 LF 1byte만 추가됐음을 provenance에 명시했으며 byte-exact라고 하지 않는다.

원본 oracle17/표시3/factual16/preparation7/runtime6, 합계 CPU49개 PASS. 독립 read-only source 검토 PASS이며 실제 pretrained/GPU parity PASS는 아니다. SH1의 같은 B3 first300 기존 raw에 대한 `ACTUAL_MATCHED_SUBSET` 비교는 지원하고, 성공하면 first4 engineering admission을 중복 관측 없이 포함하지만 full2k 증거는 아니다.

Llama W0 producer는 SH1로 지정했다. CF factual/generation 및 zsRE token reference를 모델당 한 번 생성하고 SH4가 원본 provenance를 보존한 검증 reader로 공유한다. 전체 실행 identity를 같다고 바꾸지 않고 consumed computational fingerprint와 별도 consumer binding을 확인한다. `messages/head/2026-10-09-official-llama-w0-sharing.json`이 계약이며, reader 구현은 SH1 소유로 별도 게시가 필요하다. 현재 future READY path는 실제 READY가 아니고 cross-SH reader/API도 아직 구현 완료로 표시하지 않는다.

GH는 실제 pretrained/GPU/Slurm/W&B 실행 및 기존 job 변경을 하지 않았다. 구현 CPU 검사와 실제 owner qualification/제출/완료는 분리한다. 직접 전달 및 source SHA·검산·owner ACK는 후속 영수증으로 기록한다.

## 게시 및 직접 수락

공통 구현 commit `bcdeceba`, source/소유권 게시 main `34001ec0950f00b61e89be753494f40b9da6f70f`, official tree `7a096c467655b0c2b41e6b154e3028a0e3447cf8`를 비강제 게시·remote 확인했다. SH2의 concurrent runner 게시 `18e7fbd9`도 병합해 보존했다. source157 SHA/전체 Python221개 외부 task imports0 PASS.

SH1은 기존 관련 turn `01a11bf8-5fd1-7db3-967d-76a755c1d603`, SH2는 `01a11bf7-a8a4-73b1-be9d-67289ab2c55d`에 exact expectedTurnId로 직접 전달했다. SH4는 idle 확인 뒤 `01a11c38-97f9-7a60-ba08-0f51ba66dd8c`를 시작했다. 세 owner의 각각 nonce `GH-SH1-OFFICIAL-NATIVE-ORACLE-READY-20261009-R1`, `GH-SH2-OFFICIAL-NATIVE-ORACLE-READY-20261009-R1`, `GH-SH4-OFFICIAL-ORACLE-W0-BINDING-20261009-R1`에 대한 명시 `OWNER_ACK accepted=true`를 실제 agent message로 확인했다. transport listener의 ACK 미포착은 read-only recent-turn 확인으로 해소했으며 재전송은 하지 않았다.

SH1은 sameB3 first300 original compare와 portable reader를 구현 중, SH2는 다음 source에 oracle을 결속하며 이미 봉인된61619와 oldjobs를 유지한다고 수락했다. SH4는 SH1 single W0 producer를 수락하고 자기 별도 full-W0 실행 없이 reader/READY를 입력 대기로 분리해 native/resume 준비를 계속한다. 수락은 실제 GPU parity·W0 READY·CF main 제출 완료가 아니다. SH2의61619 held/미배정 및 dependency 표시 수리는 owner 보고를 기록했을 뿐 GH scheduler 조회는0이다.

세부 영수증은 `audits/global/official-cf-native-reference-20261009-r1/direct-delivery.json`. 장기 monitor·heartbeat·자동 재시도 없이 현재 bounded 입력 인계는 완료했다.
