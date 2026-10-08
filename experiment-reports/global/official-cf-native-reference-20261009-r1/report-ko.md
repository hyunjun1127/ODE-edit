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
