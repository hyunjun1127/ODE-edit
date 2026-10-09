# FE author clamp/step + history: 네 cold chain 등록 인계

Nonce `USER-SH-FE-AUTHOR-HPARAMS-2K-CF-ZSRE-20261010-R1`.
부모가 SH1/SH2에 동일 사용자 지시를 직접 전달했으며 GH는 명시 owner ACK를 회수했다.
새 독립 배정/추가 chain을 만들지 않고 같은 accepted turn에 조정 메시지만 exact steer했다.
공통 author profile/runner는 SH1, server2 adapter는 SH2, README는 GH 단독 소유다.

## 비교 범위

기존 MEMIT_FE_HISTORY의 persistent history ridge, 매 batch 진입 모델의 z fitting/replay,
final-model native key once-per-layer append를 유지하고 원저자 clamp/step만 적용한다.
원저자 source는 `jugechengzi/FE@478134dfb24b43f4e18b47e8500893ce3f9cc50f`이다.
DOW-KE exact reproduction을 뜻하지 않으며 native MEMIT_FE 및 이전 history 행과 분리한다.

| 모델 / owner | clamp | steps | lr | decay | loss layer |
|---|---:|---:|---:|---:|---:|
| Llama3 / SH1 | 0.75 | 35 | 0.1 | 0.5 | 31 |
| Qwen2.5 / SH2 | 1.0 | 35 | 0.5 | 0.001 | 27 |

공통 KL .0625/C0 weight15000/L4–L8/FP32 eager/TF32 off, seed0, first2000,
BS100×20을 유지한다. 각 모델 CF와 zsRE는 독립 cold W0/H0이며 각 서버에서 CF→zsRE 직렬이다.
기존 작업 뒤 자원 dependency로 등록하고 cap/메모리 증액·선점·기존 job 취소는 하지 않는다.
기존62262/62263 취소 상태는 그대로이며 이번 네 행과 별개다.

CF는 기존 factual E/G/Loc/Score 및 FLU/CON DEFERRED, zsRE는
`official.evaluation.zsre_paper` 공개-query request-macro Eff/Gen/loc_ans 정확도다.
zsRE generation/W0 prediction agreement Loc/옛 token-prefix 평가로 회귀하지 않는다.
W0 및 매 committed batch checkpoint, 최종 W20 selected weights+H+contexts/RNG/identity를 보존한다.
원 가중치/산출물 불변, CF generation consumer 완료 전 최종CP 삭제 없음.

## 실제 등록 영수증

총 네 job의 actual held 검사/release와 초기 PENDING 영수증을 회수했다.

| owner | dataset | 실제 job name (ID) | afterany |
|---|---|---|---|
| SH1 Llama | CF | official-s1-cf-llama3-memit-fe-author-history (62529) | 62259:62260:62261 |
| SH1 Llama | zsRE | official-s1-zsre-llama3-memit-fe-author-history (62530) | 62529 |
| SH2 Qwen | CF | s2-qwen25-cf-fe-author-history (62531) | 62081:62083:62085 |
| SH2 Qwen | zsRE | s2-qwen25-zsre-fe-author-history (62532) | 62531 |

SH1 source `9c3fe23282bcaf2e353b1494913a323e2f037d39`, SH2 source
`a8c4c611a37210ddfa9e7db222dca6678d4cd678`. 공통 author profile SHA
`1ffbf9adeec9525b88db3d86a6dc4ace8b98bdaff1d5eb4e4ee064c24b35b125`.
각 GPU1; SH1 CPU8/98304MiB, SH2 CPU6/59392MiB, 48h ceiling(ETA 아님).
각 owner 현행 cap4와 기존 실행을 보존하며 신규 두 chain은 직렬이다.
SH1 CPU44, SH2 최초 준비 CPU11 및 최종 게시 CPU16(별도 SPHERE lifetime3 포함), source166 PASS는 GPU 검증이 아니다. zsRE 전체2K query CPU parity
입력/target mismatch0 근거를 owner가 결속했다. 실제 W20/online readback은 아직 미관측이다.
정확 source/tree/config/profile SHA, dependency/resources/관측 상태 및 W20 CP 경로는
[통합 receipt](../../../audits/global/fe-author-hparams-2k-20261010/coordination.json)에 결속한다.
[SH1 제출 영수증](../../../audits/servers/server1/fe-author-hparams-2k-20261010/submission.json) ·
[SH2 제출·행 영수증](../../../audits/servers/server2/fe-author-hparams-2k-20261010/table-rows.json).

## GH 검산 및 경계

main `fbb3a2e3`의 기존 FE_HISTORY 표 안 네 행을 그대로 사용한다. 새 표/추가 행은 없다.
`verify_rows.py`는 네 actual 고유 ID·held 검사/release·owner receipt와 README 상태를 대조하며,
나머지 표 행과 모든 기존/빈 성능 셀이 변경되지 않았는지 확인한다.
원저자 초안과 기존 native JSON을 대조한 GH CPU 검사에서는 두 모델 모두 clamp4→.75/1,
steps25→35 외 차이가 없었다. runtime parser override 검사는 SH1/SH2 실행 source 증거로 분리한다.
CPU 검산을 실제 GPU/W20/온라인 기록 성공으로 보고하지 않는다.
GH scientific source 편집/Slurm 제출/GPU 평가0. 실험 완료 대기/장기 모니터/자동 retry 없음.
원 raw/weights/credentials는 Git에 추가하지 않는다. 소형 source/receipt만 공유하므로
NO_BROADCAST_NOT_REQUIRED; 새 대형 전송은 요청하지 않았다.
