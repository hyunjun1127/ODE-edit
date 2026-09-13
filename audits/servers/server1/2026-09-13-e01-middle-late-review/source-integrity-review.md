# E01 Middle/Late 완료 관측 — 소스·복원 독립 점검

상태: `PASS_WITH_EXPLICIT_LIMITATIONS`. 이번 점검은 CPU 소스/파일/receipt 결속이다. 독립 성능 reducer 및 checkpoint tensor 차이 계산은 별도 담당 산출물이며, 실제 모델·GPU·scheduler·원격 접근은 수행하지 않았다.

## 정확한 변경 범위

기존 native 실행 source `b51dcf5ab825608bee81dd13549318d8d267e835`에서 observation source `58f50a25809779918b22ad0aceded732c097eab4`로의 Git diff는 신규 5파일, 289줄이다. 기존 파일 수정은 0이다.

- `endpoint_observation_resume.py`: 저장 endpoint 복원 후 빠진 past 관측만 수행.
- `performance_schema.py`: 원본 `seen-full.json`에 없는 `request_order`를 크기/행번호가 아니라 전체 ordered case/prompt/target identity를 확인한 뒤 생성.
- `prepare_fullseen_resume.py`: 기존 실패와 완전한 W/M endpoint 및 Current100 재사용을 결속.
- `launch_endpoint_observation.sh`: observer-only 실행.
- `tests/test_performance_schema.py`: 잘못된 order/hash/중복/분모/분자 등을 거부하는 fixture.

Native 방정식·target·history·precision·tolerance 변경은 이 diff에 없다. 새 runner는 editor/native builder를 import하지 않고, 저장된 endpoint를 `restore_checkpoint`로 적용한다. `new_native_batches/new_z/new_history_append=0`은 실행 receipt의 선언값이며, 관측 call graph가 이를 뒷받침한다. 독립 native-event 계측기로 횟수를 다시 측정했다고 주장하지 않는다. 저장 W/M를 복사하는 fixture 복원은 새로운 edit/history append가 아니다.

## 완료 결속

| endpoint | 신규 past requests | 완전 shard | 재사용 Current | full-seen | program elapsed(s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| B060 | 5900 | 47 | 100 | 6000 | 2800.919617 |
| B100 | 9900 | 78 | 100 | 10000 | 4656.180603 |

두 terminal 모두 `ENDPOINT_CURRENT_FULLSEEN_OBSERVATION_COMPLETE`이며 output 내 failure receipt가 없다. 125개 shard의 raw/receipt 전부를 재해시하고 정해진 연속 범위·endpoint 결속·초과/누락 파일 부재를 검산했다. 모든 shard 행에 기존 Current100을 정확히 이어 붙인 결과가 최종 JSON 행과 동일하다. 이것은 denominator/merge 결속이며 독립 성능 수치 검산을 대체하지 않는다.

기존 원본 full-seen도 `current_rows_reused=true` 및 `CHECKPOINT_FINAL_W_ON_ALL_SEEN_REQUESTS`로 저장되어 있다. 원본은 실제 AlphaEdit BLUE L4-only source의 B060/B100이며, 신규 평가에 5-layer/JV/다른 endpoint를 끼워 넣지 않았다. 원본과 replay의 W/M SHA는 서로 다르며 이 점검을 trajectory equivalence로 승격하지 않는다.

## 실제 guard의 범위

매 shard 전후에는 selected L4 W와 native M의 전체 tensor SHA를 비교한다. 모델의 모든 parameter에 대해 pointer/version을 확인하고 Python/NumPy/Torch/CUDA RNG가 변하지 않았음을 확인한다. **매 shard에서 모든 parameter/buffer byte를 다시 해시한 것은 아니다.** 특히 buffer의 per-shard byte 검사는 별도 기록되지 않았다.

Outer `FixtureTransaction`은 진입 시 모델 전체 parameter/buffer CPU 사본을 보관한다. 종료 시 저장 registry를 복구하고 모든 포착 tensor와 history의 pointer/byte SHA를 비교한다. 두 terminal 모두 `pointer_bytes_exact=true`, `rng_restored_exact=true`, `host_backup_bytes=32121053440`, `changed_version_count=1`을 기록했다. 이 full-byte 복원은 새로 로드한 pinned base 상태로의 반환이며, 버전 카운터 원복은 `NOT_CLAIMED_NATIVE_COPY_INCREMENTS`다. Hook registry는 source에서 재설치하지만 독립 per-hook equality 수치는 없다.

평가가 끝난 실제 endpoint W/M에서 성능을 저장한 뒤 outer transaction이 W0로 돌아간다. 최종 반환 후 W0 성능을 endpoint 성능으로 오인하는 경로가 아니다. `restore.json`의 `C0_restored=false`를 covariance 자산까지 복원했다는 뜻으로 해석하지 않는다.

## 평가 의미와 한계

실제 실행한 helper의 full sequence forward는 historical microbatch16, 수동 left-padding, position override 없음, `torch.no_grad`, 전체 target-token 평균 NLL이다. Tokenizer는 right-padding 설정을 유지한다. RS/PS는 `new_nll < true_nll`, NS는 `true_nll < new_nll`, tie는 실패다. Pairing은 case ID·prompt index·prompt+new/true target의 hash로 결속하며 row-index-only가 아니다. 정규화 adapter는 원본 파일을 수정하지 않는다.

신규 strict/token secondary는 기존 reducer가 저장한 값이며 원인·상대 우열 해석을 덧붙이지 않는다. 이번 두 observation task의 완료는 전체 E01 20cell 완료가 아니다. Original/replay 성능 유사성 또는 차이가 tensor fidelity를 판정하지 않는다. 기존 `NONEXACT_CAUSE_UNRESOLVED`를 hardware 원인으로 귀속하지 않았다.

## 재해시와 비용 범위

점검 index는 333개 파일: 326개 full SHA256, 7개 기존 대형 모델/P/endpoint 자산은 sealed SHA+현재 size/access만 확인했다. Endpoint tensor full 재해시/차이 계산은 별도 CPU 담당 결과와 결속한다. 이전 native 전체 output을 중복 재감사하지 않고 기존 preservation receipt를 재사용했다.

Member root: `edf71c309eff18e69dc5ea278c7d67d3ee0564365379dfea052c5c250348a248`.

JSON: `source-integrity-review.json`, SHA256 `f62f59d10858e1e295aacd83079dbdb2546edf7073d09f096178f4722d663a5d` (133538 bytes).

`observe_guarded` wall 합계는 B060 2467.132626초, B100 4269.783501초다. 여기에는 평가와 hash/RNG guard 비용이 함께 들어가며 순수 forward GPU kernel 시간이라고 부르지 않는다. Program elapsed에는 load/입력확인/merge/복원이 포함되며 allocation 시간과도 구분해야 한다. 과거 실패 allocation 9727/9748초는 새 관측 시간에 섞지 않는다.

재현 명령:

```sh
python3 -m project.run_scripts.baseline_mechanism_first.completed_review_integrity \
  --root /mnt/raid5/janghj/.codex/worktrees/odeeditsh1-e01-fullseen-schema-repair-r3/local/baseline-mechanism-first-e01/20260912-v1/fullseen-schema-repair-r3 \
  --execution-worktree /mnt/raid5/janghj/.codex/worktrees/odeeditsh1-e01-fullseen-schema-repair-r3 \
  --repo /mnt/raid5/janghj/.codex/worktrees/odeeditsh1-e01-completed-detailed-review-20260913-v1 \
  --output <새 source-integrity-review.json 경로>
```

GPU/model/Slurm/new evaluator/native action=0. Scientific promotion=false. 별도 과학적 원인 종합은 GH 소유다.
