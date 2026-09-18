# ENFC 사용자 지정 sequential 네 정책

최신 사용자 지시: `EN-S / EN-F / EN-COV / EN-F4` sequential을 실행하며, M B1·B2는 중단한다.
각 정책은 동일 W0/zero M4에서 O0 first1000, B100×10을 실행한다. M 완료 dependency는 없다.
기존 design/contract의 숫자와 원 native/geometry/optimizer/evaluator 코드는 변경하지 않는다.
기존 S cells에 없던 EN-S/EN-COV/EN-F4의 sequential 적용은 이번 사용자 override다.
M→S 과학 gate는 NOT_ESTABLISHED이며 실행 승인을 검증 PASS로 쓰지 않는다.

`sequential_runtime.py`는 B1의 정확 native capsule만 재사용하고 B2 이후에는 각 chain의 실제 W/M에서 native fit한다.
Current native/canonical old+new guard와 canonical Past64 guard를 함께 적용한다.
Past64는 받은 모든 사건의 최신 subject/relation을 기준으로 current overwrite를 제외한 고정 SHA 순서다.
최종 selected W에서 원 native finalizer로 M4를 한 번 갱신한다. Candidate 중 history append는 없다.
매 batch W4/M4/RNG/context/received ledger/active registry/next index의 atomic checkpoint와 commit을 남긴다.
GPU continuation은 NOT_TESTED이며 자동 resume/submission 프로그램은 없다.

EN-COV는 W0 대비 누적 activation drift이며, EN-F4의 4 gradient×6 trial도 그대로다.
T=SKIPPED_USER_DIRECTED / full_numerical_validation=NOT_ESTABLISHED.
별도 T/FD/ULP를 제출 전에 되살리지 않는다. 실제 IO/finite/state 오류는 정상 실패다.

Official current entry/native/selected 평가와 greedy32는 selection 이후다.
B5/B10 fullseen 및 Dev128을 저장한다. W0 paired 관측은 기존 exact rows를 재사용한다.
추가 R/L·N4·CA·KL-P·SCALE chain이나 teacher 재생성은 없다.
원 M partial/source/비용과 취소 기록은 보존한다. 기존 자료 삭제/이동 및 대형 전송은 없다.

## 실행

CPU 검사는 `python -m project.run_scripts.single_layer_edit_preserving_correction.sequential_preflight --repo <worktree>`.
동일 package의 `sequential_control freeze` 뒤 `submit`으로 create-once 4-arm array를 held inspection 후 release한다.
각 job은 1GPU/8CPU/60416MiB/72h/exportNONE/Requeue0, array%2다.
최초 실제 correction 또는 합법적 fallback → checkpoint/commit/history → observer → B2 entry 뒤 pause한다.
