# Llama zsRE saved W20 weights 재평가 준비

USER `USER-GH-ZSRE-SAVED-WEIGHTS-2K-REEVAL-20261009-R1` 정본 전체를 읽고 수락했다.
기존 MEMIT_FE_HISTORY CF 3개 작업과 별개이며 그 제출/실행을 취소하지 않는다.

| 원 method | 원 job | 최종 CP bytes | 현재 단계 |
|---|---:|---:|---|
| FT | 61716 | 234898913 | W20 CP fullSHA 확인 |
| MEMIT | 61717 | 1174424977 | W20 CP fullSHA 확인 |
| AlphaEdit | 61718 | 5284844677 | W20 CP fullSHA 확인 |
| AlphaEdit-BLUE | 61719 | 2113948569 | W20 CP fullSHA 확인 |
| MEMIT-FE | 61720 | 1174424785 | W20 CP fullSHA 확인 |
| SPHERE | 61721 | 5284844677 | W20 CP fullSHA 확인 |

6개 모두 actual W20/2000/20 calls terminal, original source/config/stream/ordered case digest,
latest pointer 및 실제 checkpoint bytes/fullSHA를 대조했다. 같은 inode/SHA replica 중복0.
공통 base model 및 tokenizer 기존 파일도 fullSHA로 대조했다. 실제 tensor 복원/GPU 평가는 아직0.
own 승인 완료 OURS zsRE 추가 대상0.

## 구현 및 검산

`official/runners/server1/zsre_reeval_inventory.py`는 위 inventory를 CPU-only로 생성한다.
`zsre_reeval_restore.py`는 원 checkpoint identity/method/W20/cursor, exact selected names,
FP32/shape/finite/history schema를 모두 검사한 다음 selected W만 복원한다.
native apply, optimizer, fit, W0 관측 또는 editor history 실행이 없다.
비편집 파라미터의 pointer/version을 보존하고 원 checkpoint는 쓰지 않는다.
작은 CPU fixture 5 tests PASS: selected-only restore, partial/identity/nonfinite/history 거절.
이는 actual GPU 복원/출력 parity가 아니다.

## 남은 정확한 입력

확인한 main `7f409f48`에 GH 소유 `official/evaluation/zsre_paper.py`가 아직 없다.
exact API/SHA와 eval-only tracking authority/schema READY가 필요하다.
따라서 full2000 native query CPU parity 및 actual eval-only caller/collector 최종 결속은
`SOURCE_INPUT_PENDING`, 신규 job IDs는 **[]**다. 사용자 승인 대기나 성능 gate가 아니다.
공통 evaluator/logger를 복제하거나 예정 API를 완료본이라고 표시하지 않았다.
READY 수신 후 새 source/config freeze에만 적용하며 기존 frozen job을 hotpatch하지 않는다.

## READY 채택 및 최종 caller 준비 업데이트

위 입력 대기는 GH main4533756e/implementationf1a00379 게시로 해소했다.
공통 문서 ZSRE_PAPER.md 전체를 읽고 evaluator SHA
`d6a5b34eafd27660a2dee4632c638b6bf4c3614246071711cf5159a002415a45`,
query parity module SHA `9883f16036525278bbfcdb45f5f08cc8c799f638a26f91b4d9da2a0d70103803`,
tracking schema SHA `633344063046eba678595e436b4ff3d3162a062e9b31ff244bc48ab09a9237f1`에 결속했다.
원 stream file SHA는 `f42ee4bc6e98b1133e48dc81102201ccd85ccc917a97160ee3c6a0f3a38f378c`.
로컬 tokenizer와 전체2000 요청 CPU compare_queries 결과 input/target mismatch0,
24,535 queries = Eff6035 + Gen6035 + Loc12465(BOS 포함 public loader)다.
query SHA `7909c567881531b62db728ecc06308a3387324cc4e1c907b76bb50d8dd862efc`.
이는 실제 pretrained numerical-output parity가 아니다.

새 `zsre_reeval.py`는 선택 weights만 복원하고 공통 evaluate를 한 번 호출한다.
weights/RNG 및 CP 불변을 검사하며 W0/edit/fit/generation0. eval-only identity와 진행률/최종 scalar는
동일 official.tracking을 통해 새 W&B run에 기록한다. collector는 saved prediction/target에서
요청 macro를 독립 재집계하고 이전 endpoint와의 delta/분모/source를 기록한다.
`zsre_reeval_submit.py`는 현재 cap4/전체 admitted DAG에 6GPU 평가 및 1CPUcollector를 연결한다.
1GPU/CPU8/65536MiB/4h, collector GPU0/CPU8/24576MiB/4h이며 freeGPU agent wait는 없다.
CPU caller/restore/common/fakeSDK51 tests PASS, source166 SHA/Python320/import0 PASS.
실제 GPU/online history 완료는 미관측이며 실제 제출 결과는 별도 receipt에 기록한다.

원 CP/raw/model/기존 job KEEP; 실제 전송/삭제/취소/forward/fit0.
Eff/Gen/Loc 값은 이번 준비 단계에서 변경하지 않았다. README는 GH 단독 통합.
`NO_BROADCAST_NOT_REQUIRED`: same-host 자산/CP, compact source와 inventory만 Git 공유.
