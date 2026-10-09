# Qwen 미시작 작업 평가 코드 갱신

직접 USER의 미실행 PENDING 취소·최신 평가 코드 반영·재등록 지시를 적용한다.

## 취소 및 보존

원 source `69bfbb2cdffe24072733950c671e47597ff9fbd5`의 현재 owner/Command/WorkDir/source/state를 검산했다. 61899 및 61901–61922의 미시작 23개 작업을 hold 후 후속부터 취소했다. 실행 중 FT CF 61898 / zsRE 61900은 그대로 보존한다. 원 source/raw/checkpoint/history를 변경하지 않았다.

GPT-J 평가 전용 61942–61948은 취소하지 않았다. afterany 해제 경합 방지를 위한 임시 resource hold이며 새 Qwen frontier에 연결 후 같은 ID/source로 release한다.

## 변경과 검증

새 10개 본실험만 공통 `zsre_paper.evaluate`의 Qwen public query 및 Loc request-macro를 사용한다. CF scorer/생성 일정, native fit/hparams/precision/seed/BS100×20, checkpoint는 유지한다. 기존 W0는 original receipt/raw SHA와 물리 자산/runtime/tokenizer를 확인하고 별도 consumer binding으로 참조한다. 구 관측을 새 public-query W0로 relabel하지 않는다. CF W0+W20 생성 일정은 변경하지 않는다.

CPU 43 tests PASS. source166 SHA 및 외부 task import0. 실제 Qwen tokenizer/전체2000 요청 query24858의 public AST 비교 input/target mismatch0 (R6691/P6691/N11476). native 편집 loop AST는 기존 최소 indentation repair와 동일. GPU qualification은 `NOT_RUN_USER_DISABLED`; 실제 새 GPU/온라인 PASS가 아니다.

4개 resource lane, 원 FT 2개 및 GPT-J eval-only 후속을 포함해 cap4를 지킨다. 최신 source 게시/봉인 후 신규 10 GPU + 12 GPU0 보존 단계 + collector를 held 검사/release한다. FT 보존 단계는 원 provenance의 보존만 기록하며 새 source로 소급 adoption하지 않는다. receiver 미결속/consumer 미완료면 KEEP; 전송·삭제0.

현재 문서는 source 준비 단계이다. 실제 ID/release는 별도 submission receipt에 기록한다. README는 GH 단독 통합. 대규모 raw 공유 불필요: `NO_BROADCAST_NOT_REQUIRED`.
