# zsRE 저장 W20 최종 2K 재평가

사용자 권한 `USER-GH-ZSRE-SAVED-WEIGHTS-2K-REEVAL-20261009-R1`에 따라
GH가 공통 native-query 평가기·CPU oracle·eval-only tracking을 직접 구현했다.
기존 frozen 실행, 가중치, raw는 변경하지 않는다. 재편집/W0/중간 endpoint/CF/FLUCON은 제외한다.

## 공통 입력 검산

- 실제 고정 2K × Llama/GPT-J/Qwen tokenizer files fullSHA 검산.
- 봉인된 공개 loader/evaluator AST의 prompt/target, 입력 ID, 채점 target을 전부 비교하여 불일치 0.
- Llama E/G/Loc token checks 6035/6035/12465, GPT-J 5557/5557/9694,
  Qwen 6691/6691/11476. 각 요청별 평균 후 2000요청 평균이며 token micro가 아니다.
- Llama Loc loader는 native BOS를 포함한다. 기존 10465개의 저장 Loc correctness와는
  관측 자체가 달라, CPU 재평균으로 새 점수라고 쓸 수 없다.
- query parity는 GPU 출력 parity가 아니다. 새 최종 점수는 실제 복원 가중치의 평가 후에만 게시한다.
- CPU 49 tests PASS(새 평가기·eval-only schema/실제 worker→fakeSDK·기존 tracking/Loc).
  실모델 forward/GPU/online은 GH 검산에서 0.

근거: `audits/global/zsre-2k-reeval-20261009/query-parity.json`, 동일 폴더
`query_parity.py`; API `official/evaluation/ZSRE_PAPER.md`.

## 서버 위임 상태

SH1 Llama 완료 6행, SH2 GPT-J 완료 6행을 로컬 최종 checkpoint로 재평가하도록 직접 배정했다.
SH2는 6개 CP fullSHA를 확인했다. SH3는 한정 local inventory에서 완료 zsRE CP가 없어
NOT_APPLICABLE/nojob 보고. SH4의 own inventory 결과 및 SH1/2 실제 제출은 별도 영수증으로 갱신한다.
대형 가중치 전송/삭제, 미완료 Qwen의 재편집, replica 중복 평가는 없다.

이 문서의 현재 상태는 공통 코드/CPU 검산 및 실행 배정이며, 새 12행 성능 완료를 뜻하지 않는다.
기존 README zsRE 수치는 old evaluator 관측임을 유지하고 실제 재평가 job/status 및 완료 수치를
원본 checkpoint/evaluator/stream SHA 링크와 함께 갱신한다. CF 행은 이 작업으로 수정하지 않는다.
