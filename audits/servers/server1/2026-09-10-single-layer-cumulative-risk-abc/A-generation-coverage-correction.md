# A 보조 생성 평가 coverage 정정

설계 §8은 선택된 Middle Direct-C step8의60prompt greedy 생성도 요구한다.
A의 기존539개 구현 evidence 항목에는 이 endpoint가 누락되어 있었다.
따라서539/539는 **당시 나열된 검사**의 완료이지 원문 전체 coverage의 완전한 증명이 아니었다.
기존 A report/manifest/receipt 및 모든 raw bytes는 그대로 보존한다.

새 보충은 train-only 선택 alpha0.02의 기존 `snapshot-008.pt`를 그대로 읽는다.
같은20 Current request×3prompt,같은 tokenizer/model revision,greedy32token이다.
새 direct/native edit,학습,선택,z 계산은0이다. 이 관측은 A 선택/B 방향/C 실행 입력에 영향을 주지 않는다.
진행 중 C job43274는 변경하지 않는다. C의 예정 실험 이후 별도짧은 evaluation-only
job에서 보충하여 최종 requirements/비용/receipt에 결속한다.

추가 평가 전 최종 전체 완료 주장은 하지 않는다. 현재 A publication bytes를 고쳐
누락을 숨기지 않으며 final package에 supplement identity와60행/복원증거를 추가한다.
Scientific promotion=false.
