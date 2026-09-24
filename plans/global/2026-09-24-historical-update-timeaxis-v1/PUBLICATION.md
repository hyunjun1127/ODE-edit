# Historical update timeaxis — motivation·설계 게시 안내

2026-09-24 사용자 요청으로 motivation과 설계 원본을 main에 게시한다.
현재 연구 질문과 E3 근거를 읽으려면 다음 순서로 확인한다.

1. [Motivation 재검토와 완료 E3의 가치·한계 (§1.1)](../../../audits/global/2026-09-24-write-function-timeaxis-reassessment/review-ko.md).
2. [설계 개요](README.md), [상세 설계](design-ko.md), [구현·수치 계약](runner-contract-ko.md).
3. [고정 계약](experiment-contract.json), [DAG](execution-dag.json), [설계 manifest](design-manifest.json).
4. [완료 E3 사실 보고](../../../experiment-reports/servers/server4/native-delayed-write-e3-20260924-v1/completed-review-r1/report-ko.md).

## 이번 게시가 바꾸는 것과 바꾸지 않는 것

원 설계 29파일(합계5,238,704B)을 SHA/size 그대로 게시한다. 이 파일들의
“모델 실험 미실행”, “runner 미구현”, “원격 CP 현재 미결속” 문구는 **설계 작성
당시 상태**다. 현재 실행 상태 주장으로 읽지 않는다. 이 안내는 원 설계 manifest의
멤버가 아닌 별도 publication 문서이며, 실행 lock·input archive를 갱신하지 않는다.

E3 기반 동기는 별도 review §1.1에 추가한다. whole-five-weight U, M/B/C 정의,
156셀·후속16셀, 표본·threshold·수치 기준·DAG는 변경하지 않는다.
E3를 현재 기여 변화의 증명 또는 선행 과학 gate로 삼지 않는다.

현재 구현은 원격 branch `codex/server4-historical-update-timeaxis-20260924-v1`의
수리 execution `730a4a9768e5650e01fd9afdc4e0f7895c86ea92`에 있고,
이번 게시에는 그 실행 코드를 main으로 병합하지 않는다.
현재 사용자 지시는 server4 기존53182/53183/53184를 그대로 유지하고
SH1 이관을 철회하는 것이다. SH1/SH4 모두 취소·새 제출·전송 미수행을 ACK했다.
기존 [SH1 이관 지시](../../../messages/head/2026-09-24-historical-update-timeaxis-migrate-sh1.md)는
최신 정정으로 실행하지 않는다. 이번 문서 게시가 monitoring 재개·실험 변경을
승인하지 않으며 현재 job의 진행/완료는 조회하지 않았다.

## 포함·제외

- 포함: motivation/문헌 노트/측정안, 계약·DAG·schema, 사전 고정 cell/state 표,
  ID·hash·충돌 flag만 있는 ledger/panel, source-evidence와 검증·재생성 코드.
- 제외: 모델·checkpoint·원 dataset 문장·raw 결과·로그·논문 PDF/전문 추출/대조 이미지.
- checkpoint CSV는 경로·hash 목록이지 checkpoint payload가 아니다.
- 원 설계 검증 결과는 설계 당시 CPU consistency 기록이지 이번 실제 GPU PASS가 아니다.

`build_plan.py`와 `validate_plan.py`는 출력·manifest를 덮어쓰므로 봉인 폴더에서
publication 검증용으로 실행하지 않는다. 이번에는 원 manifest의 bytes/SHA를
read-only 재검산하며, E3 근거 수치는 게시된 CSV와 대조한다.
