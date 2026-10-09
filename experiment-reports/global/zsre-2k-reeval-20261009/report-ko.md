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
NOT_APPLICABLE/nojob 보고. SH4도 own Qwen zsRE 미실행 취소 6개 경로에서 최종 CP가 없음을
확인하여 NOT_APPLICABLE/nojob이다. SH1/2 실제 제출은 별도 영수증으로 갱신한다.
대형 가중치 전송/삭제, 미완료 Qwen의 재편집, replica 중복 평가는 없다.

새 12행 성능 완료를 뜻하지 않는다. 이전 관측은 원 보고서/raw에 보존하며 README zsRE 12행/36칸을
실제 재평가 job/status로 갱신했다. CF/W0/다른 모델 행의 기존 값은 변경하지 않았다.

공통 구현 `f1a00379`, main 통합 `4533756e`, official tree
`eb116d1b772ea0b5697d37f9720b999431317638`를 네 서버에 직접 READY로 전달했다.
source166 SHA/Python315/외부 task import0 검증 통과. 원문 authority 전달 accepted turn:
SH1 `01a11fc8-ca64-7ae1-bd5a-47f0babedb60`, SH2 `01a11fce-aec4-7801-8e87-ea70b4b3d2ed`,
SH3 `01a11fce-ae19-7f90-8ae0-e2faa74471a5`, SH4 `01a11fce-ae91-7bc1-a254-c347c6ee37fa`.
직접 담당 응답 및 own inventory 게시로 최초 수신은 확인했으며 실제 제출과 구분한다.

## 실제 등록 및 표 반영

두 담당의 실제 immutable 등록/release 영수증을 읽어 검산했다. 추가 scheduler polling은 하지 않았다.
모든 12개 GPU job과 두 GPU0 collector는 held 검사를 거쳐 release됐고, 단발 초기 관측은
PENDING(Dependency)이다. 실제 평가/online startup은 미관측이며 새 성능 수치는 0행이다.

| method | server1 Llama 평가 | server2 GPT-J 평가 |
| :--- | ---: | ---: |
| FT | 61932 | 61942 |
| MEMIT | 61933 | 61943 |
| AlphaEdit | 61934 | 61944 |
| AlphaEdit-BLUE | 61935 | 61945 |
| MEMIT-FE | 61936 | 61946 |
| AlphaEdit+SPHERE | 61937 | 61947 |
| GPU0 collector | 61938 | 61948 |

Llama 실행 source `b2806a6081d66fc585db970595c4b86a71f5aafd`; GPT-J 실행 source
`ce8d536fa7eb4dfdd38f7024381e68d212f44ea0`. 양 source의 공통 evaluator/query/tracking SHA는 같다.
각 서버 최신 cap4와 기존 admitted lane을 보존했다. SH1 resource frontier는
61773/61927/61928/61929이며, FE 평가는61934 이후, SPHERE는61935 이후다.
SH2 첫 네 평가는 기존 Qwen lane61918/61920/61914/61916 이후이고,
FE/SPHERE는 각각61942/61943 이후다. 모든 자원 의존성은 afterany이며 기존 job 변경은 없다.

완료 W20 CP는 두 모델 각각6개 fullSHA 확인, replica 중복0. 서버3/4는 own완료CP없음으로
NOT_APPLICABLE/nojob. 확인된 완료 후보12개 중 missing CP0; 미완료 Qwen을 신규 편집하거나
평가하지 않는다. checkpoint별 exact path/bytes/SHA, 원 job/source/config 및 새 평가 identity는
`audits/global/zsre-2k-reeval-20261009/integration.json`에 결속했다.

SH2 준비 JSON에서 nanosecond timestamp가 전달 중 반올림된 문제는 원 CP fullSHA 재확인과
문자열 timestamp로 수리했다. checkpoint bytes는 변경되지 않았다. SH1 전체24535 query와
SH2 전체20808 query의 별도 CPU oracle 검산도 통과했다. 공통+양 restore CPU 58개 통과는
실모델 평가 성공을 뜻하지 않는다. 단독 GPU qualification/추가 fit은 없다.

README 검산 `check_table.py`: 12행/36칸, 실제job ID 중복0, 새 점수0 PASS.
실제 완료 수치는 후속 eval-only 원자료가 생겼을 때만 갱신하며, 현재 수치를 예측하거나 0으로 채우지 않는다.
