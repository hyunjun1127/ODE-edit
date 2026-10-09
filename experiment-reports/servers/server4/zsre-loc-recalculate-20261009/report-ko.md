# zsRE Loc 정책 정정 및 server4 완료 inventory

Nonce `USER-GH-ZSRE-LOC-RECALCULATE-MAIN-20261009-R1` 수락. 새 정본 `a8d8db88`의 envelope를 전체 읽었다.

새 Loc은 `100 × mean_requests(mean_loc_ans_tokens(predicted == target))`이다. token-micro로 대체하지 않으며 W0 prediction agreement는 별도 보조지표로 보존한다. 완료 raw가 있을 때 Eff/Gen도 같은 raw의 request-macro로 독립 재집계하고 저장 Specificity_loc_ans는 교차검산에만 사용한다.

## 이번 실제 결과: 해당 없음

server4 등록 Qwen zsRE FT/MEMIT/AlphaEdit/BLUE/MEMIT_FE/SPHERE job61755/61757/61759/61761/61763/61765는 모두 이전 지시에 의해 미실행 CANCELLED, elapsed0이다. archive61756/58/60/62/64/66도 취소 이력이다. 해당 run output에 raw가 없고 W20/2,000 edits가 관측되지 않았다. source/config/cohort SHA와 exact accounting은 inventory에 결속했다.

server4 own audit/report의 zsRE 참조를 검색했다. 기존 CAKE·BLUE source inventory의 `zsre.py/eval_utils_zsre.py` 등은 포함된 upstream 코드일 뿐 완료 실험 근거가 아니다. 2026-08-01~2026-10-10의 user janghj/server4 할당 accounting에서도 이름에 zsRE를 가진 실행은 발견되지 않았다. 미할당 취소 job은 node필터에 누락될 수 있으므로 위 exact12개 IDs accounting을 별도로 확인했다. own official Llama 준비 namespace는 준비/공유source 자료이며 게시된 zsRE 실행 영수증이 발견되지 않았다.

따라서 확인한 승인 완료 zsRE baseline0/ours0, 재집계0, README 수치 갱신 대상0이다. before W0agreement/after Loc/Eff/Gen/평가 case·token분모/rawSHA는 **NOT_AVAILABLE/null**로 기록하며 0을 발명하지 않는다. 실제 적용할 raw가 없어 metric reducer SHA도 해당 없음이며, 대신 inventory script SHA를 기록했다. 완전한 filesystem의 미등록 실험까지 확인했다고 주장하지 않는다.

`audits/servers/server4/zsre-loc-recalculate-20261009/inventory.json`과 `rows.csv`가 compact evidence다. GH만 common code/README를 수정한다. 이후 새 freeze에서 GH READY source를 채택하며 기존 frozen job hotpatch는 없다. CF 수치·FLUCON 결측/DEFERRED·원 raw/CP/job 및 SH2 Qwen 이전 진행은 변경0. 추가 GPU/forward/복원/재실험/취소/온라인 과거기록 수정0. tokenizer 완전동등/paper reproduction PASS 주장은 없다.
