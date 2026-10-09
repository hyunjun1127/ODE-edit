# SH1 baseline 완료 및 zsRE 산출 검산

Nonce `USER-GH-ALL-SH-BASELINE-COMPLETED-ZSRE-AUDIT-20261010-R1`.
정본 `7dacd3fb7a40847da0f136de88a307ff18c2616f` 전체를 읽고 전용 non-main WT에서 수행했다. 실제 root `/mnt/raid5/janghj/ODE-edit`, origin `hyunjun1127/ODE-edit`, server1 등록 session `01a04939-f93a-7b50-bca0-65438eab2062`를 대조했다. 다른 작업/root dirty/frozen runtime는 변경하지 않았다.

## 결과

완료된 적격 관측 **12행**: Llama CF FT/SPHERE/MEMIT-FE 3개, GPT-J/Llama/Qwen MEMIT-FE-HISTORY 3개, Llama 공개-query zsRE 6개. history는 native MEMIT-FE와 별도 variant다. 이전과 동일한 11행의 raw/terminal/config SHA를 확인하고 기존 검산을 재사용했다. zsRE6은 아래와 같이 독립 재집계도 수행했다.

Qwen history **62061**은 수정 mask source의 새 완료다. 원 frozen source 전체 SHA, ordered stream, 각100요청×20 commit, 요청 digest, history append once/층/배치, native state 연속성, W20 checkpoint fullSHA/receipt를 확인했다. 원 evaluator의 CPU token/NLL/count audit 통과. 과거 잘못된 context **61975는 제외**했다.

- Qwen history raw Efficacy `48.449999999999996`, Generalization `48.175000000000004`, Specificity `51.824999999999996`, Score `49.42893436361672`, requests2000.
- CF strict NLL preference 및 Score 정의는 변경하지 않았다. Flu/Con은 미측정 DEFERRED다. 모델 forward나 checkpoint 역직렬화/복원은 하지 않았다.

## zsRE 공개-query 검산

실제 실행 source의 `zsre_paper.py`, `zsre_query_parity.py`, public AST lock과 tokenizer 파일 SHA/bytes를 확인했다. 같은 right-pad/eos tokenizer와 전체 미확장 2K stream에서 CPU `compare_queries`를 재실행하고 기존 사전봉인 proof와 정확히 일치함을 확인했다. native prompt/target, prefix/공백/BOS/decode-retokenize 결과를 공개 loader/evaluator AST와 대조한다. 이 CPU 일치는 pretrained forward numeric parity 주장이 아니다.

입력/target mismatch **0**, 요청2000, query24535, E/G 각6035 token, loc_ans12465 token. 모든 평가 raw의 query SHA와 같은 proof/stream/tokenizer/config/source/checkpoint identity를 결속했다.

| 평가 job | 원 method | Eff | Gen | Loc |
|---:|---|---:|---:|---:|
| 61932 | FT | 14.632400793650794 | 11.816567460317462 | 25.251387670596298 |
| 61933 | MEMIT | 44.30071428571429 | 39.94732142857143 | 22.30491261977676 |
| 61934 | AlphaEdit | 95.18200396825397 | 91.4040873015873 | 31.098955955537328 |
| 61935 | AlphaEdit-BLUE | 95.8722619047619 | 92.28218253968254 | 32.828486570334576 |
| 61936 | MEMIT-FE | 14.805357142857142 | 13.574107142857141 | 0.5551871722320902 |
| 61937 | SPHERE | 95.12626984126985 | 91.36148809523809 | 31.381720284338897 |

각 raw의 실제 predicted/target ID equality와 저장 correctness bits를 대조했다. 요청 내부 teacher-forced token accuracy를 평균한 뒤 2000요청 간 평균×100으로 독립 집계했다. Loc은 loc_ans 정답 기준이며 W0 agreement/token micro/strict prompt 정확도가 아니다. 결측 요청0, finite와 token분모 정확일치. 기존 보고와 delta0, README의 소수2자리 18개 셀도 일치한다. 이전 token-ID 연결 평가 raw는 `PUBLIC_QUERY_REEVALUATION_REQUIRED_SUPERSEDED`로 분리하고 원 편집 job→평가 job 연결을 유지하여 중복 editing chain으로 세지 않는다.

독립 fsum과 원 summary의 연산순서 차이는 최대 약 `7.1e-15`pp였다. 기존 reducer의 `1e-10`pp 비교 기준은 변경하지 않았다. 저장 Specificity와 Specificity_loc_ans alias 자체는 bitwise 동일하며 독립 reducer의 bitwise 동일성을 주장하지 않는다.

## 단발 scheduler/미완료

대상18개 ID를 한 번의 `sacct -X`로 읽었다. 완료 raw를 검산한 12개 모두 owner janghj/정확 jobname/해당 source WorkDir/COMPLETED 0:0를 확인했다. scheduler 완료만으로 수치를 확정하지 않았다.

Flu/Con **62259/62260/62261 RUNNING**, **62262/62263 PENDING**, CPUcollector **62264 PENDING** (단발 snapshot 2026-10-10 약07:37:58 KST). 각 평가 COMPLETE receipt가 없으므로 partial generation을 최종 평균으로 보고하지 않는다. 반복 조회/GPU완료 대기는 하지 않았다. 기존 Llama W0의 표시 배율 검산과 historical 예외는 유지하며 새 숫자 계산을 꾸미지 않았다.

## 산출물/보존

`audits/servers/server1/baseline-completed-zsre-audit-20261010/`의 `table-rows.json/csv`, `query-proof.json`, `qwen-history.json`, CPU reducer가 정확 source/config/raw/cohort/CP/분모/상태를 담는다. 텍스트/token/caseID/CP/tensor는 Git에 넣지 않는다. `NO_BROADCAST_NOT_REQUIRED`: 원자료는 기존 local에 KEEP, compact report만 Git 공유.

새 GPU/fit/forward/CP load/재평가/제출/취소/hold/dependency 변경/삭제/온라인 history 변경은 모두0. 본 task는 CPU 검산이며 README는 GH 단독 통합한다. 다른 승인 작업은 유지한다.

결과 main publication `39f93892812be5d580b307e55ead863f7df694f7` / own commit `f5b164da`. GH direct 전달은 app-server 응답 timeout으로 COMMUNICATION_HOLD, 수신 UNKNOWN이다. 재전송하지 않았고 Git 게시를 수신 ACK/README 통합 완료로 표현하지 않는다. 정확 전달 receipt는 `delivery.json`에 기록했다.
