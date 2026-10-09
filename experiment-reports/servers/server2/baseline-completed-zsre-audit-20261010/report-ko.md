# SH2 완료 baseline·zsRE 지표 감사

Nonce `USER-GH-ALL-SH-BASELINE-COMPLETED-ZSRE-AUDIT-20261010-R1`; accepted turn `01a122d0-699d-77a2-9764-77789dcbfa24`. 정본 main7dacd3fb 전체 및 공개 zsRE API 문서를 읽고 전용 non-main WT에서 실행했다. README는 GH 단독 통합이며 SH2는 수정하지 않았다.

## 완료 및 현재 상태

단발 own scheduler snapshot: **2026-10-10 07:38:08 KST**. 총 24 baseline 행 중 **18행 수치 적격**, 6행은 미완료/실패다. 기존 GPTJ CF 6행은 최근 CPU 검산을 raw/terminal SHA 불변 재확인 후 재사용했고 scheduler 관측시각은 기존값이라고 명시했다. 나머지 18행은 이번 단발 scheduler 결과다. 본표 입력 `audited-final.json`/CSV에 exact job name/source/config/cohort/raw/terminal/분모를 기록했다.

새 완료 주요 CF 값(2,000 requests, 20 commits):

| 모델/방법 | job | Eff | Gen | Loc | Score | Flu ×100 | Con ×100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen BLUE | 61962 | 97.70 | 96.75 | 55.86 | 77.97629788389074 | 602.99 | 37.43 |
| Qwen MEMIT, 수정 cold | 62073 | 68.50 | 65.825 | 54.295 | 62.230026985355586 | DEFERRED | DEFERRED |
| Qwen MEMIT-FE, 수정 cold | 62077 | 51.05 | 51.225 | 49.53 | 50.59009938622918 | DEFERRED | DEFERRED |
| Qwen SPHERE, 수정 cold | 62079 | 99.40 | 97.70 | 64.37 | 83.72646652326851 | DEFERRED | DEFERRED |

Qwen BLUE 원 Flu=6.029855922995685 bits, Con=0.37431408204466576 cosine; valid2000/prompt20000. 원 raw/W&B를 변경하지 않고 공통 paper_cell로 마지막 표시만 ×100 half-up 두 자리 변환했다. CF raw Score와 별도 원 AlphaEdit display Score는 동일하다고 강제하지 않는다. 새 CF 원 factual raw의 NLL strict preference/요청별 macro와 20 commit·최종 checkpoint receipt를 검산했다. CP tensor는 읽지 않았다.

현재 GPTJ zsRE SPHERE61947/Qwen CF Alpha62075/Qwen zsRE Alpha62083은 PENDING, Qwen zsRE MEMIT62081/FE62085는 RUNNING, Qwen zsRE SPHERE62087은 FAILED다. FAILED에 W20 점수를 생성하지 않았다. 이 요청은 수리 권한이 아니므로 job/source를 변경하지 않았다.

## zsRE 완료 7행

| 모델/방법 | 평가 job → 원 편집 job | Eff | Gen | Loc (loc_ans) |
|---|---|---:|---:|---:|
| GPTJ FT | 61942 → 61726 | 23.146349206349203 | 17.9478373015873 | 0.6249613700323515 |
| GPTJ MEMIT | 61943 → 61728 | 93.51850732600732 | 88.86412545787546 | 30.80990243838416 |
| GPTJ AlphaEdit | 61944 → 61730 | 99.6918315018315 | 96.45428113553113 | 27.9417010245744 |
| GPTJ BLUE | 61945 → 61732 | 99.75147435897436 | 95.708894993895 | 28.834338825411812 |
| GPTJ MEMIT-FE | 61946 → 61734 | 29.1998778998779 | 27.648009768009768 | 8.449205017109922 |
| Qwen FT | 62072 → 61900 | 23.246031746031747 | 18.544821428571428 | 2.3364180206369682 |
| Qwen BLUE | 61964 자체 최신 평가 | 58.616865079365084 | 53.6624007936508 | 5.727470490106537 |

모두 predicted_token_id==target_token_id를 원 correctness bits와 대조하고, **각 요청의 token 평균 → 2,000 요청 평균 ×100**으로 독립 재집계했다. Loc도 실제 loc_ans 정답 token에 같은 방식을 적용한다. token-micro는 차이 확인용 진단으로 별도 기록하며 표에 사용하지 않았다. W0 prediction agreement/strict prompt accuracy는 Loc 대체값이 아니다. 결측 요청0, finite/정확 순서/summary 일치 확인. GPTJ token 분모 R/P/N=5557/5557/9694, Qwen=6691/6691/11476.

실제 frozen zsre_paper.py SHA `d6a5b34eafd27660a2dee4632c638b6bf4c3614246071711cf5159a002415a45`, query-parity source 및 public-source lock의 frozen bytes를 확인했다. 원 공개 loader/evaluator 자료 3개씩, 세 source root의 SHA/bytes도 원 lock과 일치했다. 전체 2000 query proof는 입력·target mismatch0이며 실제 raw/commit의 query SHA와 일치한다. GPTJ querySHA `2d27e4fc4445e709586e2d80f92dee76dfc1b6ce6cd7d333b6ca318ccb328a4b`, Qwen `6c8111384789cec7059ea1b3c2b1083d761ade2b360c99db3e5c4f1ef4941d78`이다.

stream 파일 SHA와 정규 JSON digest는 다른 계약이다. oracle_lock_sha256도 파일 SHA가 아니라 canonical JSON digest이므로 둘을 분리 검산했다. Qwen FT62072의 evaluator/tokenizer/profile은 원 raw identity 외에 실제 config.json/config_sha 및 frozen caller로 결속되어 있다. 새 source나 identity로 원 raw를 relabel하지 않았다. 이 차이를 반영하기 전 audit 도구의 중간 오류는 원 실험의 결함이 아니며, 수정한 최종 감사에는 불일치0이다.

원 공개 query의 teacher-forced prefix/공백/BOS/decode-retokenize는 기존 전체-stream CPU parity receipt를 재사용한다. **실제 pretrained forward의 독립 원본 수치 parity PASS를 뜻하지 않는다.** Qwen은 공개 non-Llama 분기의 적용이며 논문 Qwen 재현을 주장하지 않는다. eval-only 6개는 원편집 chain의 평가 결과로 연결하며 새 편집 실험으로 중복 계상하지 않는다. 옛 GPTJ token-ID 연결 결과와 Qwen FT61900 원 평가는 `PUBLIC_QUERY_REEVALUATION_REQUIRED` 역사로 분리하고 대응 완료 eval-only를 표에 사용한다. broken-context61956/60/68은 철회된 역사로 유지한다.

## 보존

GPU/모델 forward·복원/새 평가/CP load/제출·취소·hold·dependency 변경/삭제·전송/온라인 history 변경0. 원 raw/CP/source는 KEEP. OURS/PRICE/tuning/heldout/historical/이관 replica를 본표로 승격하지 않았다. `NO_BROADCAST_NOT_REQUIRED`; compact source·수치·hash만 게시했다. 반복 monitor/자동 retry/실험 완료 대기 없음.
