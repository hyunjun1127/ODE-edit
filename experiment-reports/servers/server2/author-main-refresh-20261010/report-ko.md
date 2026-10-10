# server2 완료 결과 / FE author 본표 통합 입력

Nonce USER-GH-S1-S2-FE-AUTHOR-MAIN-REFRESH-20261010-R1. Accepted turn `01a12655-1e13-7380-b3d7-83f4a30b1105`. 정본8d717f7b/envelope fullSHA 일치, registry SH2 session/CWD/origin/physical server2 확인. 동일 task의 이전 실행 receipt 없음.

한 차례 accounting snapshot: 2026-10-10T15:02:48Z (10월11일 00:02:48 KST). 숫자 적격24행(기존23+새 author CF1), 미완료2행. 저장된 원자료 CPU만 읽었으며 모델/CP 복원·GPU forward·평가 제출·job변경은 없다.

## Author 결과

| 데이터 | Job | 실제 상태 | Efficacy | Generalization | Loc | Score |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen CF | 62531 | COMPLETED / W20 원자료 검산 | 48.4 | 48.199999999999996 | 50.324999999999996 | 48.956504720767555 |
| Qwen zsRE | 62532 | RUNNING / 16 commit / W20 없음 | 미완료 | 미완료 | 미완료 | 미완료 |

62531: 20commit, history append20/layer, ordered2000, strict NLL requestmacro, R/P/N prompt분모2000/4000/20000. source a8c4c611a37210ddfa9e7db222dca6678d4cd678, config19ee6abf7a11f86a5a91a5af591f880967dfa2e3194c1ff32725fbe98a4ed90b, author profile1ffbf9ad….
W20 checkpoint8,535,429,071 bytes, fullSHA b3b66bbd2f9756f659448418d860d6986b933ee2791a4652e6f5d5dfd3d55dca를 실제 전체 파일 해시로 검산했다. CP 로드 없음. CF FLU/CON은 DEFERRED, future consumer pending. native FE62077의 수치나 generation을 author 결과에 이식하지 않는다.

Qwen/Llama 본표 label은 GH가 `MEMIT-FE (FE author hparams + history)`로 통합한다. 이 SH2 보고에서는 Qwen native FE를 legacy 별도 보존하며 GPTJ native FE는 author 실험이 없는 legacy로 명시한다. README 직접 변경 없음. SPHERE resume62538은 PENDING이며 아직 부모9+자식11의 완료를 주장하지 않는다.

## 새 generation 완료

| 모델/방법 | 평가 Job | Flu raw bits | Con raw cosine | Flu ×100 | Con ×100 |
| --- | --- | --- | --- | --- | --- |
| GPTJ FT | 62864 | 5.109818496507617 | 0.030430689364499153 | 510.98 | 3.04 |
| GPTJ MEMIT | 62865 | 5.779298936182971 | 0.36841812571964955 | 577.93 | 36.84 |
| GPTJ SPHERE | 62869 | 6.16176987217272 | 0.40894079247862974 | 616.18 | 40.89 |

각 endpoint2000/20000prompts, valid Flu/Con각2000. native reader로 case raw payload SHA/identity, 원순서, sampling stream, execution logical/physical work, 저장된 case 점수의 재집계와 summary 일치를 검산했다. 원 CP/config/source 및 비변이 receipt 결속. 이번엔 TF-IDF reference 재fit/생성 재실행/새 pretrained 재현 검증을 하지 않았다. 표시만 unrounded raw×100 후 half-up2, 원 raw/W&B는 불변. 나머지 generation 평가의 실제 RUNNING/PENDING은 JSON inventory에 분리한다.

## 검산과 보존

기존 적격23행은 raw/terminal SHA 불변 확인 후 CF strict reducer 및 zsRE 저장 predicted-target IDs→token correctness→request mean→2000 request mean을 다시 계산했다. zsRE11개 완료행은 frozen public-query evaluator/oracle sourceSHA와 기존 전체stream query proof 결속을 재확인했으며 tokenmicro/W0agreement를 Loc로 쓰지 않았다. CPU query 일치는 pretrained forward parity가 아니다. 원commit/CP/order 증거는 이전 승인 audit를 참조하고, 이번 새 author CP만 전체 payload를 재해시했다.

산출물은 `audits/servers/server2/author-main-refresh-20261010/{table-rows.json,table-rows.csv,inventory.json,review.py,finalize.py}`. raw/text/token/CP는 Git에 없음. 기존 cap2/dependency/job/CP/source/W&B 유지. 독립 reviewer 없음. NO_BROADCAST_NOT_REQUIRED. 장기 monitor/자동retry 없음.
