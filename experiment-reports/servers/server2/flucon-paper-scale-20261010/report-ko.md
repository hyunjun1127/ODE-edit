# SH2 FLU/CON 논문 표시 배율·완료 결과 검산

Nonce: `USER-GH-FLUCON-PAPER-SCALE-TABLE-REFRESH-20261010-R1`.
Accepted turn: `01a121bf-dc12-7420-9754-c50086b87646`.
정본 e99b61e3 전체와 main-results-policy를 읽고 전용 clean non-main WT에서 수행했다. 이후 GH 표시 API 15091336을 포함한 main f94bcfd8을 ff-only 채택했다. README는 GH 단독 소유이며 직접 수정하지 않았다.

## 표 입력 결과

| 관측 | 원 Flu (bits) | 원 Con (cosine) | Flu ×100 | Con ×100 |
|---|---:|---:|---:|---:|
| Qwen CF FT 61898, W20/2000 | 4.71017497777678 | 0.030072135827285053 | 471.02 | 3.01 |
| Qwen SH2 W0, producer 61898 | 6.252105796227186 | 0.2591242773267912 | 625.21 | 25.91 |

모두 반올림 전 2,000개 valid request 평균에 100을 곱한 뒤 Decimal half-up 두 자리다. 각 endpoint는 20,000 generation prompts이며 Flu/Con이 같은 생성 관측을 사용한다. Flu는 백분율이 아니고 Con도 정답률이 아니다. 원 JSON/W&B 값·키·history는 그대로다. 별도 raw_unit/raw_value/paper_display_x100/display_unit을 기록했다. 공통 `paper_cell`과 독립 Decimal 산술의 대조 4개, 공통 CPU tests 5개 PASS. 실제 pretrained generation 재현이나 온라인 readback PASS가 아니다.

단발 scheduler snapshot은 2026-10-10 02:43:21 KST이다. 기존 결과 reducer를 현재 raw에 다시 적용해 18개 현재 Qwen/GPTJ zsRE 행과 기존 완료 GPTJ CF 6행을 기록했다. 완료 수치 적격 13행(기존 GPTJ CF 6 포함), 나머지는 실제 RUNNING/PENDING으로 남겼다. Qwen BLUE CF 61962는 RUNNING이므로 W20 점수를 만들지 않았다. 수정 Qwen CF 62073/75/77/79는 DEFERRED이며 0 placeholder가 없다. GPTJ zsRE 61943/47은 PENDING이고 완료 61942/44/45/46만 public-query raw를 재집계했다. 오래된 완료 GPTJ CF의 scheduler snapshot은 기존 receipt를 재사용했다고 명시하고 최종 raw/terminal SHA와 2K 순서·수치는 이번에 다시 검산했다.

최종 GH 입력은 `audits/servers/server2/flucon-paper-scale-20261010/table-rows-final.json` 및 CSV다. 각 행에 job/name/source/config/ordered cohort/raw SHA/분모/관측시각이 있다. zsRE는 public-query token correct bits를 request macro로 재집계했다. 옛 W0 agreement를 Loc로 재명명하지 않았다. history/heldout/tuning/replica는 새 본표 결과로 포함하지 않았다.

## Qwen W0 교차 소유자 연결

SH3 selected W0 61813/source93341767은 factual 관측이며 생성은 NOT_MEASURED다. SH3의 후속 provenance 3a6c7ee9를 수신하고 실제 공개된 최신 파일과 SH2의 기존 full-SHA 검증 asset receipt를 직접 대조했다. 동일 model revision a09a3545, tokenizer/config 6개와 model shard 4개 bytes/fullSHA, CF stream66edc483 및 ordered first2000 SHA, cold W0가 일치한다. SH3 preparation/initial metadata도 읽기 전용 SSH로 확인했다.

SH2 W0 생성의 2,000개 원 observation 파일을 fullSHA 검증하고 각 record의 occurrence/case/generation_prompts/relation/target identity를 같은 CF stream에 대조했다. 원 generation profile `cf-cake-native-casebatch-kv-total100-globalrng-v1`, seed20261007, case batching/global RNG/topk5/total100/noEOS, reference identity75e595c7와 기존 세 reference SHA를 별도로 결속했다. source69bfbb2c/runtime/실행 receipt와 원 관측을 보존한다.

결론은 **동일 base-model/cohort W0 행의 생성 두 셀을 SH2 provenance로 별도 연결 가능**이다. SH3 factual 4셀은 변경하지 않는다. SH3가 생성했다고 표시하거나 SH3 seed20261002를 generation seed로 사용하지 않는다. cross-hardware/서로 다른 Python runtime의 bitwise 출력 동등성은 주장하지 않는다. 초기 불완전 provenance 상태의 보류 검토와 최종 연결 영수증을 구분하며, 최종 근거는 `w0-compatibility-final.json`이다.

## 보존·운영

이번 작업 GPU/model load/forward/job 제출·취소·dependency 변경/CP 삭제·이전/온라인 history 변경 0. 기존 context/hparams/raw/source/frozen jobs/CP는 KEEP. 모델 payload는 기존 실제 full-SHA receipt를 재사용했으며 대형 모델 재검산·복제는 하지 않았다. `NO_BROADCAST_NOT_REQUIRED`; 작은 source/숫자/hash만 Git 게시한다. 신규 monitor/heartbeat/자동 retry 없음. SH2→SH3 관련 turn 직접 요청은 accepted turn `01a121bf-d90c-7961-ac25-8e757c17d1a7`로 접수되었고 후속 provenance를 받았다. 최초 GH ACK transport는 timeout으로 완료 수신을 주장하지 않았으며 최종 게시 결과는 별도 직접 전달한다.
