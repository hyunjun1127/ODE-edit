# 조건부 최종 CF/zsRE adapter

사용자 nonce: USER-FE-QWEN-WORKING-CURRENT-Z-20261012-R1.
SH1은 소스/CPU 구현만 수행했다. Rent 실행/연속 관측은 requester/root 소유,
SH2 HOLD 유지. SH1 신규 제출·GPU 모델 계산·기존 실행 변경은 0이다.

## 구현

- 기존 selection-r3 frozen source/attempt는 수정하지 않았다. 새 준비 경로에만
  최종 benchmark source를 동결한다.
- `prepare`의 `stage=final_benchmark`는 네 held-out arm의 기존 raw/명시적
  수치실패를 독립 CPU 재검산하고 저장 selection-result와 정확 비교한다.
  SELECTED 및 L2>0 없이는 두 benchmark config도 만들지 않는다.
- 동일 선택 L2로 CF/zsRE 각각 원래 first2000 독립 cold model을 로드한다.
  writer/target supplier/firstforward/history/BF16/저자 기본 backend는 불변이다.
  기준 L2=0 승격, W0-z 대체, zsRE 별도 tuning, 편집 모델 재사용은 없다.
- zsRE native edit request는 원본 prompt/subject/target를 한 번씩 전달한다.
  공개 query의 decode/re-tokenize와 loc_ans 확장은 evaluator에서만 한다.
- W0 2000, current100 매 batch, allseen 500/1000/1500/2000 raw와 query를 저장한다.
  CF strict NLL 및 R/P/N 기존 계약, zsRE public-query E/G/loc_ans request macro를 유지한다.
- CPU reducer는 예측/목표 ID, query별 위치/순서, correctness bits, 요청별 평균과
  token 분모를 독립 대조한다. 최종 collector는 25개 endpoint와 20 commit/W0,
  최종 checkpoint SHA 및 양 dataset 동일 선택값을 확인하고 zsRE W0→W20 Loc 차이를 낸다.
- latest-only 저장은 평가/raw 저장 뒤 동일 host lock을 사용한다. 이미 생성된
  held-out latest는 exact config/allocated bytes에 한해 남은 용량 계산에서 공제한다.
  reserve/atomic tmp/raw margin은 줄이지 않는다.
- W&B final group/profile과 held-out group/profile을 분리한다. current/allseen 축은
  기존 공통 키이다. CF FLU/CON은 DEFERRED, zsRE generation 필드는 없다.

## CPU 검산과 한계

75 tests PASS: 기존 native writer AST/numeric/비변이/스토리지/transport 회귀 67개와
최종 adapter 8개. 후자는 실제 pinned author load_data AST, 작은 Qwen형 synthetic
public-query forward, 실제 current100 observer→raw→independent reducer→scalar schema,
변조 거절 및 선택 gate를 포함한다. 별도 pretrained GPU 검증은 수행하지 않았다.

기존 Qwen tokenizer와 원래 zsRE 2000건을 전량 CPU 대조했다.
24,858 queries, R/P/N token 분모 6,691/6,691/11,476, input/target mismatch 0.
이는 입력 정합이며 pretrained 출력 일치 또는 회복 성공 주장이 아니다.
정확 원본 CF/zsRE 경로·bytes/SHA는 `final-streams.json`에 게시했다.

최종 설정/실행 성공 여부는 아직 관측하지 않았다. 선택이 NO_PROMOTION이면
이 adapter는 실행을 허용하지 않는다. 양 W20이 모두 완료되어도 collector는
자동으로 working/recovery 성공을 선언하지 않고 실제 수치와 궤적을 반환한다.
README/기존 raw/CP/W&B history는 변경하지 않았다.
NO_BROADCAST_NOT_REQUIRED: 소형 소스/manifest만 Git 게시, 모델/CP/raw 전송 없음.

## API

`official/diagnostics/fe_qwen_current/README.md`의 Conditional final 절을 따른다.
`prepare NEW_FINAL_ROOT final-bindings.json` → exactly two configs;
동결 source의 `launch CONFIG LOCK` → root가 승인된 rent 자원 안에서 실행;
`benchmark NEW_FINAL_ROOT` → 두 W20의 `final-result.json` 독립 CPU 검산.
