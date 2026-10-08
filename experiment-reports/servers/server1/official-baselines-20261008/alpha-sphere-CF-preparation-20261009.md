# Llama3 CF AlphaEdit / SPHERE 추가 준비

사용자 직접 지시 `alphaedit 과 sphere도 run올리자`에 따라 두 CF chain만 추가한다. Official existing-file first2000, BS100×20, server1 cap4, W20 checkpoint 보존 및 FLU/CON 후속 평가 정책을 적용한다. 기존 61657–61664와 frozen source는 변경하지 않는다.

## 구현

- `official.baselines.registry`의 ALPHAEDIT/SPHERE native apply·parser·request·call options를 사용한다. 두 방법 모두 원 layers4–8/L2=1을 유지하며 PRICE/GPT2 계수로 대체하지 않는다.
- 기존 projector full SHA `6d356468c6408dca694c1907ffde31cddb99f7e67fa2e74502910d5afbede5ec`, 4,110,419,877 bytes, FP32 `[5,14336,14336]`, physical layers4–8/.02를 CPU로 결속했다. 재생성/다운로드하지 않았다.
- native module-global cache_c를 cold zero로 초기화하고 매 native apply 후 실제 누적 history를 checkpoint/restore/연속성 감사에 연결했다. apply 반환 weights_copy를 history로 오인하지 않는다.
- 각 chain의 factual cold W0와 기존 factual 중간 평가는 유지한다. 공통 generation W0 READY는 요구하지 않으며 W0/W20 모두 신규 FLU/CON 생성·점수·progress를 기록하지 않는다.
- 두 qualification의 native B3 및 B2→B3 resume 비교/독립 CF scorer 실제 검증 뒤 두 chain이 시작한다. GPU 자원 의존성은 현재 소유 GPU 말단 전체 뒤로 연결하며 collector는 자기 네 GPU jobs afterany다.

## 검증과 한계

Focused CPU 97 tests PASS 및 최종 server1 전체 CPU 187 tests PASS (49.596s); source verifier 157 SHA / Python262 / external task imports0 PASS. 실제 pretrained GPU qualification, 온라인 W&B 및 과학 완료는 이 준비 보고의 PASS 대상이 아니다. 이번 검토는 owner CPU 검토이며 별도 independent reviewer는 사용하지 않았다. 실제 등록 영수증은 별도 기록한다.

## zsRE 공통 입력 채택

`GH-SH1-ZSRE-WANDB-METRICS-READY-20261009-R1` 수락. 준비 WT는 `a0408faada5d493f562331609b9d19d987bee3f4`를 포함한 main `ccc1f5d6`까지 안전 ff 동기화했다. 기존 자체 수정은 보존했다.

`official.tracking.official_zsre_metrics(summary, config_values=..., endpoint=..., edits=..., pre_state_edits=..., post_state_edits=...)`를 own Tracking caller에 읽기 전용 연결했다. 이미 측정한 official summary에서만 zsre 별칭을 만들며 request macro/Specificity W0 agreement/loc_ans 구분과 축을 유지한다. 새 forward·공통 logger 복제·shared 수학 변경은 없다. CPU fake transport에서 두 namespace 일치를 검사했다. 이번 입력으로 zsRE job을 제출하지 않았다. 기존 frozen CF/zsRE source에는 적용하지 않는다.

모델/자료/CP/raw는 local-only KEEP. 소형 source/report만 Git 공유하며 same-host 자산은 `NO_BROADCAST_NOT_REQUIRED`다.
