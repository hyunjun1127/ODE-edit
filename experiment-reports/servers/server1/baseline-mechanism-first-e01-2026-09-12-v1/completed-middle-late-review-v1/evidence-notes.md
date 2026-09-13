# E01 완료 관측의 계측·복원·비용 사실

- Middle B060/Late B100은 native 10-batch endpoint를 재사용하며 관측 보완 native/z/history count 모두 0이다.
- B060/B100 실제/원본 checkpoint를 CPU mmap 및 FP64 chunk로 비교했다. 파일 SHA를 접근 전후 모두 검산했다. 상대 Frobenius 분모는 원본 tensor norm이다. W와 M은 모두 nonexact이며 원인 UNRESOLVED다. 성능 유사성을 weight parity로 승격하지 않는다.
- B001/B020 tensor 비교 및 E1-A 10-arm CPU join은 기존 sealed receipt를 재사용하고 전수 재감사하지 않았다.
- L4 네 entry만 native/E1 첫 batch가 존재한다. 4/20 cell, 나머지 L5–L8 16 cell은 NOT_RUN. Cold 1 + warm 세 window 각10 = canonical valid unique native31 batch이며 취소/실패 비용을 지우지 않는다.
- 신규 B051/B091 계측은 target->write->query->signed output 자료다. B060/B100 terminal signed/General 관측으로 바꾸어 읽지 않는다. General은 Wikipedia128의 별도 per-sequence NLL이며 canonical RS/PS/NS가 아니다.
- Signed panel은 Current/Historical 160 backward pairs 및 General16이다. 각 window 대표 FD는 별도 1 backward/5 diagnostic forwards. 패널 전체 FD 검증으로 부풀리지 않는다. query prefix에는 SUBJECT_UNRESOLVED가 남으며 subject span을 추정하지 않는다.
- native compute_z 호출100/batch는 기록되나 최종 trainingNLL/iteration/stop/clamp는 NOT_OBSERVED다. Projected C0/history exact spectrum·rank·condition은 NOT_RECORDED. native nonsymmetric-system symmetry error는 condition number가 아니다.
- 관측 restore는 pointer/bytes/RNG 기록이며 parameter version을 원복했다는 주장은 없다. 관측 패널의 temporary-state version0와 native transaction changed_version1을 구분한다.
- compute-summary의 ALLOCATION_TOTAL 행만 해당 job 총 GPU seconds다. program elapsed는 allocation 내부, native subcomponents는 native total 내부다. 이를 더하지 않는다. 계측 wall은 host wall이며 CUDA kernel/FLOPs 측정이 아니다.
- 최초128 gate elapsed는 fullseen 전체 비용이 아니다. Whole E01 완료 또는 원 trajectory 동등성은 미검증 상태다. 원인 종합은 GH 소유다.

## 실제 첫-batch signed / General 요약

R/P는 true-new NLL margin, N은 new-true NLL margin의 signed derivative이며 양수가 해당 preference 개선 방향이다. General은 NLL derivative여서 음수가 낮은 NLL 방향이다. 단일 entry 미분을 10-batch terminal 결과로 해석하지 않는다.

| entry | panel | category | signed derivative mean | positive | negative | denominator |
|---|---|---|---:|---:|---:|---:|
| 5000 | Current | signed_N | -0.0544522677 | 15 | 17 | 32 |
| 5000 | Current | signed_R | 35.3527702 | 32 | 0 | 32 |
| 5000 | General | signed_GENERAL_NLL | -0.00567437593 | 7 | 9 | 16 |
| 5000 | Historical | signed_N | -0.0832892638 | 16 | 16 | 32 |
| 5000 | Historical | signed_P | 0.0409764575 | 23 | 9 | 32 |
| 5000 | Historical | signed_R | 0.0490002853 | 16 | 16 | 32 |
| 9000 | Current | signed_N | -0.194505486 | 9 | 23 | 32 |
| 9000 | Current | signed_R | 34.042425 | 29 | 3 | 32 |
| 9000 | General | signed_GENERAL_NLL | 0.00438606304 | 9 | 7 | 16 |
| 9000 | Historical | signed_N | 0.0447788874 | 15 | 17 | 32 |
| 9000 | Historical | signed_P | 0.0178318295 | 13 | 19 | 32 |
| 9000 | Historical | signed_R | -0.0314420189 | 13 | 19 | 32 |

| entry | General state | NLL mean (128 sequences) |
|---|---|---:|
| 5000 | W0 | 2.05983718 |
| 5000 | entry | 2.30413844 |
| 5000 | native | 2.30387261 |
| 9000 | W0 | 2.05983718 |
| 9000 | entry | 2.45722555 |
| 9000 | native | 2.46433396 |

`forward_pairs=2965`는 계약 패널2964 + 동일 fixed input의 own-input-stability 추가1이며 canonical 평가 denominator에 더하지 않는다. Tensor SHA는 dtype/shape prefix 포함 historical 방식이고 checkpoint serialized-file SHA와 별개다. B060/B100 actual tensor SHA가 execution lock expected_W/M과 정확히 일치했다.
