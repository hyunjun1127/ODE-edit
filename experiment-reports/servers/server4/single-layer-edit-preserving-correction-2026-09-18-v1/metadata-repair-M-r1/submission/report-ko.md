# ENFC M 저장 메타데이터 기술 수리·재제출 submission

상태: **M10_REPAIR_RELEASED_AWAITING_ACTUAL_INITIAL**. 사용자 “기술적 오류는 해당 오류 보고 이후 SH가 직접 수정후 재제출해”를 적용했다.

## 첫 오류와 최소 수리

50021_0의 첫 N4 final-L4 저장 후 weights_only reload가 TorchVersion unsupported global로 실패했다.
`torch.__version__`는 str처럼 출력되지만 실제 subclass이며 pickle metadata에서 안전 로더가 거절했다.
Tiny CPU 재현과 원 파일의 scoped safe-global CPU 검사로 원인을 확인했다. 무제한 pickle load로 우회하지 않았다.
수리는 endpoint metadata를 builtin str로 기록하는 것으로 제한했다. 원 runtime/native/alltoken/geometry/
optimizer/observer/guard/threshold bytes는 동일하다. 수치 실패나 효능 결론, ENOSPC 실패가 아니다.

50021_0/1은 FAILED, 실행·대기 중인 exact _2–9만 취소했다. 다른 job 변경0, 원파일 삭제0,
cancelled process rollback NOT_VERIFIED. 첫 실패·부분 산출물·원 source와 비용은 보존했다.
같은 source 재시도에서 완료된 B2/B3 native fileSHA/weights_only/finite/weightSHA/요청순서/W0/M0/P4/
history0를 CPU 검사했다. B1+B2+B3 native REUSE, B4–10 최대7fit/700target; 중단된 B4 incomplete 작업을
완료 capsule로 만들지 않았다. 재사용은 full process 또는 GPU crash-resume과 다르다.
새 좁은 CPU7검사는 serialization 및 재사용 valid/order/warm/P/weight/history 오류경계다.
기존 CPU82와 skip4/storage3의 변경 없는 근거를 재사용하며 새로운 T/FD/noop/teacher 검증0이다.

## 전체 M 제출

배열 **50050_[0–9]%2**, 10/10 held owner/source/args/resource/dependency 검사 후
2026-09-18T04:22:31.192135+00:00 release. 각1GPU/8CPU/60416MiB/48h/exportNONE/Requeue0, cap2, hour cap=null.

| Index | 독립 cold episode | Native | 요청 ordinal |
| --- | --- | --- | --- |
| 0 | b001 | 원 cold7 REUSE | [0,100) |
| 1–2 | b002–b003 | 완료된 50021 cold native REUSE | [100,300), 각100 |
| 3–9 | b004–b010 | RUN_MISSING, 최대7fit | [300,1000), 각100 |

각 episode의8arm은 동일 native capsule을 공유하고 history0, W0/M0 independent reset을 유지한다.
80 final L4 endpoint/RAND±/CA-EXACT/공식평가 저장 의무는 불변이다. T dependency/failcancel0, S/R/L0.
T=**SKIPPED_USER_DIRECTED**, full_numerical_validation=**NOT_ESTABLISHED**.

## 실제 관측 범위

관측시각 2026-09-18T04:22:44.189818+00:00.
실제 M 초기 gate는 아직 미관측이다. 제출 완료와 과학 완료를 구분하며 이 task만 bounded 관찰한다.

## Source·저장·비용

Frozen execution `87f65ea2abcbe7e77e04367f73a001d63443734b` / tree `980c04b3e998caab0284b855041f6ad3e2e30356`.
Archive SHA `d3dd9b928a25da42ce4bca4e7dcc3288ef9cd5122d57ef83c88291b5d639a65f`.
Execution lock SHA `635dd6e953e32e8278a8d7ff5c2cb3f3c9a6ad3625fbed14bc67da9a3c71a5a1`.
원자료 root `/data/janghj/ODE-edit/local/single-layer-edit-preserving-correction/20260918-v1/M/attempt-metadata-r1`. 실행 source와 본 publication source는 구분한다.

제출 free=195109154816B, 원72GiB reserve=77,309,411,328B.
USER_WAIVED_RESERVE_PENDING_USER_SPACE_CLEANUP은 유지한다. 사용자 cleanup 완료/여유공간 원인은
미검증이며 exclusive reserve나 미래80endpoint 저장 보장이 아니다. 실제 IO/atomic·저장완결성 오류는
실패로 남긴다. SH 삭제/이동0.

기존 T49928/M49973 3231GPU초, 50021 실패·중단 시도2301GPU초를 별도 보존한다.
새 50050 allocation은 진행 중이므로 최종 비용을 확정하지 않는다. 재사용 native/teacher와
batch/extern 또는 중첩 component timer를 다시 합산하지 않는다.

## 증거와 한계

[Receipt](receipt.json), [input manifest](input-manifest.json).
별도 독립 red agent는 사용하지 않았으며 SH 자체 source/state/reuse/resource/raw-free 검사를 수행했다.
파일/CPU hash를 T numerical PASS 또는 GPU continuation으로 승격하지 않는다.
M 상세 통계와 S/R/L 확대는 이번 initial 인계 범위 밖이다.
monitoring_active=true, automatic_resume=false.
