# 완료 결과 및 FE-author 본표 통합

사용자 nonce `USER-GH-S1-S2-FE-AUTHOR-MAIN-REFRESH-20261010-R1`.
2026-10-11 KST SH1/SH2 직접 수락 및 단발 결과 검산을 회수하고 GH가 README를 통합했다.
task 경로의 20261010은 지시 식별자이며 실제 SH2 관측은 10월11일 00:02:48 KST,
SH1 영수증은 00:03:41 KST다. 새로운 GPU 계산/등록/취소/CP 변경은 없다.

## FE 행 이동

상단 별도표의 author4행을 Llama/Qwen 모델별 본표 CF/zsRE 칸으로 통합했다.
이름은 `MEMIT-FE (FE author hparams + history)`이며 실제 method `MEMIT_FE_HISTORY`다.
원저자 clamp/steps는 Llama .75/35, Qwen1/35이고 history 유지 variant다.
DOW-KE exact reproduction을 뜻하지 않는다. 이전 native FE 두 모델의 값은
이전 설정 표로 이동해 보존했다. 이전 history3행은 별도 유지한다.
GPT-J author 실험은 없으므로 원 native 행에 legacy/author 미실행을 명시했다.

| 모델 | author CF job | Score | Eff | Gen | Loc | author zsRE |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Llama3 | 62529 W20/2000 | 90.58 | 99.70 | 95.55 | 79.22 | 62530 RUNNING, 14 commits, W20 미완료 |
| Qwen2.5 | 62531 W20/2000 | 48.96 | 48.40 | 48.20 | 50.33 | 62532 RUNNING, 16 commits, W20 미완료 |

CF8개 값과 zsRE6개 상태 칸을 반영했다. author CF Flu/Con4칸은 DEFERRED이며
구 native CP 평가값을 복사하지 않았다. checkpoint/source/config/profile/ordered2000 및
20commit 검산은 아래 owner 원자료 영수증에 결속했다.
GH도 두 author CF 저장 raw를 SHA 확인 후 직접 읽고 strict NLL 성공을 독립 재집계했다.
Llama R/P/N=1994/2000,3822/4000,15843/20000;
Qwen=968/2000,1928/4000,10065/20000. 모든 request의 prompt 수1/2/10을 확인했다.
정확 요청별 평균을 유리수로 계산한 뒤 half-up2 표시한다. Qwen Loc50.325는50.33이며
summary의 부동소수 표현50.324999999999996을 raw 변경 없이 구분했다.

## 추가 완료 generation

| 기존 실행 | 평가 job | Flu ×100 | Con ×100 | 위치 |
| --- | --- | ---: | ---: | --- |
| GPT-J FT61650 | 62864 | 510.98 | 3.04 | 본표 |
| GPT-J MEMIT61725 | 62865 | 577.93 | 36.84 | 본표 |
| GPT-J SPHERE61781 | 62869 | 616.18 | 40.89 | 본표 |
| Qwen 이전 history62061 | 62583 | 532.39 | 0.41 | 별도 history표 |

8개 생성 셀을 완료값으로 교체했다. 각2000 case/20000 prompt,
valid Flu/Con2000을 확인했다. raw bits/cosine은 보존하고 unrounded mean×100 후 반올림한다.
Qwen AlphaEdit generation62871은 RUNNING으로 두 상태 셀을 갱신했다.
나머지 미완료 생성 평가와 SPHERE B9재개62538은 owner 관측 PENDING을 유지한다.

SH1 완료13/미완료1, SH2 완료24/미완료2의 compact행을 회수했다.
기존 완료 zsRE17행은 공개-query/source 증거와 저장 predicted/target token ID의
요청별 E/G/loc_ans 평균을 다시 검산해 기존값과 같았다. W0agreement/tokenmicro를
Loc로 사용하지 않았다. CPU query/parsing 일치는 pretrained forward parity 주장이 아니다.
author zsRE2행과 Qwen SPHERE 재개1행은 미완료라 최종값을 넣지 않았다.
PRICE 및 W0 행과 그 외 기존 factual값은 그대로 보존했다.

## 근거와 검증

- [SH1 보고](../../servers/server1/author-main-refresh-20261010/report-ko.md), [exact rows](../../../audits/servers/server1/author-main-refresh-20261010/table-rows.json): main cba132a9, own90997cbf.
- [SH2 보고](../../servers/server2/author-main-refresh-20261010/report-ko.md), [exact rows](../../../audits/servers/server2/author-main-refresh-20261010/table-rows.json): own f3d7d2b8, 후속 main2330ae1e.
- [직접 ACK/accepted turn](../../../audits/global/author-main-refresh-20261010/dispatch.json).
- [GH exact CF 분자/분모](../../../audits/global/author-main-refresh-20261010/exact-counts.json), [read-only 집계](../../../audits/global/author-main-refresh-20261010/exact_cf_counts.py).
- [소형 receipt→표 셀 검증](../../../audits/global/author-main-refresh-20261010/verify.py).

server1/server2 cap2 및 기존 jobs/dependency/source/raw/CP/W&B 불변.
신규 GPU/forward/CPload/삭제/전송/recurring monitor 없음. SH2의 역방향 RPC 응답은
timeout이었지만 GH가 exact accepted turn의 완료 응답과 main게시 자료를 직접 회수했다.
