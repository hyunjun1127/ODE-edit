# WITHDRAWN / SUPERSEDED — 이전 FE 결과

사용자 `USER-FE-ORIGINAL-W0-RESET-20261011-R1`로 아래 모든 FE 결과를 현재 공식 비교에서 철회했다.
이 파일은 README d79098b1의 작은 수치·출처 이력만 보존한다. 새 author-repo W0-fixed 결과가 아니다.
이전 checkpoint는 삭제 승인 범위이며 현재 owner 정리 진행 중이다. 아래 과거 KEEP/CP 존재 문구는
당시 관측일 뿐 현재 보존 지시가 아니다. actual 삭제 여부/수량은 후속 owner manifest/receipt를 따른다.
raw/log/config/source 및 수치 이력은 보존한다. 삭제 완료를 미리 주장하지 않는다.

## 철회된 이전 모델별 본표 행

### Llama3

| 이전 방법 | CF Score | CF Eff | CF Gen | CF Loc | CF Flu ×100 | CF Con ×100 | zsRE Eff | zsRE Gen | zsRE Loc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MEMIT-FE (FE author hparams + history) | 90.58 | 99.70 | 95.55 | 79.22 | DEFERRED | DEFERRED | ING: official-s1-zsre-llama3-memit-fe-author-history (62530) | ING: official-s1-zsre-llama3-memit-fe-author-history (62530) | ING: official-s1-zsre-llama3-memit-fe-author-history (62530) |

### Qwen2.5

| 이전 방법 | CF Score | CF Eff | CF Gen | CF Loc | CF Flu ×100 | CF Con ×100 | zsRE Eff | zsRE Gen | zsRE Loc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MEMIT-FE (FE author hparams + history) | 48.96 | 48.40 | 48.20 | 50.33 | DEFERRED | DEFERRED | ING: s2-qwen25-zsre-fe-author-history (62532) | ING: s2-qwen25-zsre-fe-author-history (62532) | ING: s2-qwen25-zsre-fe-author-history (62532) |

### GPT-J

| 이전 방법 | CF Score | CF Eff | CF Gen | CF Loc | CF Flu ×100 | CF Con ×100 | zsRE Eff | zsRE Gen | zsRE Loc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MEMIT-FE (legacy native; author 미실행) | 73.85 | 87.15 | 85.70 | 57.22 | PENDING: s2-flucon-gptj-memit_fe-61780 (62868) | PENDING: s2-flucon-gptj-memit_fe-61780 (62868) | 29.20 | 27.65 | 8.45 |

## 철회된 이전 별도표와 당시 설명

아래는 변경 전 스냅샷이며 모든 FE 결과에 WITHDRAWN 상태를 적용한다.

이전 **MEMIT-FE + history** CF 3행은 FE-author 설정의 본표 결과와 구분해 보존한다.
SH1이 세 모델을 source `eaf78c33`으로 등록했다. Qwen 61929는 B1 FP64 history 선형계의
임시 행렬 메모리 부족으로 실패하여 메모리 수리 source `5d6dfd58`의 61975로 cold 재제출했다.
이후 61975의 context-mask 오류로 다시 cold 제출한 **62061**의 완료값을 사용한다.
오류가 있던 61975 값은 제외하며 zsRE eval-only 작업과 구분한다.

| 별도 variant | 모델 | dataset | server | 상태 / job name (ID) | Score (CF) | Eff | Gen | Loc | Flu ×100 (CF) | Con ×100 (CF) |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| MEMIT_FE_HISTORY | GPT-J | CF | server1 | W20 COMPLETE: official-s1-cf-gptj-memit-fe-history (61927) | 50.18 | 50.35 | 49.90 | 50.28 | 533.43 | 1.05 |
| MEMIT_FE_HISTORY | Llama3 | CF | server1 | W20 COMPLETE: official-s1-cf-llama3-memit-fe-history (61928) | 64.66 | 80.10 | 74.10 | 48.99 | 527.39 | 7.58 |
| MEMIT_FE_HISTORY | Qwen2.5 | CF | server1 | W20 COMPLETE: official-s1-cf-qwen25-memit-fe-history (62061) | 49.43 | 48.45 | 48.18 | 51.83 | 532.39 | 0.41 |

**FE-author 본표 통합(사용자 최신 지시):** 위에 따로 두었던 Llama/Qwen author 4행은
아래 모델별 **MEMIT-FE (FE author hparams + history)** 행의 CF/zsRE 셀로 이동했다.
Llama clamp/steps=0.75/35, Qwen=1/35이며 persistent history와 batch별 z 재계산을 유지한다.
실행 method는 `MEMIT_FE_HISTORY`다. native MEMIT-FE나 위 이전 history 결과의 이름만 바꾼 것이 아니다.
GPT-J는 author 실험이 없어 기존 본표 결과를 legacy native로 명시한다. 신규 실험은 추가하지 않았다.
CF Flu/Con에는 같은 author checkpoint의 평가만 연결하며 이전 설정의 생성 점수를 복사하지 않는다.
DOW-KE exact reproduction을 주장하지 않는다.
[author source/config/profile·실제 등록](../../../experiment-reports/global/fe-author-hparams-2k-20261010/report-ko.md).

이전 Llama/Qwen native 설정 결과는 아래 이력 표에 보존한다(최신 author 본표 값 아님).
구설정 factual 수치는 보존하고, 해당 checkpoint의 후속평가 상태만 이번 owner snapshot으로 확인했다.

| 모델 | 이전 방법 | CF Score | CF Eff | CF Gen | CF Loc | CF Flu ×100 | CF Con ×100 | zsRE Eff | zsRE Gen | zsRE Loc |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama3 | MEMIT-FE (이전 native 설정) | 64.55 | 80.45 | 73.68 | 48.85 | 420.07 | 8.40 | 14.81 | 13.57 | 0.56 |
| Qwen2.5 | MEMIT-FE (이전 native 설정) | 50.59 | 51.05 | 51.23 | 49.53 | PENDING: s2-flucon-qwen25-memit_fe-62077 (62872) | PENDING: s2-flucon-qwen25-memit_fe-62077 (62872) | 0.00 | 0.00 | 0.03 |

2026-10-10 SH1 bounded 검산: 세 모델 모두 실제20 commit/2,000건과 history once·최종CP·raw를
대조한 완료값이며 native MEMIT-FE 본표 값이 아니다. 17:10:08 KST 단발 관측에서
GPT-J **62581**과 Llama **62582** generation이 완료됐고, 각2,000 case/20,000 prompts의
reference-bound CPU 재채점·원CP/fullSHA·20commit/순서 검산으로 위 FLU/CON 네 수치를 반영했다.
raw bits/cosine을 보존하고 원평균×100 뒤 half-up2로 표시한다.
Qwen **62583** 및 GPU0 collector **62584**는 2026-10-11 검산에서 완료됐다.
2,000 case의 raw Flu5.323883976741921/Con0.0041069385104137695를 ×100해 **532.39/0.41**로 반영했다.
기존62262/62263 CANCELLED 이력과 완료 native 생성62259–62261은 보존한다. 현재 server1/2 cap은 각각2다.
SH2는62581–62583을 중복 등록하지 않는다. 새 결과 검산은 온라인 readback 검증 주장이 아니다.
[최신 history 생성 결과·zsRE 재검산·중복방지 인계](../../../experiment-reports/servers/server1/baseline-refresh-s2-flucon-20261010/report-ko.md).
[최신 결과·실제 평가 등록·checkpoint 결속](../../../experiment-reports/servers/server1/completed-table-flucon-cap3-20261010/report-ko.md).
[최신 history 및 zsRE 검산](../../../experiment-reports/servers/server1/baseline-completed-zsre-audit-20261010/report-ko.md).
[완료 결과와 평가 등록](../../../experiment-reports/servers/server1/flucon-paper-scale-20261010/report-ko.md).
[최신 SH1 결과](../../../experiment-reports/servers/server1/main-table-refresh-20261009/report-ko.md) ·
[정확한 수치·source/config·raw SHA](../../../audits/servers/server1/main-table-refresh-20261009/table-rows.json) ·
[Qwen OOM 수리 이력](../../../experiment-reports/servers/server1/memit-fe-history-three-model-2k/oom-repair-r1.md).

