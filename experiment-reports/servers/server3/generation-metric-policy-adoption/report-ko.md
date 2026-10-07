# SH3 fluency·consistency 공통 정책 수락 및 baseline 취소 범위 확인

**POLICY_ACCEPTED_FUTURE_CALLER_ADOPTION_REQUIRED.** SH3의 새 baseline 실행 배정은 없다. 2026-10-07 11:47:06 UTC에 owner `janghj`의 Slurm queue를 이름 필터 없이 한 번 확인했으며, server3 `ubuntu` 할당·요청 job과 node 미지정 PENDING은 모두 **0개**였다. 따라서 SH3 소유의 여섯 stock baseline 활성 취소 대상은 **0개**, 실제 취소도 **0개**다. 다른 서버에 있는 owner job 15행은 SH3 권한 대상으로 취급하지 않았다. 이 snapshot은 현재 시점 관측이며 향후 실행 예약이나 지속 감시가 아니다.

Nonce `USER-GH-ALL-SH-BASELINE-FLUENCY-CONSISTENCY-RERUN-20261007-R1`를 수락했다. 정본 main `b0cee1a302c10d48e206a31c7c5cf318ac36d8c3`의 SH3 envelope, 전체 계약, generation 정책, 기존 RPN/W&B 정책을 읽고 경로·size·SHA를 [감사 기록](../../../../audits/servers/server3/generation-metric-policy-adoption/acceptance.json)에 결속했다. 전달된 envelope SHA가 일치한다. 실제 host/session/worktree boundary PASS이며 기존 동일 nonce 접수/제출 기록은 없었다.

정본 여섯 stock 방법은 MEMIT, PRUNE, RECT, AlphaEdit, AlphaEdit-BLUE, CAKE다. 전체 재실행의 실험 owner는 SH1 GPT2-XL, SH2 GPT-J, SH4 Llama3다. SH3 envelope의 model/method/historical candidate 목록은 비어 있고 SH3 신규 모델 실행 권한은 없다. 취소 판단은 owner·node·Command·task 역할·source/manifest·state가 모두 결속된 실제 job에만 적용한다. 이번에는 server3 queue 행 자체가 없어 개별 `scontrol` 또는 `scancel`을 실행하지 않았다. PRICE ours, W0, FE, MEMIT-H/MEMIT-HJ, 다른 task는 보호한다.

향후 SH3가 별도로 승인받아 **새로 봉인하는** ours/baseline caller는 SH1 공통 generation evaluator와 세 reference 자산의 정확 SHA/profile을 함께 결속한다. CounterFact의 고정 `generation_prompts`를 전체 prompt 포함 최대100 token, native EOS로 종료하고, 모델·요청 occurrence·generation-prompt index 기반 seed20261007을 사용한다. 편집 RNG·W/H/context는 observer 전후 보존한다. 한 endpoint/case/profile/seed의 생성 관측을 두 지표에 재사용한다.

Fluency는 전체 생성문 word n-gram entropy `2/3·H2 + 4/3·H3`의 case macro **bits**다. Consistency는 relation/target 전체 snippets를 고정 vocab/idf로 벡터화한 cosine의 valid-case macro다. lexical 유사도이며 factual 인증 점수로 해석하지 않는다. 결측은0으로 대체하지 않고 planned/valid count 및 이유를 남긴다. R/P/N 성공률의 기존 3항 조화평균에는 두 지표를 넣지 않는다. W&B에는 scalar·분모·edits축·실제 job ID만 기록하고 원문 생성·token·case ID·tensor는 업로드하지 않는다. SH1 asset READY/source는 다음 별도 승인 신규 source 봉인 시 확인한다.

이번 SH3 작업은 정책·취소범위 감사다. SH1 공통 평가기 구현, 자산 전송, GPU/model 평가, 새 Slurm submit, W&B 온라인 smoke, 과거 run backfill은 수행하지 않았다. 재검토자는 정본의 SH3 책임과 취소 범위를 독립적으로 읽기 전용 확인했으며 과학 raw 또는 Slurm을 재조회하지 않았다. 낮은 과학 성능은 정책 적용 여부나 기술 성공 판정에 사용하지 않는다. 기존 source/raw/job은 그대로 보존하고 이번 한정 점검을 종료한다.
