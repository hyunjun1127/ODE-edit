# server3 zsRE Loc 정정 적용 대상 점검

최신 nonce `USER-GH-ZSRE-LOC-RECALCULATE-MAIN-20261009-R1`과 main `a8d8db88` envelope를 전체 읽고 수락했다. 새 Loc는 `100 × mean_requests(mean_loc_ans_tokens(predicted == target))`다. W0 prediction agreement는 보조 지표로 분리한다. token-micro로 대체하거나 summary 이름만 바꾸지 않는다.

**점검범위 내 본인 완료 zsRE baseline/ours 결과 0개. 재집계 대상 없음.** 기존 official 제출 요청 receipt는 BLOCKED_NOT_SUBMITTED/registered_jobs=[]이고, zsRE caller adoption receipt 역시 NOT_SUBMITTED다. 이들은 당시 상태의 증거이며 옛 자산 blocker가 지금도 모두 그대로라는 주장은 아니다. 이번 local 프로젝트 metadata 점검에서도 완료 zsRE 증거를 발견하지 않았다. Qwen 61813은 CF W20이므로 이 계산에서 제외했다. 다른 서버의 Qwen migration 및 zsRE 결과를 server3 실적으로 계산하지 않았다.

`audits/servers/server3/official-baselines-20261008/zsre-loc-review-20261009/inventory.json`에 경로/크기/SHA/점검시각과 coverage를 기록했다. 알려진 local 루트의 깊이6 이하 소형 terminal/result/completion/submission/status를 읽었으며, 원 source/worktree 복사본·raw chunk·환경·모델·symlink를 제외했다. 전체 파일시스템 부재증명은 아니다. registry worktree 목록과 own official report/receipt도 확인했다. 실제 W20 zsRE raw가 없으므로 before/after·Eff/Gen·token/case 분모·reducer SHA는 NOT_APPLICABLE/null이며 성능 0점이 아니다. CSV도 값은 공란이다. inventory script SHA는 reducer SHA와 구별했다.

향후 승인된 새 source freeze에 GH 공통 READY 구현을 채택한다. 이번 작업에서 공통 코드/README/기존 frozen caller를 수정하지 않았다. CF 수치와 Flu/Con DEFERRED 그대로, 새 GPU/forward/Slurm/온라인 history 변경/CP 복원·이동·삭제는 0이다. 원 실행·raw·CP는 KEEP. 집계 정확성을 원 tokenizer 완전 동등성 또는 논문 재현 PASS로 확대하지 않는다. owner CPU audit이며 별도 reviewer는 사용하지 않았다.
