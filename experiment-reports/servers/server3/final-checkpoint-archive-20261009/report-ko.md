# server3 최종 checkpoint 중앙 보존 정책 수락

server3는 `USER-GH-ALL-SH-FINAL-CHECKPOINT-ARCHIVE-SERVER1-20261009-R1-SERVER3`를 수락했다. main `12eff7e3`의 정책 SHA `aa7cc7ee…a202d6`과 조건부 전송·삭제 승인 SHA `ae903819…b201`을 읽고 결속했다. **현재 증거로 최종 보존 적격 checkpoint는 확인되지 않아 전송 0건, server1 검증 0건, 원본 삭제 0건이다.** 정책 수락을 실제 archive 완료로 표시하지 않는다.

Official Qwen server3 runner는 구현만 게시됐고 등록 job이 0개라 checkpoint가 아직 없다. 기존 MEMIT-HJ의 `000-01000-0000.pt`(과거 기록 크기 5,313,662,767B)는 계획된 main 10k 중 1k의 임시 상태다. 원 collector가 `TECHNICAL_INCOMPLETE`를 기록했고 저장공간 점검은 `REPRODUCTION_KEEP`으로 분류했다. 이번 점검에서 이 파일을 열거나 SHA를 다시 계산하지 않았으며, 최종 계산·소비자 종료가 충족됐다고 판단하지 않았다. 다른 역사적 checkpoint에도 종료·소유권·소비자 검증 없이 일괄 적용하지 않는다.

향후 실제 최종 endpoint가 생기면 source SH가 정확한 regular-file allowlist와 full SHA/크기·run/job/계산 완료·소비자 종료를 봉인한다. SH1이 목적지 용량과 최소 64GiB 여유를 확인하고 `/mnt/raid5/janghj/ODE-edit/local/checkpoint-archive/server3/<task>/<run>/<attempt>/job-<id>/<fullsha>/` 아래 unique 경로를 발급한 뒤 별도 사본의 SHA·identity를 독립 검증해 `VERIFIED_DESTINATION`을 발행해야 한다. 그 후 source와 소비자 상태를 재확인하고 해당 payload 파일만 개별 삭제한다. SH1의 공통 수신 API가 준비되기 전에는 임의 전송·source측 대형 복제를 시작하지 않는다. 기존 frozen runner/job 및 noCP 정책은 바꾸지 않았다.

한정 근거는 [server3 audit](../../../../audits/servers/server3/final-checkpoint-archive-20261009/receipt.json), [MEMIT-HJ 1k 리뷰](../memit-hj-20260930-v2/review-1k-20261002-v1/report-ko.md), [저장공간 점검](../price-storage-inventory-20261007/report-ko.md), [official Qwen audit](../../../../audits/servers/server3/official-baselines-20261008/audit.json)이다. 새 GPU·Slurm job·모델 평가·반복 모니터링은 수행하지 않았다.
