# V12 SH4 → SH3 중단·인계

Nonce: `ODEEDIT-GH-SH4-SH3-JLZ-V12-MIGRATION-20261004-R1`

2026-10-04 10:35:22 UTC, 실제 owner/source/launcher SHA/full sbatch argv를 재확인한 뒤 **collector 58174 → main 58173** 순서로 취소했다. 두 job은 PENDING에서 CANCELLED가 됐으며, 대상 queue 0 및 AllocTRES null을 확인했다. 본선·collector 실행 비용은 각각 0 GPU초다. 같은 task 추가 registration/active worker는 발견되지 않았다. 다른 job 변경은 없다.

완료 파일럿 58172(410 GPU초), CPU/source/원raw는 모두 KEEP이다. S4 본선은 시작하지 않았으며 main commit 0이다. v11 STOP을 유지하고 SH4의 새 submit/release/retry/과학 실행 권한을 중지했다. 모니터링·자동 재개는 false다.

실행 source는 `7852d66ed5467f94ebda73507b6ccdbf5c921328`이며, publication `22ce7bfd`와 구별한다. 인계 JSON은 본 audit의 `migration-to-s3-r1/handoff.json`이다. exact config/lock/fullargv/runtime/native/evaluator closure, 자산의 기존 full SHA + 현재 stat 증거, 완료 pilot provenance를 포함한다.

선택 가능한 소형 allowlist는 106개 정규 파일 / 952,645B이다. SH3가 없는 파일만 sole pull하고 SHA/size 검증·create-once·source KEEP을 지킨다. 구현·정본은 Git을 우선 사용한다. 모델/C0/dataset/CP/원 평가 raw의 대형 전송은 허용하거나 수행하지 않았다. S4 W0 metadata는 S3 comparability를 보장하지 않으며, S3 actual qualification과 cold main이 별도로 필요하다.

앱의 직접 메시지 도구가 unavailable을 반환해 SH3/GH direct 메시지는 전달 완료로 주장하지 않는다. Git main의 cessation receipt와 handoff가 영속 인계 경로다. 원source 및 중단 전 파일을 수정·삭제하지 않았다. 새 GPU 실행과 상세 결과 리뷰는 하지 않았다.
