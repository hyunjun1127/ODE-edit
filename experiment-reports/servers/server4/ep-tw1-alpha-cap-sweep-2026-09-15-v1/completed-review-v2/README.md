# EP alpha-cap 독립 정본 v2

[상세 보고](diagnostic-report-ko.md) · [구판→신판 mapping](source-to-v2.json)

이 패키지는 완료된 aggregate publication만 재구성한다. 원raw/model/evaluator를 읽거나 실행하지 않는다. 기존 v1 검증 수준을 유지한다.

```bash
python -B -m project.run_scripts.server4_completed_review.report_separation.build build
python -B -m project.run_scripts.server4_completed_review.report_separation.plots --output NEW_FIGURES
/usr/bin/python3 -B -m project.run_scripts.server4_completed_review.report_separation.checks --output NEW_CHECKS
```

build는 v2가 없는 clean worktree에서 create-once 실행한다. plots는 새 출력에 쓰고 checks는 기존 CSV·GFM HTML·hash만 확인한다. 실행/분석/현재 publication commit은 manifest에서 구분한다. 새 numerical PASS를 부여하지 않는다.
