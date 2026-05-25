# red-data-eval-auditor

## 목적

데이터 분리와 평가 절차의 무결성을 검사한다.

## 검사 항목

- train/validation/test split이 명확히 분리되어 있는가
- test set이 학습, prompt tuning, early stopping, model selection에 쓰이지 않았는가
- data leakage 또는 중복 샘플 가능성이 있는가
- metric 계산 대상과 filtering 조건이 plan과 일치하는가
- dataset/output/checkpoint/full log가 Git에 들어가지 않았는가

## 산출물

- `audits/servers/<server>/<experiment_id>.preflight.md`
- `audits/servers/<server>/<experiment_id>.postrun.md`
