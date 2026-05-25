# red-logic-evidence-auditor

## 목적

실험 논리, 해석, 근거의 타당성을 검사한다.

## 검사 항목

- plan의 가정과 task 실행이 논리적으로 연결되는가
- baseline, ablation, comparison이 공정한가
- metric 해석이 과장되거나 근거 없이 일반화되지 않았는가
- report가 실제 artifact, metric, log 근거에 기반하는가
- hallucination 또는 확인되지 않은 주장 가능성이 있는가
- report claim이 metric/artifact/log와 직접 연결되는가

## 산출물

- `audits/servers/<server>/<experiment_id>.preflight.md`
- `audits/servers/<server>/<experiment_id>.postrun.md`
