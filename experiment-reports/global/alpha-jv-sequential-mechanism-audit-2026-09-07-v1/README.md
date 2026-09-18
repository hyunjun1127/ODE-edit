# GH 분석 산출물 안내

- 주 보고서: [gh-mechanism-review-ko.md](gh-mechanism-review-ko.md)
- 후보표: [sweep-candidates.csv](sweep-candidates.csv), [정책](sweep-candidate-policy.json)
- 신규 기전 그림: [mechanism-evidence.png](derived/mechanism-evidence.png)
- 독립 소형 행렬 검증: [controller_barrier_audit.csv](derived/controller_barrier_audit.csv)
- λ 13점 × 80 node: [same-state shadows](supporting/llama-lambda-shadow-all1040.csv)
- 후보별 민감도: [52행 summary](supporting/candidate-sensitivity-summary.csv)
- 실행 knob와 baseline hparams: [감사](supporting/sweep-runtime-knobs-audit.md)

모든 신규 계산은 기존 publication의 CPU 분석이다. 원본 raw tensor/evaluator 재검증 및 신규 GPU 실험을 수행하지 않았다. 보고서의 가설과 실행 후속 후보는 새 실험 결과가 아니다. 원본 source/SH1/SH2 report는 불변이며 현재 GH 패키지는 별도 보고서다.

## 재현

기존 환경 `/mnt/raid5/janghj/EasyEdit/.venv/bin/python`의 NumPy/Matplotlib를 사용한다. 모델을 import/load하지 않는다. 다음 input roots는 봉인된 publication이다.

```bash
/mnt/raid5/janghj/EasyEdit/.venv/bin/python \
  experiment-reports/global/alpha-jv-sequential-mechanism-audit-2026-09-07-v1/analyze.py \
  --sh1 /mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s06-alpha-jv-sequential-detailed-analysis-v1/experiment-reports/servers/server1/alpha-jv-sequential1000-layer-review-2026-09-07-v1 \
  --sh2 /mnt/raid5/janghj/.codex/worktrees/odeedit-gh-alpha-jv-review-20260907/experiment-reports/servers/server2/alpha-native-response-v31-sequential-routing-2026-09-06-v1/main-four-terminal-v1 \
  --output /absolute/path/to/a/new-create-once-analysis-directory
```

`supporting/llama_sensitivity.py`, `supporting/summarize_candidate_sensitivity.py`는 표준 Python만 사용한다. 전자는 파일 내 봉인 input 경로를 읽고 script 옆에 create-once CSV/JSON을 생성한다. 후자는 해당 CSV에서 group 요약을 만든다. 재실행하려면 script 사본을 새 빈 분석 디렉터리에 두어야 하며 기존 산출물을 덮어쓰지 않는다. 입력 source path가 다른 host에서는 같은 SHA의 publication 경로를 명시적으로 재binding하고 기록해야 한다.

`sweep_candidates.py` 역시 script 옆에 create-once 후보표를 만드는 코드이며 scheduler 호출이 없다. `seal_report.py`는 source 핵심 파일 동일성, 독립 spectrum 계산, CSV/PNG byte 재현, 문서 link를 점검하고 이 패키지의 manifest/receipt를 생성했다. 패키지는 create-once이므로 동일 경로에 재실행해 덮어쓰지 않는다.

현재 host에서 두 번 생성한 CSV/PNG는 byte가 같았다. 자세한 범위는 `validation.json` 참조. `analysis-manifest.json`과 `rooted-receipt.json`은 본 GH 패키지만 결속하며, 입력 publication 82개 member의 identity는 `derived/verification.json`에 있다.
