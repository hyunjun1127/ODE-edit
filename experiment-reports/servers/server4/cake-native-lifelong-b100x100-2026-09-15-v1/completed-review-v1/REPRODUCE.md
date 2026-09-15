# CPU 재현

새 output만 사용한다. Scheduler 조회는 기존 receipt를 재사용한다.

```bash
python -B -m project.run_scripts.server4_completed_review.cake full --output NEW_CAKE
python -B -m project.run_scripts.server4_completed_review.plot_cake --package CAKE_PACKAGE --output NEW_FIGURES
python -B -m project.run_scripts.server4_completed_review.validate_publication --output NEW_VALIDATION
```

실제 환경 Python3.12.3 / EasyEdit .venv. 9개 그림은 같은 CSV/코드/Matplotlib 환경에서 byte-identical 재생성됐다. CAKE codePNG는 이 package figures/plot-manifest.json에 결속된다. W/M 미저장으로 GPU continuation 또는 tensor replay는 재현 범위가 아니다.
