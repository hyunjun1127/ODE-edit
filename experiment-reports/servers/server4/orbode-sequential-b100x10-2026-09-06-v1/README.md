# ORBODE sequential canonical report

[한국어 보고서](orbode-sequential-b100x10-fourcell-factual-ko.md)

Scope: arm-local B100×10 cumulative W/cache sequential. Reports are online/immediate; final full-history evaluation was not recorded.

Analysis-only reproduction (use the pinned CPU environment with NumPy/Pandas/Matplotlib):

```bash
python -m project.run_scripts.ordered_response_barrier_ode.sequential_analysis --raw-root RAW_ROOT --output NEW_ANALYSIS_DIR
python -m project.run_scripts.ordered_response_barrier_ode.sequential_report --tables NEW_ANALYSIS_DIR --raw-root RAW_ROOT --output NEW_PACKAGE --test-receipt TEST_RECEIPT
python -m project.run_scripts.ordered_response_barrier_ode.sequential_package_verify NEW_PACKAGE
```

Static figures: regenerate from sealed CSV using plot-reproduction.json. No model/GPU or evaluator invocation.
