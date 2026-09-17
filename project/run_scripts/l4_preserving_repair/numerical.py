"""Pre-GPU technical choices, not efficacy gates or adjustable science knobs."""
POLICY = dict(
    version='L4R_NUMERICAL_V1', epsilon_L=1e-4, epsilon_B=1e-6,
    repeat_E_H=5e-5, repeat_B=5e-7, strict_preference_repeat='EXACT_ID_SET',
    minimum_agreement=.1, max_endpoints=6, shrink=.5, max_condition=1e6,
    KKT_scaled=1e-8,
    FD_relative_weight_steps=[1e-4,5e-5], FD_direction='each objective normalized full W8 gradient',
    FD_relative_tolerance=.15, FD_absolute_tolerance=1e-6,
    FD_stability_relative=.15, FD_noise_multiplier=8.,
    JVP_projection_relative=1e-3, JVP_projection_absolute=1e-7,
    functional_E_H=5e-5, functional_B=5e-7,
    materialization='CPU FP32 anchor clone; for j in fixed Q order add FP32(float(c_j)*Q_j); zero exact copy',
    guard_competitor='WN highest non-target; lowest token index on ties',
    early_cohort='first received B100, observer after selection seal',
    checkpoint='SKIPPED_USER_DIRECTED', exact_restart='NOT_AVAILABLE')
