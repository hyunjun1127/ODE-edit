"""Explicit migration authority, no additional scientific qualification."""
AUTHORITY = 'USER-GH-S4-S2-QWEN-BASELINES-MIGRATION-RESULTS-20261009-R1'
QUALIFICATION_STATUS = 'NOT_RUN_USER_DISABLED'

def require_execution_enabled():
    # Authority permits only the exact source-owner twelve-row contract.
    return AUTHORITY
