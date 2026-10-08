"""Explicit USER zsRE six-arm launch; no CF scheduling changes."""
INSTRUCTION = 'USER-SH2-ZSRE-WANDB-LAUNCH-20261009-R1'
METHODS = ('MEMIT', 'FT', 'ALPHAEDIT', 'ALPHAEDIT_BLUE', 'MEMIT_FE', 'SPHERE')


def profile():
    return dict(instruction_id=INSTRUCTION, dataset='zsre', requests=2000,
        batch_size=100, batches=20, methods=list(METHODS), project_gpu_cap=4,
        model_smoke_method='MEMIT', model_smoke_batches=1,
        cold_independent_chains=True, generation=False,
        checkpoint='LATEST1_FINAL_W20_KEEP', automatic_retry=False)


def enabled(manifest):
    value = manifest.get('zsre_launch_profile')
    if value is None:
        return False
    if value != profile() or 'checkpoint_only_profile' in manifest:
        raise ValueError('EXACT_ZSRE_USER_PROFILE_REQUIRED')
    return True
