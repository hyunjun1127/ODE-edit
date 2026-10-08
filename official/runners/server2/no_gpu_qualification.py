"""Explicit USER overlay: cold main computations, no qualification observations."""
INSTRUCTION = 'USER-GH-SH1-SH2-OFFICIAL-NO-GPU-QUAL-RERUN-20261009-R1'
DISABLED = 'NOT_RUN_USER_DISABLED'
ROLES = ['W0_CF', 'W0_ZSRE', 'CF_MEMIT', 'ZSRE_FT', 'CF_ALPHAEDIT',
         'ZSRE_MEMIT', 'CF_ALPHAEDIT_BLUE', 'ZSRE_ALPHAEDIT', 'CF_MEMIT_FE',
         'ZSRE_ALPHAEDIT_BLUE', 'CF_SPHERE', 'ZSRE_MEMIT_FE', 'ZSRE_SPHERE']


def profile():
    return dict(instruction_id=INSTRUCTION, qualification=DISABLED,
        checkpoint_resume_GPU_equivalence=DISABLED, native_oracle_extra_forward=DISABLED,
        cf_generation_schedule='DEFERRED_CHECKPOINT_EVALUATION', zsre_generation=False,
        cold=True, project_cap=4, requests=2000, batch_size=100, batches=20,
        kept_completed_CF_FT_job='61650', roles=ROLES, automatic_retry=False)


def enabled(manifest):
    value = manifest.get('no_gpu_qualification_profile')
    if value is None:
        return False
    if value != profile():
        raise ValueError('NO_GPU_QUALIFICATION_EXACT_USER_OVERLAY')
    forbidden = ('qualification_plan', 'qualification_plan_sha256', 'native_parity_plans',
        'cf_native_reference_plan', 'qualification_input_plan', 'qualification_receipt',
        'zsre_smoke_receipt', 'zsre_smoke_plan', 'zsre_launch_profile', 'checkpoint_only_profile')
    if any(key in manifest for key in forbidden):
        raise ValueError('DISABLED_GPU_PROOF_GATE_STILL_PRESENT')
    return True


def cell(role):
    if role not in ROLES:
        raise ValueError('EXACT_USER_REGISTERED_SUBSET')
    if role.startswith('W0_'):
        return role[3:].lower(), 'MEMIT', 'w0'
    dataset, method = role.split('_', 1)
    return dataset.lower(), method, 'chain'


def check_mode(manifest, mode):
    if enabled(manifest) and mode not in ('w0', 'chain'):
        raise ValueError('GPU_QUALIFICATION_USER_DISABLED')
