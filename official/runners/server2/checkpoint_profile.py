"""Explicit future CF profile; generation is deferred, never reported as zero.

This is a caller schedule change only. Native methods/hparams, precision,
factual observations and the official checkpoint serializer are unchanged.
"""
INSTRUCTION = 'USER-SH2-OFFICIAL-CF-CHECKPOINT-ONLY-20261009-R1'
SCHEDULE = 'DEFERRED_CHECKPOINT_EVALUATION'
METHODS = ('FT', 'MEMIT', 'ALPHAEDIT', 'ALPHAEDIT_BLUE', 'MEMIT_FE', 'SPHERE')


def deferred(manifest):
    value = manifest.get('checkpoint_only_profile')
    if value is None:
        return False
    expected = dict(instruction_id=INSTRUCTION, dataset='cf', requests=2000,
        batch_size=100, batches=20, methods=list(METHODS), project_gpu_cap=3,
        generation_schedule=SCHEDULE, generation_W0=False, generation_W20=False,
        final_checkpoint_required=True, evaluation_consumer_pending=True,
        archive_delete_allowed=False)
    if value != expected:
        raise ValueError('CHECKPOINT_ONLY_EXACT_USER_PROFILE_REQUIRED')
    return True


def profile():
    value = dict(instruction_id=INSTRUCTION, dataset='cf', requests=2000,
        batch_size=100, batches=20, methods=list(METHODS), project_gpu_cap=3,
        generation_schedule=SCHEDULE, generation_W0=False, generation_W20=False,
        final_checkpoint_required=True, evaluation_consumer_pending=True,
        archive_delete_allowed=False)
    deferred({'checkpoint_only_profile': value})
    return value
