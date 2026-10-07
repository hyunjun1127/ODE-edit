"""Small immutable local evidence helpers; never contains model tensors."""
import hashlib
import json
import os
from pathlib import Path
import uuid

SCHEMA = 'counterfact-cake-generation-metrics-v1'
PROFILE = 'cf-cake-prompt-inclusive-total100-eos-corrected-v1'
EVAL_SEED = 20261007
GENERATION_METRIC_SCHEMA = SCHEMA
GENERATION_PROFILE = PROFILE


class GenerationError(RuntimeError):
    """Typed technical failure, as distinct from unavailable metric values."""


def require(ok, reason):
    if not ok:
        raise GenerationError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def immutable_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()
    if path.exists():
        require(path.read_bytes() == data, 'IMMUTABLE_GENERATION_IDENTITY_CONFLICT')
        return
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    try:
        with temporary.open('xb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            require(path.read_bytes() == data, 'IMMUTABLE_GENERATION_IDENTITY_CONFLICT')
    finally:
        if temporary.exists():
            temporary.unlink()


def case_seed(model_identity, occurrence, prompt_index, eval_seed=EVAL_SEED):
    """Stable across arms/jobs; ordered occurrences are deliberately not deduped."""
    forbidden = {'arm', 'baseline', 'job_id', 'job_display_id', 'array_job_id',
                 'array_task_id', 'step_id', 'run_id', 'attempt'}
    def validate(value):
        if isinstance(value, dict):
            require(not set(value) & forbidden, 'GENERATION_SEED_ARM_JOB_IDENTITY_FORBIDDEN')
            for item in value.values():
                validate(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                validate(item)
    validate(model_identity)
    require(isinstance(occurrence, int) and not isinstance(occurrence, bool) and occurrence >= 0,
            'GENERATION_OCCURRENCE_IDENTITY')
    require(isinstance(prompt_index, int) and prompt_index >= 0, 'GENERATION_PROMPT_INDEX')
    key = dict(model_identity=model_identity, ordered_occurrence=occurrence,
               prompt_index=prompt_index, eval_seed=eval_seed)
    return int(digest(key)[:16], 16) % (2**63 - 1)
