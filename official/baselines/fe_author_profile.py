"""Explicit author clamp/steps overlay, never changes legacy FE defaults."""
import json
from pathlib import Path
from dataclasses import asdict
from . import registry

INSTRUCTION='USER-SH-FE-AUTHOR-HPARAMS-2K-CF-ZSRE-20261010-R1'
TASK='fe-author-hparams-2k-20261010'
METHOD='MEMIT_FE_HISTORY'
PROFILE=Path(__file__).resolve().parents[1]/'hparams/MEMIT_FE_HISTORY_AUTHOR/profiles.json'

def profile(model):
    if model not in ('llama3','qwen25'):raise ValueError('AUTHOR_MODEL_SCOPE')
    value=json.loads(PROFILE.read_text())[model]
    base=json.loads((PROFILE.parents[1]/'MEMIT_FE'/f'{model}.json').read_text())
    changed={k for k in base if base[k]!=value['hparams'][k]}
    if set(base)!=set(value['hparams']) or changed!={'clamp_norm_factor','v_num_grad_steps'}:
        raise ValueError('AUTHOR_ONLY_CLAMP_STEPS_OVERRIDE')
    return value

def resolve(model,*,device=0):
    values=profile(model)['hparams'];effective=dict(values,device=device)
    # Preserve parser-only defaults (max_length=40, batch_size=1), which are
    # absent from the source JSON and are not scientific override fields.
    expected=asdict(registry.hparams(METHOD,model));expected.update(effective)
    hp=registry.hparams(METHOD,model,overrides=effective)
    if asdict(hp)!=expected:raise ValueError('AUTHOR_RUNTIME_PARSER_MISMATCH')
    return hp
