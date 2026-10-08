"""One immutable, content-addressed PRICE configuration resolver."""
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType

ROOT = Path(__file__).resolve().parents[1] / 'hparams' / 'ours'
MODELS = ('llama3', 'qwen25', 'gptj')
PRICE_KEYS = frozenset(('beta_base', 'c', 'beta_max_scale', 'cap_mode', 'n_exp',
    'K_grace', 'tau_F', 'lr', 'eps', 'betas', 'lambda_N', 'lambda_KL', 'max_updates'))


def plain(value):
    if isinstance(value, Mapping): return {k: plain(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [plain(v) for v in value]
    return value


def canonical(value):
    return json.dumps(plain(value),sort_keys=True,separators=(',', ':'),ensure_ascii=False,allow_nan=False).encode()


def freeze(value):
    if isinstance(value, dict): return MappingProxyType({k:freeze(v) for k,v in value.items()})
    if isinstance(value, list): return tuple(freeze(v) for v in value)
    return value


def validate_price(p):
    if set(p) != PRICE_KEYS: raise ValueError('PRICE_KEYS')
    for name in PRICE_KEYS - {'cap_mode','betas','c'}:
        if type(p[name]) not in (int,float) or not math.isfinite(p[name]):
            raise ValueError('PRICE_FINITE:'+name)
    for name in ('n_exp','K_grace','max_updates'):
        if type(p[name]) is not int: raise ValueError('PRICE_INTEGER:'+name)
    if p['cap_mode'] not in ('native','none'): raise ValueError('PRICE_CAP_MODE')
    if p['cap_mode']=='native':
        if type(p['c']) not in (int,float) or not math.isfinite(p['c']) or p['c']<=0:
            raise ValueError('PRICE_NATIVE_CAP')
    elif p['c'] is not None: raise ValueError('PRICE_UNCAPPED_REQUIRES_NULL')
    if not (p['beta_base']>0 and p['beta_max_scale']>=0 and p['n_exp']>=0
            and 0<=p['K_grace']<p['max_updates'] and p['max_updates']>=1
            and p['tau_F']>0 and p['lr']>0 and p['eps']>0
            and p['lambda_N']>=0 and p['lambda_KL']>=0): raise ValueError('PRICE_DOMAIN')
    if (not isinstance(p['betas'],(list,tuple)) or len(p['betas'])!=2 or
        any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<1 for v in p['betas'])):
        raise ValueError('PRICE_ADAM_BETAS')


def resolve(model, arm=None, *, preset=None, override=None):
    if model not in MODELS: raise ValueError('PRICE_MODEL')
    if preset not in (None,'native'): raise ValueError('PRICE_PRESET')
    if preset and arm: raise ValueError('PRICE_ARM_PRESET_AMBIGUOUS')
    writer=json.loads((ROOT/model/'writer.json').read_text())
    p=json.loads((ROOT/model/'price.json').read_text())
    w=writer['hparams']
    # Baseline file changes never silently change the copied ours writer.
    original=Path(__file__).resolve().parents[2]/writer['source']['path']
    if (hashlib.sha256(original.read_bytes()).hexdigest()!=writer['source']['sha256']
            or json.loads(original.read_text())!=w): raise ValueError('WRITER_SOURCE_COPY_MISMATCH')
    if preset=='native':
        p.update(c=w['clamp_norm_factor'],beta_base=w['clamp_norm_factor'],
                 beta_max_scale=w['clamp_norm_factor'],lr=w['v_lr'],lambda_N=w['v_weight_decay'])
    if arm:
        if type(arm) is not str or Path(arm).name!=arm: raise ValueError('PRICE_ARM_NAME')
        row=json.loads((ROOT/'arms'/(arm+'.json')).read_text())
        if set(row)!= {'base','override'} or row['base']!=model: raise ValueError('PRICE_ARM_BASE')
        if not set(row['override'])<=PRICE_KEYS: raise ValueError('PRICE_OVERRIDE_KEYS')
        p.update(row['override'])
    if override is not None:
        if not isinstance(override,dict) or not set(override)<=PRICE_KEYS: raise ValueError('PRICE_OVERRIDE_KEYS')
        p.update(override)
    validate_price(p)
    layers=w['layers']
    if (not layers or any(type(l) is not int or l<0 for l in layers) or layers!=sorted(set(layers))
            or type(w['v_loss_layer']) is not int or w['v_loss_layer']<max(layers)
            or type(w['mom2_update_weight']) not in (int,float)
            or not math.isfinite(w['mom2_update_weight']) or w['mom2_update_weight']<=0):
        raise ValueError('WRITER_DOMAIN')
    result=dict(schema='ours-price-resolved-v1',model=model,arm=arm,preset=preset,
        writer=writer,price=p,eligible_layers=layers,anchor_layer=max(layers),
        nll_layer=w['v_loss_layer'],lambda_C=w['mom2_update_weight'],K_eval=p['max_updates']+1)
    result['sha256']=hashlib.sha256(canonical(result)).hexdigest()
    return freeze(result)


def profile(resolved, runtime):
    """Runtime may add architecture/input settings, never contradict the resolver."""
    value=dict(runtime)
    fields=dict(resolved['price'],eligible_layers=resolved['eligible_layers'],
        anchor_layer=resolved['anchor_layer'],nll_layer=resolved['nll_layer'],
        lambda_C=resolved['lambda_C'],K_eval=resolved['K_eval'])
    fields['beta_max_native_scale']=fields['beta_max_scale']  # legacy receipt alias, derived only
    for k,v in fields.items():
        if k in value and plain(value[k])!=plain(v): raise ValueError('DUPLICATE_PROFILE_KNOB:'+k)
        value[k]=v
    value['resolved_config']=resolved
    value['resolved_config_sha256']=resolved['sha256']
    return freeze(value)


def knobs(p):
    """Every production consumer checks one bound resolver, not fallback defaults."""
    r=p['resolved_config'];expected=r['price']
    if p['resolved_config_sha256']!=r['sha256']: raise ValueError('RESOLVED_CONFIG_BINDING')
    if hashlib.sha256(canonical({k:v for k,v in r.items() if k!='sha256'})).hexdigest()!=r['sha256']:
        raise ValueError('RESOLVED_CONFIG_HASH')
    for k,v in expected.items():
        if plain(p[k])!=plain(v): raise ValueError('RESOLVED_PROFILE_MISMATCH:'+k)
    if p['beta_max_native_scale']!=expected['beta_max_scale']: raise ValueError('SCALE_ALIAS_MISMATCH')
    return expected
