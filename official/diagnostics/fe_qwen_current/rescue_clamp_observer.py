"""Same native observations/storage/quality stop, explicit clamp .5 identity."""
from unittest.mock import patch
from official.diagnostics.fe_qwen_current import rescue_observer as shared
from official.diagnostics.fe_qwen_current.rescue_clamp import SPEC,tracking_config


def snapshot(*args,**kwargs):
    with patch.object(shared,'SPEC',SPEC),patch.object(shared,'tracking_config',tracking_config):
        return shared.snapshot(*args,**kwargs)


def observed_write(*args,**kwargs):
    original=shared.write_new
    def write(path,value):
        value=dict(value,clamp_norm_factor=.5,author_hparam_profile=SPEC['protocol'])
        return original(path,value)
    with patch.object(shared,'SPEC',SPEC),patch.object(shared,'write_new',write):
        return shared.observed_write(*args,**kwargs)


observed_targets=shared.observed_targets
finish=shared.finish
