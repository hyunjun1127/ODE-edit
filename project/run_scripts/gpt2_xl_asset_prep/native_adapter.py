"""Frozen native layer_stats loader: exact hparams argument, no editor/apply imports."""
import importlib
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
from .common import require

def cache_only(plan,layer):
    native=Path(plan['source_snapshot'])
    for name in ('easyeditor','easyeditor.util','easyeditor.models','easyeditor.models.rome'):
        if name not in sys.modules:
            m=ModuleType(name);m.__path__=[str(native/Path(*name.split('.')))];sys.modules[name]=m
    ls=importlib.import_module('easyeditor.models.rome.layer_stats')
    def forbidden(*a,**k):raise RuntimeError('REUSE_MUST_NOT_LOAD_DATASET_OR_MODEL')
    ls.load_dataset=forbidden
    row=next(x for x in plan['stats'] if x['layer']==layer)
    path=Path(row['asset']['path']);require(path.is_file(),'CACHE_MISSING')
    fake=SimpleNamespace(config=SimpleNamespace(n_positions=1024,_name_or_path='gpt2-xl'))
    hp=SimpleNamespace(device=0)
    # stats_dir -> gpt2-xl/wikipedia_stats/exact native basename, original CombinedStat schema.
    result=ls.layer_stats(fake,None,f'transformer.h.{layer}.mlp.c_proj',path.parents[2],
        'wikipedia',['mom2'],model_name='gpt2-xl',sample_size=100000,precision='float32',
        batch_tokens=3072,hparams=hp,progress=lambda loader,**kw:loader)
    require(result.mom2.count==row['recorded_count'],'NATIVE_COUNT')
    return result.mom2.moment(),result.mom2.count
