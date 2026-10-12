"""Prepare small native source/config inputs, without running a model.

SH1 supplies and validates the final scientific binding schema. This is an
input bundle, not a fake execution/READY or a second scientific implementation.
"""
import argparse
import json
import shutil
from pathlib import Path
from omegaconf import OmegaConf
from official.diagnostics.fe_qwen_current.patch import native_source
from .fe_fp32_host_prepare import sha


def member(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--author', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--stream', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    pins = dict(algs=('configs/algs/memit.yaml','4799ac1df4e726e3dd2abcc54201e9f87f1c76cb35680059c0684806df8809d3'),
                llms=('configs/llms/qwen2.5-7b.yaml','0e1481d06d530746b13aea6df5dec2a4449aef8928120a2a12c4d7da72b50183'))
    for rel, digest in pins.values():
        if sha(a.author/rel) != digest:
            raise ValueError('pinned author YAML mismatch')
    original = a.author/'algs/memit/FE-memit_main.py'
    patched = native_source(original.read_text())  # checks original SHA and writer AST
    dst = a.out/'native'
    copied = []
    for path in sorted(a.author.rglob('*')):
        rel = path.relative_to(a.author)
        if any(part.startswith('.') or part == '__pycache__' for part in rel.parts):
            continue
        if not path.is_file() or path.is_symlink() or path.suffix not in ('.py','.yaml','.yml'):
            continue
        target = dst/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path,target)
        copied.append(str(rel))
    (dst/'algs/memit/memit_main.py').write_text(patched)
    sequential = Path(__file__).resolve().parents[2]/'diagnostics/fe_native/sequential_z.py'
    shutil.copyfile(sequential,dst/'algs/memit/native_sequential_z.py')
    profiles = {k:OmegaConf.to_container(OmegaConf.load(dst/rel),resolve=True) for k,(rel,_) in pins.items()}
    cfg = OmegaConf.to_container(OmegaConf.load(dst/'configs/config.yaml'),resolve=True)
    cfg.pop('defaults',None)
    cfg.update(profiles)
    cfg.update(model_dtype='float32',bs=100,num_edits=2000,seed=0)
    for key in ('cache_dir','h_cache_dir','zs_cache_dir','data_dir','results_dir','cov_dir'):
        cfg[key] = str(a.out/key)
    cfgpath = a.out/'native-config.json'
    cfgpath.write_text(json.dumps(cfg,indent=2))
    assets = json.loads(a.assets.read_text())
    c0 = []
    cache = Path(cfg['cache_dir'])/'stats/Qwen-Qwen2.5-7B-Instruct'
    cache.mkdir(parents=True)
    for layer, value in sorted(assets['C0'].items()):
        link = cache/f'layer-{layer}.npz'
        link.symlink_to(value['member']['path'])
        c0.append(dict(layer=int(layer),original=value['member'],link=str(link)))
    result = dict(status='HOST_INPUTS_PREPARED_FINAL_CORE_BINDING_PENDING',
                  scientific_core_owner='SH1',dataset='cf',new_job_ids=[],
                  native=dict(author_commit='478134dfb24b43f4e18b47e8500893ce3f9cc50f',path=str(dst),
                              original_fe=member(dst/'algs/memit/FE-memit_main.py'),
                              members=[member(x) for x in sorted(dst.rglob('*')) if x.is_file()]),
                  author_yaml={k:member(dst/rel) for k,(rel,_) in pins.items()},
                  native_config=member(cfgpath),main_stream=member(a.stream),
                  model=dict(members=assets['model']['members'],identity=assets['model']['identity']),
                  C0=c0,source_copy_extensions=['.py','.yaml','.yml'],model_copies=0,
                  writer_AST_unchanged=True,solver_dtype_rewrites=0)
    (a.out/'host-inputs.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(status=result['status'],native_members=len(result['native']['members']),
                          output=str(a.out/'host-inputs.json'))))


if __name__ == '__main__':
    main()
