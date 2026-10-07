"""Exact native Llama source/parameter bindings; metadata only, no model load.

EasyEdit supplies stock MEMIT/AlphaEdit. Pinned local CAKE and BLUE supply
CAKE, AlphaEdit-BLUE, PRUNE and RECT. The native parsers and numerical bodies
are never replaced with PRICE math. Large assets use prior SHA plus fresh stat.
"""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess


METHODS = ('MEMIT', 'PRUNE', 'RECT', 'ALPHAEDIT', 'ALPHAEDIT_BLUE', 'CAKE')
EASYEDIT = Path('/data/janghj/EasyEdit/easyeditor')
CAKE = Path('/data/janghj/ODE-edit/local/cake-native-lifelong/20260915-v1/attempt-v1/upstream/CAKE')
BLUE = Path('/data/janghj/ODE-edit/local/blue-alphaedit-sequential-comparison/attempt-v1/blue-source')
CAKE_COMMIT = 'c8243e1d7e43ca9cf64d552f96221fcb9561aac2'
BLUE_COMMIT = '311b076a92e4ed0f14f5c8b4909732da781bc5f7'
STATS_ROOT = Path('/data/janghj/EasyEdit/examples/data/stats')
ASSET_RECEIPT = Path('/data/janghj/ODE-edit/local/jlz-price-cap-base-repair-2k/method-metrics-20261007/config.json')
PROJECTOR = Path('/data/janghj/EasyEdit/examples/null_space_project_Meta-Llama-3-8B-Instruct.pt')
PROJECTOR_SHA = '6d356468c6408dca694c1907ffde31cddb99f7e67fa2e74502910d5afbede5ec'
PROJECTOR_BYTES = 4110419877
PROJECTOR_RECEIPT = Path('/data/janghj/ODE-edit/local/blue-alphaedit-sequential-comparison/attempt-v1/execution-tech-r2/execution.lock.json')
CONTEXT_RECEIPT = Path('/data/janghj/ODE-edit/local/cake-native-lifelong/20260915-v1/attempt-v1/execution.lock.json')
SPECS = {
    'MEMIT': dict(root=EASYEDIT, package='models.memit', main='memit_main',
        parser='MEMITHyperParams', parser_file='memit_hparams', easyedit=True,
        hparams='/data/janghj/EasyEdit/hparams/MEMIT/llama3-8b.yaml', layers=[4, 5, 6, 7, 8]),
    'ALPHAEDIT': dict(root=EASYEDIT, package='models.alphaedit', main='AlphaEdit_main',
        parser='AlphaEditHyperParams', parser_file='AlphaEdit_hparams', easyedit=True,
        hparams='/data/janghj/EasyEdit/hparams/AlphaEdit/llama3-8b.yaml', layers=[4, 5, 6, 7, 8]),
    'CAKE': dict(root=CAKE, commit=CAKE_COMMIT, package='Cake', main='Cake_main',
        parser='CakeHyperParams', parser_file='Cake_hparams', easyedit=False,
        hparams=str(CAKE / 'hparams/Cake/Llama3-8B.json'), layers=[4, 5, 6, 7, 8]),
    'ALPHAEDIT_BLUE': dict(root=BLUE, commit=BLUE_COMMIT, package='AlphaEdit', main='AlphaEdit_main',
        parser='AlphaEditHyperParams', parser_file='AlphaEdit_hparams', easyedit=False,
        hparams=str(BLUE / 'hparams/AlphaEdit/Llama3-8B-blue.json'), layers=[4, 8]),
    'PRUNE': dict(root=BLUE, commit=BLUE_COMMIT, package='memit', main='memit_main',
        parser='MEMITHyperParams', parser_file='memit_hparams', easyedit=False,
        hparams=str(BLUE / 'hparams/MEMIT/Llama3-8B.json'), layers=[4, 5, 6, 7, 8]),
    'RECT': dict(root=BLUE, commit=BLUE_COMMIT, package='memit', main='memit_rect_main',
        parser='MEMITHyperParams', parser_file='memit_hparams', easyedit=False,
        hparams=str(BLUE / 'hparams/MEMIT/Llama3-8B.json'), layers=[4, 5, 6, 7, 8]),
}
COMMON = ('util/generate.py', 'util/globals.py', 'util/hparams.py',
          'util/logit_lens.py', 'util/nethook.py', 'util/runningstats.py')
SCIENTIFIC_FIELDS = ('layers', 'clamp_norm_factor', 'layer_selection', 'fact_token',
    'v_num_grad_steps', 'v_lr', 'v_loss_layer', 'v_weight_decay', 'kl_factor',
    'mom2_adjustment', 'mom2_update_weight', 'rewrite_module_tmp', 'layer_module_tmp',
    'mlp_module_tmp', 'attn_module_tmp', 'ln_f_module', 'lm_head_module',
    'mom2_dataset', 'mom2_n_samples', 'mom2_dtype', 'nullspace_threshold', 'L2',
    'blue', 'edit_layer', 'temperature', 'causal_scores')


def require(condition, label):
    if not condition:
        raise RuntimeError(label)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def member(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'NATIVE_SMALL_MEMBER_REGULAR:' + str(path))
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha(path))


def verify(row):
    path = Path(row['path'])
    require(path.is_file() and not path.is_symlink() and path.stat().st_size == row['bytes']
            and sha(path) == row['sha256'], 'NATIVE_SMALL_MEMBER_CHANGED:' + str(path))
    return path.resolve()


def asset_stat(row):
    """Do not rehash protected multi-GiB C0/projector during source review."""
    path = Path(row['path'])
    require(path.is_file() and not path.is_symlink() and path.stat().st_size == row['bytes']
            and len(row['sha256']) == 64, 'NATIVE_LARGE_ASSET_STAT:' + str(path))
    stat = path.stat()
    for key, actual in [('inode', stat.st_ino), ('mtime_ns', stat.st_mtime_ns)]:
        require(key not in row or row[key] == actual, 'NATIVE_LARGE_ASSET_CHANGED:' + key)
    return dict(row, path=str(path.resolve()), inode=stat.st_ino, mtime_ns=stat.st_mtime_ns)


def source_files(method):
    spec = SPECS[method]
    base = spec['package'].replace('.', '/')
    rome = 'models/rome' if spec['easyedit'] else 'rome'
    return (f'{base}/{spec["main"]}.py', f'{base}/compute_z.py', f'{base}/compute_ks.py',
        f'{base}/{spec["parser_file"]}.py', f'{rome}/layer_stats.py',
        f'{rome}/repr_tools.py', f'{rome}/tok_dataset.py', *COMMON)


def effective_source(relative, original, namespace):
    """Import/path plumbing only. Numerical function ASTs remain unchanged."""
    text = original.decode()
    for package in ('util', 'rome'):
        text = text.replace(f'from {package} ', f'from {namespace}.{package} ')
        text = text.replace(f'from {package}.', f'from {namespace}.{package}.')
    if relative == 'Cake/Cake_main.py':
        require(text.count('from notebooks.util import hparams\n') == 1, 'CAKE_UNUSED_IMPORT_EXACT')
        text = text.replace('from notebooks.util import hparams\n', '')
    if relative == 'util/globals.py' and 'with open("globals.yml", "r") as stream:' in text:
        require(text.count('with open("globals.yml", "r") as stream:') == 1, 'GLOBALS_PATH_EXACT')
        text = text.replace('with open("globals.yml", "r") as stream:',
            'with open(Path(__file__).resolve().parents[1] / "globals.yml", "r") as stream:')
        require(text.count('    Path(z)\n') == 1, 'GLOBALS_RELATIVE_PATH_EXACT')
        text = text.replace('    Path(z)\n',
            '    (Path(z) if Path(z).is_absolute() else Path(__file__).resolve().parents[1] / z)\n')
    return text


def native_function_ast(path, name):
    node = next(node for node in ast.parse(Path(path).read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == name)
    return ast.dump(node, include_attributes=False)


def build_bindings(config=None):
    """Return c['native']; root caller seals this exact mapping in its lock.

    Optional native_asset_receipt/native_projector_receipt/native_context_receipt
    select existing small provenance receipts, not new assets. Context reuse is
    the preserved historical native-input cache, not a new native GPU PASS.
    """
    import yaml
    config = config or {}
    prior_path = Path(config.get('native_asset_receipt', ASSET_RECEIPT))
    prior = json.loads(prior_path.read_text())
    assets = {Path(row['path']).name: row for row in prior['assets']}
    p_receipt = Path(config.get('native_projector_receipt', PROJECTOR_RECEIPT))
    old_projector = next(row for row in json.loads(p_receipt.read_text())['members']
                         if row['path'] == str(PROJECTOR))
    require(old_projector['sha256'] == PROJECTOR_SHA and old_projector['bytes'] == PROJECTOR_BYTES,
            'NATIVE_PRIOR_PROJECTOR_SHA')
    projector = asset_stat(old_projector)
    ctx_receipt = Path(config.get('native_context_receipt', CONTEXT_RECEIPT))
    ctx = json.loads(ctx_receipt.read_text())['contexts']
    if config.get('native_context_path'):
        selected = member(config['native_context_path'])
        require((selected['bytes'], selected['sha256']) == (ctx['bytes'], ctx['sha256']),
                'NATIVE_SELECTED_CONTEXT_EQUALS_PRESERVED_INPUT')
        ctx = selected
    contexts = json.loads(verify(ctx).read_text())
    require(isinstance(contexts, list) and len(contexts) == 2 and contexts[0] == ['{}']
            and len(contexts[1]) == 5 and all(type(t) is str for group in contexts for t in group),
            'NATIVE_CONTEXT_CACHE_SCHEMA')
    bindings = {}
    for method, spec in SPECS.items():
        root = spec['root']
        if 'commit' in spec:
            head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], check=True,
                                  capture_output=True, text=True).stdout.strip()
            require(head == spec['commit'], 'NATIVE_PINNED_COMMIT:' + method)
        else:
            head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], check=True,
                                  capture_output=True, text=True).stdout.strip()
        hp = yaml.safe_load(Path(spec['hparams']).read_text())
        # EasyEdit's original parser converts YAML scientific-notation strings.
        for key, value in list(hp.items()):
            if isinstance(value, str) and re.fullmatch(r'[+-]?\d+(?:\.\d*)?[eE][+-]?\d+', value):
                hp[key] = float(value)
        require(hp['layers'] == spec['layers'] and hp['v_lr'] == .1
                and hp['v_num_grad_steps'] == 25 and hp['v_loss_layer'] == 31,
                'NATIVE_STOCK_LLAMA_HPARAMS:' + method)
        files = [dict(relative=relative, **member(root / relative)) for relative in source_files(method)]
        if not spec['easyedit']:
            files.append(dict(relative='globals.yml', **member(root / 'globals.yml')))
        stats = []
        for layer in spec['layers']:
            name = f'model.layers.{layer}.mlp.down_proj_float32_mom2_100000.npz'
            require(name in assets, 'NATIVE_C0_PRIOR_MISSING:' + name)
            stats.append(dict(layer=layer, **asset_stat(assets[name])))
        binding = dict(method=method, root=str(root), source_commit=head,
            namespace='_llama_native_' + method.lower(), package=spec['package'], main=spec['main'],
            parser=spec['parser'], easyedit=spec['easyedit'], files=files,
            hparams=member(spec['hparams']), layers=spec['layers'],
            scientific_fields={key: hp[key] for key in SCIENTIFIC_FIELDS if key in hp},
            stats_root=str(STATS_ROOT), stats_model_name='Meta-Llama-3-8B-Instruct', stats=stats,
            asset_receipt=member(prior_path), context_reuse=dict(contexts=ctx,
                receipt=member(ctx_receipt), source='PRIOR_NATIVE_INPUT_CACHE',
                cross_native_generator_bitwise_parity_claim=False),
            native_has_history=method in ('ALPHAEDIT', 'ALPHAEDIT_BLUE', 'CAKE'),
            native_z_disk_cache=False, checkpoint_saved=False)
        if binding['native_has_history']:
            binding.update(projector=dict(projector, physical_layers=[4, 5, 6, 7, 8], cutoff=.02),
                projector_receipt=member(p_receipt),
                projector_slots=[layer - 4 for layer in spec['layers']])
        if method == 'PRUNE':
            terminal_text = (root / 'experiments/evaluate.py').read_text()
            require(terminal_text.count('adjusted_weight = original_weight + upd_matrix[k]') == 1
                and 'torch.log(S_upd) - torch.log(torch.tensor(max_sigma, device=\'cuda\')) + max_sigma' in terminal_text,
                'PRUNE_ORIGINAL_TERMINAL_BASE_AND_FORMULA')
            binding.update(terminal_source=member(root / 'experiments/evaluate.py'),
                explicit_repair='PRUNE_TERMINAL_BASE_FIX')
        bindings[method] = binding
    return bindings
