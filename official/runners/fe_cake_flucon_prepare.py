"""CPU-only preparation of FE/CAKE saved-W20 Flu/Con configs on the evaluating host.

Each checkpoint is bound to its own run records before any GPU work: file SHA-256
against the recorded allowlist below (from the run's pointer/COMPLETE/commit/B20
receipt, as published under audits/), payload identity, cursor and request order,
and the SHA-256 of every selected weight. 63382 must also equal the FP32 hashes in
its commit B20; the author FE B20.pt weights must be bit-identical to the native
final save named by the run's final-parity record. Inputs copied to another host
must match the transfer receipt. Model files are re-hashed against the run's own
asset manifest. No model forward, no edit, no checkpoint is changed.
"""
import argparse
import csv
from pathlib import Path

import torch

from official.evaluation.generation.assets import load_assets
from official.experiments.prepare import digest, write_new
from official.runners.server1 import flucon_eval as shared
from official.runners.server1.assets import member as asset_member
from official.runners.server1.common import member, read
from official.runners.server1.fe_history_prepare import runtime
from .fe_cake_flucon import (AUTHORITY, LOCAL, NATIVE, NATIVE_Z, NODES, OFFICIAL, raw_sha, require,
                             selected_names, tracking_values)

BASE = Path('/mnt/raid5/janghj/ODE-edit/local')
S1 = BASE / 'official-baselines/server1'
FE_NATIVE = BASE / 'fe-collapse-repair-20261011'
FE_QWEN_STORAGE = Path('/var/tmp/odeedit-fe-collapse-20261011')
CAKE_S1 = BASE / 'cake-six-2k-server1-20261011'
FE_S3 = Path('/data/janghj/ODE-edit/local/fe-author-llama-cf-s3-20261011/preparation-r3')
CAKE_S4 = Path('/data/janghj/ODE-edit/local/cake-qwen25-cf-server4-20261011')
RUN_ASSETS = S1 / 'memit-fe-history-three-model-2k/preparation-r1'
HOSTS = dict(
    server1=dict(reference=BASE / 'baseline-generation-eval-assets/20261007/reference-ready-r1/manifest.json',
                 stream=S1 / 'streams/cf-stream.json'),
    server2=dict(reference=BASE / 'qwen-baselines-server2-20261009/preparation-r1/generation-local.json',
                 stream=BASE / 'qwen-baseline-mask-cold-rerun-20261010/registration-r1/streams/cf-stream.json'))
STREAM_FILE_SHA = '66edc483a8d4bcadedd479e4c36759a686ad61a38741d8870a9052b795710e37'
ORDER_SHA = '0b912d11659eb087ee71a391b7bea1e02ecc9d999a48254eb8559434965640f4'
REFERENCE_IDENTITY = '75e595c7f26ec334830e9bb9ca6028098c19ea84a9509a5713985847683f8ea6'
WANDB_ENV = '/mnt/raid5/janghj/ODE-edit/servers/local/wandb.env'
HUB = Path('/mnt/raid5/janghj/.cache/huggingface/hub')
PRECISION = dict(
    official='FP32 run: the saved FP32 selected weights are copied exactly into the FP32 evaluation model.',
    native='BF16 author FE main.py run: the saved BF16 selected weights are cast exactly to FP32 (BF16 is a '
           'subset of FP32; round trip verified). Evaluation stays FP32 eager, TF32 off, like every Flu/Con '
           'cell. Non-edited weights are the FP32-loaded base snapshot as in every Flu/Con cell (for GPT-J the '
           'stored snapshot is FP32, so they are not the BF16-rounded values the native run held in memory).')

# Recorded W20 identities (run pointer/COMPLETE/commit/receipt, see `record`).
CANDIDATES = {
    'llama3-memit_fe_history-63382': dict(
        schema=OFFICIAL, model='llama3', method='MEMIT_FE_HISTORY', job_id='63382', source_host='server3',
        record='audits/servers/server3/fe-author-llama-cf-s3-20261011/w20-table-row-63382.json',
        files=dict(checkpoint=(FE_S3, 'runs/cf/checkpoint/batch-20-bbf87bff0e0b36e5.pt'),
                   pointer=(FE_S3, 'runs/cf/checkpoint/latest.json'), terminal=(FE_S3, 'runs/cf/COMPLETE.json'),
                   commit=(FE_S3, 'runs/cf/commits/batch-20.json'), config=(FE_S3, 'configs/cf.json'),
                   run_assets=(FE_S3, 'assets-llama3.json')),
        expected=dict(checkpoint=('bbf87bff0e0b36e50bff84ef61c59aaa1c05dbe16dda9b6165c75129d9350cb6', 5284843587),
                      terminal=('e9bdb8a05756222c36acc8a6260ccd1603812a79ff99ede30d818abafd8b64e6', 4487),
                      config=('5bc0839e47d1c9e5330384c3dc6621d1be568808c1c40116fc9cb3d91ba9b985', 3766))),
    'gptj-fe_author_native-63389': dict(
        schema=NATIVE, model='gptj', method='FE_AUTHOR_NATIVE', job_id='63389', evaluation_job_id='63390',
        source_host='server1',
        record='audits/global/fe-collapse-repair-20261011/native-comparison.json (arm gptj_repaired), '
               'native-gptj-repaired-b20-final-parity.json, native-gptj-repaired-common-complete.json',
        files=dict(checkpoint=(FE_NATIVE, 'native-causal-r1/runs/gptj_repaired/checkpoints/B20.pt'),
                   pointer=(FE_NATIVE, 'native-causal-r1/runs/gptj_repaired/checkpoints/B20.json'),
                   terminal=(FE_NATIVE, 'native-causal-r1/evaluations/gptj_repaired/COMPLETE.json'),
                   config=(FE_NATIVE, 'native-causal-r1/configs/gptj_repaired.json'),
                   native_final=(FE_NATIVE, 'native-causal-r1/runs/gptj_repaired/cache/saved_weights/memit/'
                                 'counterfact_stream_2000-gptj_repaired-EleutherAI-gpt-j-6B.pt'),
                   run_assets=(RUN_ASSETS, 'assets/gptj.json')),
        expected=dict(checkpoint=('6e27548a125c8afd496fd4957b705929818adfd2899b0b9f2a3f472aaf24eb3f', 7247783445),
                      pointer=('1d9982fee8004f42e16c6e153dd439cbee87c8225e782fb061dde53369752768', 378),
                      terminal=('36e83ff720968b2811da17acb1dd5094812b3756c4da9729d35a2f73545aa8be', None),
                      config=('f40227522a23a34839fc785fef034c770ab0788b8395a2a5121fefab7c5d3d6f', 57551),
                      native_final=('c29f1f036618ad10384ef085dd81527f9691fe26888aae509eb56180a2cae707', 805310189))),
    'qwen25-fe_author_native-63385': dict(
        schema=NATIVE, model='qwen25', method='FE_AUTHOR_NATIVE', job_id='63385', evaluation_job_id='63387',
        source_host='server1',
        record='audits/global/fe-collapse-repair-20261011/native-comparison.json (arm qwen_current), '
               'native-qwen-b20-native-final-parity.json, native-qwen-common-evaluation.json, '
               'native-qwen-control-storage-migration.json',
        files=dict(checkpoint=(FE_QWEN_STORAGE, 'qwen-current-checkpoints/B20.pt'),
                   pointer=(FE_NATIVE, 'native-r2/runs/qwen25/checkpoints/B20.json'),
                   terminal=(FE_NATIVE, 'native-eval-r1/results/qwen25/COMPLETE.json'),
                   config=(FE_NATIVE, 'native-r2/configs/qwen25.json'),
                   native_final=(FE_NATIVE, 'native-r2/runs/qwen25/cache/saved_weights/memit/'
                                 'counterfact_stream_2000-native-sequential-qwen25-Qwen-Qwen2.5-7B-Instruct.pt'),
                   run_assets=(RUN_ASSETS, 'assets/qwen25.json')),
        expected=dict(checkpoint=('1c9dad7424428291aa308aee32681241c3bf06dd90ea3dfaf7208483d0d08cce', 7856481561),
                      pointer=('301ce84b560e331ac63b5ac2b8a97e02744c1c1282319066eb811be527e30b86', 364),
                      terminal=('8669fba6c6a351afb75ce28ba738168e169ca41f917184ffc68c6d0ad84bbed9', None),
                      config=('c9578794bf583fef48fcc93700573ef0be74c662d19b54115b976e9fc5c76d49', 49183),
                      native_final=('139248a00e552ff02b2e081070cacfc12356c7a712ec8ee580344380c16f6289', 678957167))),
    'llama3-cake-63296': dict(
        schema=OFFICIAL, model='llama3', method='CAKE', job_id='63296', source_host='server1',
        record='audits/servers/server1/cake-six-2k-20261011/w20-table-row-63296.json',
        files=dict(checkpoint=(CAKE_S1, 'runs/llama3-cf/checkpoint/batch-20-ab95a58c41b9a3aa.pt'),
                   pointer=(CAKE_S1, 'runs/llama3-cf/checkpoint/latest.json'),
                   terminal=(CAKE_S1, 'runs/llama3-cf/COMPLETE.json'),
                   commit=(CAKE_S1, 'runs/llama3-cf/commits/batch-20.json'),
                   config=(CAKE_S1, 'preparation/configs/llama3-cf.json'),
                   run_assets=(RUN_ASSETS, 'assets/llama3.json')),
        expected=dict(checkpoint=('ab95a58c41b9a3aa48eaefb0fc429b3a3307b72045c49c98b546a826c18cd778', 5284840965),
                      pointer=('2c8342429a0a28788b46476a5287c5b13d4fe230f641561a97eeb62a6015d775', 237),
                      terminal=('7ff99865887862bcf67ed5031ec6bb55509fbe3cebaff98b39b8a374415a4681', 3040),
                      commit=('36132176c07206c5176183a88075a3c08536cffa964beea2bc9c59410b54094a', 1720))),
    'gptj-cake-63298': dict(
        schema=OFFICIAL, model='gptj', method='CAKE', job_id='63298', source_host='server1',
        record='audits/servers/server1/cake-six-2k-20261011/w20-table-row-63298.json',
        files=dict(checkpoint=(CAKE_S1, 'runs/gptj-cf/checkpoint/batch-20-da58f9b4452488a3.pt'),
                   pointer=(CAKE_S1, 'runs/gptj-cf/checkpoint/latest.json'),
                   terminal=(CAKE_S1, 'runs/gptj-cf/COMPLETE.json'),
                   commit=(CAKE_S1, 'runs/gptj-cf/commits/batch-20.json'),
                   config=(CAKE_S1, 'preparation/configs/gptj-cf.json'),
                   run_assets=(RUN_ASSETS, 'assets/gptj.json')),
        expected=dict(checkpoint=('da58f9b4452488a3aede31f65b6fd0fc97d4df7871c6001d33251345475611e0', 8053082005),
                      pointer=('cffc556e083eff3ae27cfbe4352997f6ce585b6eae138c302d7b2305419639fd', 237),
                      terminal=('f1ec423453685560553167033ec4c81d2839a41489f65fe62c1397153c503962', 3024),
                      commit=('f477431b70ff823db2272f5fa8168c10741aaf84cf892629ac83f1fb466d7386', 1714))),
    'qwen25-cake-63324': dict(
        schema=OFFICIAL, model='qwen25', method='CAKE', job_id='63324', source_host='server4',
        record='audits/servers/server4/cake-qwen25-cf-move-20261011/w20-table-row-63324.json',
        files=dict(checkpoint=(CAKE_S4, 'output/checkpoint/batch-20-be1956ab6d4bcec6.pt'),
                   pointer=(CAKE_S4, 'output/checkpoint/latest.json'), terminal=(CAKE_S4, 'output/COMPLETE.json'),
                   commit=(CAKE_S4, 'output/commits/batch-20.json'), config=(CAKE_S4, 'config.json'),
                   run_assets=(CAKE_S4, 'assets.json')),
        expected=dict(checkpoint=('be1956ab6d4bcec6bf23ff37872ff7733d8e50c2c0a462b0837ae7e09563cdf9', 8535426713),
                      pointer=('1c3434ed9e87b8dd55bf7ecffc52931d346615d271f7a48e3f8fde9f75bdfc41', 237),
                      terminal=('f527f6549bc004044146c3fc2da95b57b888af3ceec13aaaacd7f778f51d2a9d', 2959),
                      commit=('956f0abac7accfc70c9aeb275ad8e50cace9e35a88f5b3fba7ff2e0d27bee14f', 1689),
                      config=('597b9f470e01afdbed83fcbcbf21b7f9d97eb9c5535395b647537f84bb8f43c6', 3140))),
}


def transfer_rows(path):
    with open(path, newline='') as stream:
        rows = list(csv.reader(stream, delimiter='\t'))
    names = ('key', 'host', 'source', 'source_sha256', 'source_bytes', 'local', 'local_sha256', 'status', 'seconds')
    require(rows and all(len(r) == len(names) for r in rows), 'TRANSFER_RECEIPT_SCHEMA')
    return {r[5]: dict(zip(names, r)) for r in rows}


def locate(key, spec, inputs, transfers):
    """Original paths locally, or the copy under inputs/<key>/ bound to its transfer row."""
    paths, provenance = {}, {}
    for role, (base, relative) in spec['files'].items():
        source = Path(base) / relative
        if inputs is None:
            paths[role] = source
            continue
        local = Path(inputs) / key / relative
        row = transfers.get(str(local))
        found = member(local)
        require(row is not None and row['status'] == 'OK' and row['source'] == str(source)
                and row['source_sha256'] == row['local_sha256'] == found['sha256']
                and int(row['source_bytes']) == found['bytes'], 'TRANSFER_RECEIPT_' + role.upper())
        paths[role] = local
        provenance[role] = dict(source_host=row['host'], source_path=row['source'], sha256=found['sha256'],
                                bytes=found['bytes'])
    return paths, provenance


def recorded(spec, role, value):
    if role in spec['expected']:
        sha, size = spec['expected'][role]
        require(value['sha256'] == sha and (size is None or value['bytes'] == size), 'RECORDED_SHA_' + role.upper())
    return value


def stream_records(path):
    stream = member(path)
    require(stream['sha256'] == STREAM_FILE_SHA, 'CF_STREAM_FILE')
    records = read(path)
    records = records['records'] if isinstance(records, dict) else records
    require(len(records) == 2000 and digest([r['case_id'] for r in records]) == ORDER_SHA, 'CF_FIRST2000_ORDER')
    return stream, records


def bind_official(spec, files, records):
    pointer, terminal, commit = read(files['pointer']), read(files['terminal']), read(files['commit'])
    config = read(files['config'])
    identity = terminal['identity']
    checkpoint = recorded(spec, 'checkpoint', member(files['checkpoint']))
    require(pointer == dict(batch=20, file=files['checkpoint'].name, final_W20=True,
                            identity_sha256=digest(identity), sha256=checkpoint['sha256']), 'W20_POINTER')
    require(config['config_sha256'] == identity['config_sha256']
            == digest({k: v for k, v in config.items() if k != 'config_sha256'}), 'RUN_CONFIG_IDENTITY')
    require(member(files['run_assets'])['sha256'] == config['assets']['sha256'], 'RUN_ASSET_MANIFEST')
    require(digest(records) == identity['stream_sha256'] == config['stream_sha256'], 'RUN_STREAM_IDENTITY')
    require(config['method'] == spec['method'] and config['model'] == spec['model'] and config['dataset'] == 'cf',
            'RUN_SCOPE')
    if spec['method'] == 'CAKE':
        final = terminal['final_cursor']
        require(terminal['requests'] == 2000 and terminal['batches'] == 20
                and terminal['generation'] == 'DEFERRED_TO_W20_CHECKPOINT'
                and final['batch'] == 20 and final['edits'] == 2000 and final['checkpoint']['latest'] == pointer
                and terminal['checkpoint']['sha256'] == member(files['pointer'])['sha256'], 'CAKE_TERMINAL')
        require(commit['identity'] == identity and commit['cursor'] == final, 'CAKE_COMMIT_B20')
        expected_cursor = {k: final[k] for k in ('batch', 'edits', 'request_sha256', 'current', 'all_seen')}
        recorded_weights = None
    else:
        final = terminal['final_cursor']
        require(terminal['method'] == spec['method'] and terminal['dataset'] == 'cf'
                and terminal['native_apply_calls'] == 20 and terminal['requests'] == 2000
                and final['completed_batch'] == 20 and final['generation_status'] == 'DEFERRED_TO_SAVED_W20_CHECKPOINT',
                'FE_TERMINAL')
        require(commit['batch'] == 20 and commit['checkpoint'] == pointer and commit['identity'] == identity
                and commit['cursor'] == final, 'FE_COMMIT_B20')
        require(config['ordered_case_ids_sha256'] == ORDER_SHA, 'FE_RUN_ORDER')
        expected_cursor = final
        recorded_weights = final['edit']['after']['selected_weights']
    hparams = config['hparams']
    payload = torch.load(files['checkpoint'], map_location='cpu', mmap=True, weights_only=False)
    require(payload['schema'] == OFFICIAL and payload['batch'] == 20 and payload['identity'] == identity
            and payload['method'] == spec['method'] and payload['evaluation_cursor'] == expected_cursor
            and set(payload['cache_c']) == {str(layer) for layer in hparams['layers']}, 'OFFICIAL_PAYLOAD')
    names = selected_names(hparams)
    require(set(payload['weights']) == names, 'OFFICIAL_SELECTED_WEIGHTS')
    hashes = {}
    for name in sorted(names):
        value = payload['weights'][name]
        require(value.dtype == torch.float32 and bool(torch.isfinite(value).all()), 'OFFICIAL_FP32_FINITE')
        hashes[name] = raw_sha(value)
    if recorded_weights is not None:
        require(hashes == recorded_weights, 'PAYLOAD_NOT_RECORDED_W20')
    original = dict(job_id=spec['job_id'], method=spec['method'], checkpoint_schema=OFFICIAL, identity=identity,
        checkpoint=checkpoint, pointer=member(files['pointer']), terminal=recorded(spec, 'terminal', member(files['terminal'])),
        config=recorded(spec, 'config', member(files['config'])), commit=recorded(spec, 'commit', member(files['commit'])),
        hparams=hparams, ordered_case_ids_sha256=ORDER_SHA, expected_history=True, payload_tensor_sha256=hashes,
        recorded_selected_weights_sha256=recorded_weights)
    tokenizer = identity['tokenizer_sha256']
    return original, read(files['run_assets']), tokenizer, identity['model_revision']


def bind_native(spec, files, records):
    config, receipt, terminal = read(files['config']), read(files['pointer']), read(files['terminal'])
    checkpoint = recorded(spec, 'checkpoint', member(files['checkpoint']))
    pointer = recorded(spec, 'pointer', member(files['pointer']))
    native_identity = config['identity_sha256']
    require(digest({k: v for k, v in config.items() if k != 'identity_sha256'}) == native_identity
            and config['model'] == spec['model'] and config['requests'] == 2000
            and config['current_batch_targets_only'] is True and config['model_dtype'] == 'bfloat16'
            and config['original_stream']['sha256'] == STREAM_FILE_SHA, 'NATIVE_RUN_CONFIG')
    require({k: receipt[k] for k in ('sha256', 'bytes', 'completed_batches', 'completed_edits', 'identity_sha256')}
            == dict(sha256=checkpoint['sha256'], bytes=checkpoint['bytes'], completed_batches=20,
                    completed_edits=2000, identity_sha256=native_identity), 'NATIVE_B20_RECEIPT')
    require(terminal['native_config'] == native_identity and terminal['W20_available'] is True
            and terminal['results']['20']['checkpoint'] == receipt, 'NATIVE_COMMON_EVALUATION_TERMINAL')
    require(member(files['run_assets'])['sha256'] == config['model_assets']['sha256'], 'RUN_ASSET_MANIFEST')
    z_policy = NATIVE_Z[config.get('z_reference', 'batch_entry')]
    payload = torch.load(files['checkpoint'], map_location='cpu', mmap=True, weights_only=False)
    native = payload['config']
    require(payload['schema'] == NATIVE and payload['identity_sha256'] == native_identity
            and payload['completed_batches'] == 20 and payload['completed_edits'] == 2000
            and payload['z_policy'] == z_policy
            and payload['seen_case_ids'] == [r['case_id'] for r in records], 'NATIVE_PAYLOAD')
    require(native['model_dtype'] == 'bfloat16' and native['bs'] == 100 and native['num_edits'] == 2000
            and native['algs']['add_old_keys'] is True and native['diagnostic_identity_sha256'] == native_identity,
            'NATIVE_PAYLOAD_CONFIG')
    hparams = dict(rewrite_module_tmp=native['llms']['rewrite_module_tmp'], layers=native['llms']['layers'],
                   source='B20.pt config.llms (author FE hparams)')
    names = selected_names(hparams)
    require(set(payload['weights']) == names, 'NATIVE_SELECTED_WEIGHTS')
    final = torch.load(files['native_final'], map_location='cpu', mmap=True, weights_only=False)
    final_member = recorded(spec, 'native_final', member(files['native_final']))
    require(set(final) == names, 'NATIVE_FINAL_WEIGHT_SET')
    hashes = {}
    for name in sorted(names):
        value = payload['weights'][name]
        require(value.dtype == torch.bfloat16 and bool(torch.isfinite(value).all())
                and final[name].dtype == torch.bfloat16 and torch.equal(final[name], value), 'NATIVE_FINAL_BIT_IDENTITY')
        hashes[name] = raw_sha(value)
    assets = read(files['run_assets'])
    identity = dict(schema='native-fe-consumer-binding-not-official-identity', native_identity_sha256=native_identity,
        z_policy=z_policy, native_config_sha256=recorded(spec, 'config', member(files['config']))['sha256'],
        author_commit=config['author_commit'], ode_patch_source=config['ode_patch_source'],
        covariance_repair=config['covariance_repair'], model_revision=assets['model']['identity']['revision'],
        tokenizer_sha256=assets['tokenizer_sha256'], checkpoint_sha256=checkpoint['sha256'])
    original = dict(job_id=spec['job_id'], evaluation_job_id=spec['evaluation_job_id'], method=spec['method'],
        checkpoint_schema=NATIVE, identity=identity, checkpoint=checkpoint, pointer=pointer,
        terminal=recorded(spec, 'terminal', member(files['terminal'])), config=member(files['config']),
        native_final=final_member, hparams=hparams, ordered_case_ids_sha256=ORDER_SHA, expected_history=True,
        payload_tensor_sha256=hashes, recorded_selected_weights_sha256=None)
    return original, assets, identity['tokenizer_sha256'], identity['model_revision']


def normalize_assets(output, model, manifests, revision, tokenizer):
    """Full SHA of this host's snapshot members against the run's own asset manifest."""
    signature = lambda a: {Path(m['path']).name: (m['sha256'], m['bytes']) for m in a['model']['members']}
    first = manifests[0]
    require(all(signature(a) == signature(first) and a['model']['identity']['revision'] == revision
                and a['tokenizer_sha256'] == tokenizer for a in manifests), 'RUN_ASSET_MANIFESTS_DIFFER')
    snapshot = HUB / ('models--' + first['model']['identity']['model_id'].replace('/', '--')) / 'snapshots' / revision
    members = []
    for name, (sha, size) in sorted(signature(first).items()):
        item = asset_member(snapshot / name, allow_symlink=True, expected_sha=sha, expected_bytes=size)
        item['path'] = str(snapshot / name)
        members.append(item)
    path = output / 'assets' / f'{model}.json'
    write_new(path, dict(model=dict(snapshot=str(snapshot), members=members, identity=first['model']['identity']),
                         tokenizer_sha256=tokenizer, run_manifest_signature=signature(first)))
    return member(path)


def prepare(server, output, keys, inputs=None, transfer=None):
    require(server in NODES and keys and set(keys) <= set(CANDIDATES), 'PREPARATION_SCOPE')
    output = Path(output).absolute()
    require(output.is_relative_to(LOCAL) and not output.exists(), 'NEW_OUTPUT_UNDER_TASK')
    require((inputs is None) == (transfer is None), 'INPUTS_AND_TRANSFER_TOGETHER')
    transfers = transfer_rows(transfer) if transfer else {}
    host = HOSTS[server]
    refs = load_assets(host['reference'])
    require(refs.sha == REFERENCE_IDENTITY, 'REFERENCE_IDENTITY')
    stream, records = stream_records(host['stream'])
    bound, manifests = {}, {}
    for key in keys:
        spec = CANDIDATES[key]
        files, provenance = locate(key, spec, inputs, transfers)
        binder = bind_official if spec['schema'] == OFFICIAL else bind_native
        original, assets, tokenizer, revision = binder(spec, files, records)
        original.update(record=spec['record'], source_host=spec['source_host'], transfer=provenance or None)
        bound[key] = original
        manifests.setdefault(spec['model'], []).append((assets, revision, tokenizer))
    assets = {}
    for model, rows in manifests.items():
        require(len({(r[1], r[2]) for r in rows}) == 1, 'MODEL_REVISION_OR_TOKENIZER_DIFFER')
        assets[model] = normalize_assets(output, model, [r[0] for r in rows], rows[0][1], rows[0][2])
    configs, inventory = [], []
    evaluator = member(Path(shared.__file__).resolve().parents[2] / 'evaluation/generation/native_observer.py')
    for key in keys:
        spec, original = CANDIDATES[key], bound[key]
        c = dict(instruction=shared.INSTRUCTION, registration_authority=AUTHORITY, server=server, key=key,
            model=spec['model'], original=original, assets=assets[spec['model']], asset_schema='history',
            stream=stream, reference=member(host['reference']), reference_identity=refs.sha, evaluator=evaluator,
            runtime=runtime(), output=str(output / 'runs' / key), tracking_env_file=WANDB_ENV,
            precision=dict(evaluation='FP32 eager, TF32 off (shared evaluator, unchanged)',
                           stored_selected_weights='bfloat16' if spec['schema'] == NATIVE else 'float32',
                           cast='exact BF16->FP32' if spec['schema'] == NATIVE else 'none',
                           note=PRECISION['native' if spec['schema'] == NATIVE else 'official']))
        c['config_sha256'] = digest(c)
        tracking_values(c, 'a' * 40)
        path = output / 'configs' / f'{key}.json'
        write_new(path, c)
        configs.append(member(path))
        inventory.append(dict(key=key, job_id=spec['job_id'], checkpoint=original['checkpoint'],
                              payload_tensor_sha256=original['payload_tensor_sha256'], config=configs[-1],
                              status='BOUND_TO_RUN_RECORDS'))
        print(key, original['checkpoint']['sha256'], 'BOUND', flush=True)
    result = dict(instruction=shared.INSTRUCTION, registration_authority=AUTHORITY, server=server,
        configs=configs, inventory=inventory, reference=member(host['reference']),
        transfer_receipt=member(transfer) if transfer else None, no_model_forward=True, no_edit=True,
        checkpoints_read_only=True)
    write_new(output / 'preparation.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--server', required=True, choices=sorted(NODES)); p.add_argument('--output', required=True)
    p.add_argument('--keys', required=True); p.add_argument('--inputs'); p.add_argument('--transfer')
    a = p.parse_args()
    prepare(a.server, a.output, a.keys.split(','), a.inputs, a.transfer)
