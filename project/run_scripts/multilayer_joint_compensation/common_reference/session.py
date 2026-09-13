"""Read-only common bundle binding for subsequent A runtimes.

No preparation/native target/history computation occurs here. Existing frozen
A0 source remains unchanged. Host FP32 selected states are a storage policy;
PredictionBatch's differentiable transfer still applies both actual weights
through every token of the full model.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import torch
from ..contracts import ABC, DATA, MODEL, member, digest, save, hf_member, Ledger
from ..observations import JointView
from ..banks import current_rows, protection_rows
from ..track_a.native_geometry import NativeWriterMetric


def verify_publication(path, *, status, expected_sha=None):
    from ..contracts import sha
    path = Path(path).absolute()
    if expected_sha is not None and sha(path) != expected_sha:
        raise RuntimeError('PUBLICATION_IDENTITY_MISMATCH')
    receipt = json.loads(path.read_text())
    if receipt['status'] != status:
        raise RuntimeError('PUBLICATION_NOT_VALID')
    if digest(receipt['members']) != receipt['members_root']:
        raise RuntimeError('PUBLICATION_MEMBER_ROOT')
    for item in receipt['members']:
        if member(item['path']) != item:
            raise RuntimeError('PUBLICATION_MEMBER_BYTES_CHANGED')
    return receipt


@dataclass
class CommonSession:
    root: Path
    ready: dict
    entry: dict
    rows: dict
    saved_teachers: dict
    geometries: tuple
    model: object
    tokenizer: object
    evaluator_tokenizer: object
    view: JointView
    records: list
    ledger: Ledger


def open_session(common, output, ledger, *, common_sha):
    """Verify sources before model load; restore exact We and check packing."""
    import os
    from project.run_scripts.single_layer_cumulative_risk import binding
    from scripts.fixed_counterfact import load_prefix
    if os.environ.get('CUMRISK_ASSET_ROOT') != str(ABC):
        raise RuntimeError('ASSET_ROOT_BINDING')
    root = Path(common).absolute(); output = Path(output)
    ready = verify_publication(root/'READY.json',
        status='COMMON_GPU_PREPARED_READY_FOR_SH2_VERIFY', expected_sha=common_sha)
    records = load_prefix(DATA, 10000)
    entry = torch.load(root/'entry.pt', map_location='cpu', weights_only=True, mmap=True)
    rows = torch.load(root/'prediction-rows.pt', map_location='cpu', weights_only=True)
    saved = {r: torch.load(root/f'We-teacher-{r}.pt', map_location='cpu', weights_only=True)
             for r in rows}
    geometries = []
    for layer in (4, 8):
        with ledger.time(f'geometry_reuse_L{layer}'):
            state = torch.load(root/f'native-geometry-L{layer}.pt', map_location='cpu',
                               weights_only=True, mmap=True)
            geometries.append(NativeWriterMetric.from_state(state))
        del state
    save(output/'model-tokenizer-source.json', dict(revision=MODEL.name,
        members=[hf_member(MODEL/name) for name in
                 ('config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json')],
        timing='before model/tokenizer load'))
    torch.manual_seed(20260911)
    model, tok, evaltok = binding.load_model(ledger)
    params = dict(model.named_parameters()); names = tuple(entry['names'])
    for name, w0 in zip(names, entry['W0']):
        if not torch.equal(params[name].detach().cpu(), w0):
            raise RuntimeError('MODEL_W0_IDENTITY')
    with torch.no_grad():
        for name, weight in zip(names, entry['We']):
            params[name].copy_(weight.to(params[name]))
    view = JointView(model, names, ledger)
    if dict(zip(names,view.entry_sha)) != entry['entry_identity']['weights']:
        raise RuntimeError('ACTUAL_ENTRY_IDENTITY')
    inventory = entry['raw_effective_inventory']
    repacked = {'Current':current_rows(records, inventory['current_effective'], entry['contexts'], tok)}
    for role, ids in inventory['bank'].items():
        repacked[role] = protection_rows(records, ids, role, tok)
    if digest(repacked) != digest(rows):
        raise RuntimeError('ACTUAL_TOKENIZER_PACKING_IDENTITY')
    save(output/'common-binding.json', dict(ready=member(root/'READY.json'),
        entry_identity=entry['entry_identity'], all_panels_sha=digest(rows),
        members_root=ready['members_root'], common_source_head=ready['source_head'],
        selected_state_storage='HOST_FP32', native_geometry_storage='HOST_FP64',
        model_and_forward='FULL_FP32', native_padding_side=tok.padding_side,
        evaluator_padding_side=evaltok.padding_side))
    def count_forward(module, positional, kw):
        ids = kw.get('input_ids', positional[0] if positional else None)
        ledger.add('actual_model_forward_invocations')
        if ids is not None: ledger.add('actual_forward_padded_tokens', ids.numel())
        if kw.get('attention_mask') is not None:
            ledger.add('actual_forward_nonpadding_tokens', int(kw['attention_mask'].sum()))
    model.register_forward_pre_hook(count_forward, with_kwargs=True)
    return CommonSession(root, ready, entry, rows, saved, tuple(geometries),
                         model, tok, evaltok, view, records, ledger)
