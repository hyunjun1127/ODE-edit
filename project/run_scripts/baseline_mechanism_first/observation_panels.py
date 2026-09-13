"""E1 forward/query and signed diagnostics; never native writer inputs.

All target-pair sequences retain every model position. Subject span is explicitly
unresolved: tokenizer subsequence guesses are not physical subject attribution.
Selected parameter temporary views restore the original storage and version;
the outer native transaction remains caller-owned. No model/state builder import.
"""
from __future__ import annotations

from contextlib import contextmanager
import gzip
import json
from pathlib import Path
import time

import torch

from .contracts import ContractBoundary, canonical, digest, member, save
from .evaluation import diagnostic_layout, diagnostic_margin
from .fixtures import capture_rng, restore_rng, tensor_sha
from .instrumentation import ReadOnlyCapture
from .panels import hash_order
from .signed_response import AllPositionContraction


@contextmanager
def temporary_weight(weight, value):
    """Storage swap for observation only, preserving original object/version."""
    if value.shape != weight.shape or value.dtype != weight.dtype or not torch.isfinite(value).all():
        raise ContractBoundary('OBSERVATION_WEIGHT_SCHEMA')
    original, pointer, version = weight.data, weight.data_ptr(), weight._version
    try:
        weight.data = value.detach().to(weight.device, copy=True)
        yield
    finally:
        weight.data = original
        if weight.data_ptr() != pointer or weight._version != version:
            raise ContractBoundary('OBSERVATION_WEIGHT_RESTORE')


def position_tags(batch):
    """Target prediction position zero is last prefix token, not continuation."""
    tags = []
    for attention, mask in zip(batch['attention_mask'], batch['target_prediction_mask']):
        target = mask.nonzero().flatten()
        if not len(target):
            raise ContractBoundary('EMPTY_TARGET_PREDICTION_MASK')
        first = int(target[0])
        tags.append(['PADDING' if not bool(attention[i]) else
                     ('CONTINUATION_INPUT' if i > first else 'PREFIX_SUBJECT_UNRESOLVED')
                     for i in range(len(attention))])
    return tags


def per_position_energy(queries, K, F, delta):
    """Streaming caller supplies one sequence pair, not whole-panel tensors."""
    if queries.ndim != 3 or K.ndim != 2 or F.shape != K.shape or queries.shape[-1] != K.shape[0]:
        raise ContractBoundary('QUERY_FACTOR_GEOMETRY')
    if delta.shape[1] != K.shape[0]:
        raise ContractBoundary('QUERY_DELTA_GEOMETRY')
    q = queries.detach().to(dtype=torch.float32)
    k, f, d = (x.detach().to(q) for x in (K, F, delta))
    values = dict(raw_Ktq_sq=(q @ k).double().square().sum(-1),
                  writer_Ftq_sq=(q @ f).double().square().sum(-1),
                  actual_delta_q_sq=(q @ d.T).double().square().sum(-1))
    if not all(bool(torch.isfinite(x).all()) for x in values.values()):
        raise ContractBoundary('NONFINITE_QUERY_OBSERVATION')
    return {key: value.cpu().tolist() for key,value in values.items()}


def _general_nll(model, ids):
    device = next(model.parameters()).device
    if len(ids) < 2:
        raise ContractBoundary('GENERAL_SELECTED_ROW_TOO_SHORT')
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(input_ids=tokens[:,:-1], attention_mask=torch.ones_like(tokens[:,:-1]), use_cache=False).logits.float()
    return torch.nn.functional.cross_entropy(logits.transpose(1,2), tokens[:,1:])


@torch.no_grad()
def evaluate_general(model, rows, *, state_label):
    """Actual full-sequence next-token NLL; caller owns state/restore guard."""
    out = []
    for row in rows:
        value = float(_general_nll(model, row['input_ids']))
        if not torch.isfinite(torch.tensor(value)):
            raise ContractBoundary('NONFINITE_GENERAL_NLL')
        out.append(dict(ordinal=row['ordinal'], state=state_label, nll=value,
                        predicted_tokens=len(row['input_ids'])-1, input_sha256=digest(row['input_ids'])))
    return out


def signed_inventory(records, current, historical, general_rows):
    groups = []
    for panel, population, category in [('Current',current,'R'),('Historical',historical,'R'),
                                       ('Historical',historical,'P'),('Current',current,'N'),
                                       ('Historical',historical,'N')]:
        if not population:
            continue
        chosen = hash_order(population, 'signed:'+panel+':'+category)[:32]
        if len(chosen) != 32:
            raise ContractBoundary('SIGNED_SUBPANEL_TOO_SMALL', panel=panel, category=category)
        groups.extend(dict(panel=panel, ordinal=i, category=category, prompt_index=0,
                           case_id=records[i]['case_id']) for i in chosen)
    general = hash_order([x['ordinal'] for x in general_rows], 'signed:General')[:16]
    if len(general) != 16:
        raise ContractBoundary('SIGNED_GENERAL_TOO_SMALL')
    return dict(pairs=groups, general_ordinals=general,
                selection='PRESEALED_CANONICAL_JSON_HASH; FIRST_P/FIRST_N; NO_OUTCOME',
                factor_derivative='ONLY_IF_ORIGINAL_R_PASSED_IN_LOCK')


@contextmanager
def _stream(path):
    path = Path(path)
    if any(x.is_symlink() for x in [path.parent,*path.parent.parents]):
        raise ContractBoundary('SYMLINK_OBSERVATION_OUTPUT')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as compressed:
            yield lambda row: compressed.write(canonical(row)+b'\n')


def run_observations(model, evaltok, weight, native_module_name, records,
                     current_ordinals, historical_ordinals, general_manifest,
                     entry_weight, endpoint_weight, actual_native_delta, K, F, output, lock,
                     *, w0_weight=None, diagnostic_R=None):
    """Run sealed observations only. Must run AFTER native endpoint capture.

    lock: layer, entry_n; optional signed_enabled (requires priority cell),
    Optional diagnostic_R and w0_weight are tensors, not JSON-lock members.
    Manifest may be loaded dict or create-once file path. Canonical evaluator
    measurements and alpha FD belong to caller and are never repeated here.
    """
    started = time.monotonic()
    output = Path(output)
    if output.exists():
        raise ContractBoundary('OBSERVATION_CREATE_ONCE_OUTPUT_EXISTS')
    if any(x.is_symlink() for x in [output.parent,*output.parent.parents]):
        raise ContractBoundary('SYMLINK_OBSERVATION_OUTPUT')
    output.mkdir(parents=True)
    general = json.loads(Path(general_manifest).read_text()) if isinstance(general_manifest,(str,Path)) else general_manifest
    rows = general['rows']
    if general['row_identity'] != digest(rows):
        raise ContractBoundary('GENERAL_MANIFEST_ROW_IDENTITY')
    module = model.get_submodule(native_module_name)
    if module.weight is not weight:
        raise ContractBoundary('OBSERVATION_MODULE_WEIGHT_IDENTITY')
    state = [(p,p.data_ptr(),p._version,p.requires_grad,p.grad) for p in model.parameters()]
    before_sha = tensor_sha(weight)
    rng = capture_rng()
    receipt = dict(status='RUNNING', writer_calls=0, canonical_evaluator_calls=0, fd_calls=0,
                   forward_pairs=0, signed_backward_pairs=0, general_forward_sequences=0,
                   general_backward_sequences=0, raw_position_rows=0,
                   subject_token_span='NOT_RECORDED_NO_SUBSEQUENCE_GUESS',
                   general_W0='MEASURED' if w0_weight is not None else 'CALLER_SEPARATE_EVALUATION_REQUIRED',
                   numerical_layout='PAIR2_DIAGNOSTIC_NOT_CANONICAL_MB16_PARITY',
                   input_preservation='ALL_CAUSAL_POSITIONS; PADDING_TAGGED_NOT_HIDDEN',
                   native_state_mutation=0, native_delta_sha256=tensor_sha(actual_native_delta),
                   direction_source='ACTUAL_ENDPOINT_DIFFERENCE_FP32',
                   source_pre_addition_update='CALLER_SEPARATE_ARTIFACT',
                   temporary_state_policy='DATA_STORAGE_SWAP_NO_COPY_INTO_LIVE_PARAMETER')
    signed = bool(lock.get('signed_enabled', False))
    priority = lock.get('layer') in (4,8) and lock.get('entry_n') in (5000,9000)
    if signed and not priority:
        raise ContractBoundary('SIGNED_SCOPE_NOT_PRESEALED_PRIORITY_CELL')
    inventory = signed_inventory(records,current_ordinals,historical_ordinals,rows) if signed else None
    if inventory:
        if lock.get('signed_panel_identity') != digest(inventory):
            raise ContractBoundary('SIGNED_PANEL_PRESEAL_MISMATCH')
        save(output/'signed-panel.lock.json',inventory)
    try:
        for p,*_ in state:
            p.requires_grad_(False)
        K,F,actual_native_delta=(x.detach().to(device=weight.device,dtype=torch.float32) for x in (K,F,actual_native_delta))
        with temporary_weight(weight,entry_weight):
            first_batch = first_q = None
            with _stream(output/'query-positions.jsonl.gz') as emit:
                for panel, ordinals in [('Current',current_ordinals),('Historical',historical_ordinals)]:
                    for ordinal in ordinals:
                        record = records[ordinal]
                        for category,count in [('R',1),('P',len(record['paraphrase_prompts'])),('N',len(record['neighborhood_prompts']))]:
                            for index in range(count):
                                batch = diagnostic_layout(evaltok,record,category=category,prompt_index=index,device=weight.device)
                                with torch.no_grad(), ReadOnlyCapture(module,destination=weight.device) as captured:
                                    model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False)
                                if len(captured.records) != 1:
                                    raise ContractBoundary('SINGLETON_MODULE_MULTIPLE_CALLS')
                                q = captured.records[0]['input']
                                if first_batch is None:
                                    first_batch = batch; first_q = q.detach().cpu().clone()
                                energies = per_position_energy(q,K,F,actual_native_delta)
                                tags = position_tags(batch)
                                masks = batch['target_prediction_mask'].cpu().tolist()
                                for side in range(2):
                                    for pos,tag in enumerate(tags[side]):
                                        emit(dict(panel=panel,ordinal=ordinal,case_id=record['case_id'],category=category,
                                                  prompt_index=index,side=('new','true')[side],position=pos,position_tag=tag,
                                                  target_prediction=masks[side][pos],input_identity=batch['identity'],
                                                  **{k:v[side][pos] for k,v in energies.items()}))
                                        receipt['raw_position_rows'] += 1
                                receipt['forward_pairs'] += 1
                for row in rows:
                    ids = torch.tensor([row['input_ids'][:-1]],device=weight.device,dtype=torch.long)
                    with torch.no_grad(), ReadOnlyCapture(module,destination=weight.device) as captured:
                        model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
                    if len(captured.records) != 1:
                        raise ContractBoundary('SINGLETON_MODULE_MULTIPLE_CALLS')
                    energies = per_position_energy(captured.records[0]['input'],K,F,actual_native_delta)
                    for pos in range(ids.shape[1]):
                        emit(dict(panel='General',ordinal=row['ordinal'],category='GENERAL',side='text',position=pos,
                                  position_tag='GENERAL_TEXT_INPUT',target_prediction=True,
                                  input_identity=digest(row['input_ids']),**{k:v[0][pos] for k,v in energies.items()}))
                        receipt['raw_position_rows'] += 1
                    receipt['general_forward_sequences'] += 1
            if signed:
                factor = None
                if diagnostic_R is not None:
                    factor = (diagnostic_R.to(device=weight.device,dtype=torch.float32),F)
                with _stream(output/'signed-response.jsonl.gz') as emit:
                    for item in inventory['pairs']:
                        batch=diagnostic_layout(evaltok,records[item['ordinal']],category=item['category'],prompt_index=0,device=weight.device)
                        with torch.enable_grad(), AllPositionContraction(module,actual_native_delta.to(weight.device),
                                position_mask=batch['attention_mask'].bool(),diagnostic_factor=factor) as capture:
                            scalar=diagnostic_margin(model,evaltok,records[item['ordinal']],item['category'],0)
                            result=capture.compute(scalar)
                        result['direction_source']='ACTUAL_ENDPOINT_DIFFERENCE_FP32'
                        emit(dict(**item,scalar=float(scalar.detach()),**result))
                        receipt['signed_backward_pairs'] += 1
                    by_ordinal={r['ordinal']:r for r in rows}
                    for ordinal in inventory['general_ordinals']:
                        with torch.enable_grad(), AllPositionContraction(module,actual_native_delta.to(weight.device),diagnostic_factor=factor) as capture:
                            scalar=_general_nll(model,by_ordinal[ordinal]['input_ids'])
                            result=capture.compute(scalar)
                        result['direction_source']='ACTUAL_ENDPOINT_DIFFERENCE_FP32'
                        emit(dict(panel='General',ordinal=ordinal,category='GENERAL_NLL',scalar=float(scalar.detach()),**result))
                        receipt['general_backward_sequences'] += 1
            general_entry=evaluate_general(model,rows,state_label='ENTRY')
            receipt['general_forward_sequences'] += len(rows)
        with temporary_weight(weight,endpoint_weight):
            if first_batch is not None:
                with torch.no_grad(), ReadOnlyCapture(module,destination='cpu') as captured:
                    model(input_ids=first_batch['input_ids'],attention_mask=first_batch['attention_mask'],use_cache=False)
                q=captured.records[0]['input']
                receipt['own_input_stability']=dict(input_identity=first_batch['identity'],entry_sha256=tensor_sha(first_q),
                    endpoint_sha256=tensor_sha(q),equal=torch.equal(first_q,q),max_abs=float((first_q-q).abs().max()),
                    claim='ONE_FIXED_INPUT_SINGLETON_ONLY_NOT_OTHER_LAYERS')
                receipt['forward_pairs'] += 1
            general_endpoint=evaluate_general(model,rows,state_label='NATIVE')
            receipt['general_forward_sequences'] += len(rows)
        general_w0='NOT_MEASURED_BY_THIS_API'
        if w0_weight is not None:
            with temporary_weight(weight,w0_weight):
                general_w0=evaluate_general(model,rows,state_label='W0')
            receipt['general_forward_sequences'] += len(rows)
        save(output/'general-nll.json',dict(entry=general_entry,native=general_endpoint,
                                          W0=general_w0,scalar='GENERAL_NLL_SEPARATE'))
        receipt['status']='OBSERVATIONS_FINITE'
    except BaseException as exc:
        receipt.update(status='TECHNICAL_EXCEPTION',exception_type=type(exc).__name__,exception=str(exc))
        raise
    finally:
        for p,ptr,version,flag,grad in state:
            p.requires_grad_(flag)
            if p.data_ptr()!=ptr or p._version!=version or p.grad is not grad:
                receipt['parameter_guard']='FAILED'
        receipt['selected_version_change']=weight._version-next(v for p,_,v,_,_ in state if p is weight)
        receipt['selected_bytes_restored']=tensor_sha(weight)==before_sha
        receipt['rng_consumed_before_restore']=capture_rng()!=rng
        restore_rng(rng)
        receipt['rng_restored']=capture_rng()==rng
        receipt.setdefault('parameter_guard','PASS')
        if receipt['parameter_guard']!='PASS' or not receipt['selected_bytes_restored'] or receipt['rng_consumed_before_restore']:
            receipt['status']='OBSERVER_STATE_INTERFERENCE'
        receipt['output_members']=[member(path) for path in sorted(output.iterdir()) if path.is_file()]
        receipt['wall_seconds']=time.monotonic()-started
        save(output/'observation-receipt.json',receipt)
    if receipt['parameter_guard']!='PASS' or not receipt['selected_bytes_restored'] or receipt['rng_consumed_before_restore']:
        raise ContractBoundary('OBSERVER_STATE_INTERFERENCE',receipt=receipt)
    return receipt
