"""Outcome-selected, physical-checkpoint suffix diagnostics; no new editing.

Original first100 NS1000 category ordering and MB16 packing are retained. Only
groups containing selected rows are forwarded. Each new/true TF path retains
all valid input tokens. The frozen L4 output is made a differentiable leaf, so
VJPs reach the suffix input but never parameter gradients. Checkpoints/deltas
are never saved; token/key/gradient artifacts are local analysis data only.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import gc
import importlib
import math
import os
from pathlib import Path
import resource
import time
import traceback

import torch

from . import model_runtime as rt
from . import validation as val
from .common import (ATTEMPT, CONTRACT, INITIAL, S4CELL, WEIGHT, EXECUTION_POLICY, digest, mapped,
                     read, sha256, source_map, tensor_sha, write_csv, write_json)


INTERVALS = ((1, 10), (50, 100))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_dependencies(gate, archival, bookkeeping):
    require(archival.get('status') == 'PASS', 'ARCHIVAL_A01_NOT_PASS')
    require(bookkeeping.get('status') == 'PASS', 'ACTIVATION_BOOKKEEPING_F00_NOT_PASS')


def pack_original(kernel, tok, pairs, device):
    """Byte-for-byte arithmetic/packing policy of original evaluate_pairs."""
    encoded = [kernel._encode_pair(tok, pair) for pair in pairs]
    full = [prompt + target for prompt, target in encoded]
    require(bool(full), 'EMPTY_GROUP')
    maximum = max(len(ids) - 1 for ids in full)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.full((len(pairs), maximum), int(pad), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    positions, targets = [], []
    for row, (prompt, target) in enumerate(encoded):
        require(bool(prompt) and bool(target), 'EMPTY_PROMPT_OR_TARGET')
        inputs, labels = (prompt + target)[:-1], (prompt + target)[1:]
        start = maximum - len(inputs)
        ids[row, start:] = torch.tensor(inputs, device=device)
        mask[row, start:] = 1
        target_start = start + len(prompt) - 1
        require(labels[-len(target):] == target, 'TEACHER_FORCED_TARGET_ALIGNMENT')
        positions.append(list(range(target_start, maximum)))
        targets.append(list(target))
    return dict(input_ids=ids, attention_mask=mask, target_positions=positions,
                target_ids=targets, position_ids='IMPLICIT_NOT_PASSED',
                packing='ORIGINAL_CATEGORY_MB16_MANUAL_LEFT', microbatch_size=16)


class ActivationOffload:
    """Offload saved activations, not duplicate resident frozen model weights."""
    def __init__(self, model):
        self.weight_storages = {p.untyped_storage().data_ptr() for p in model.parameters()}
        self.bytes = 0
        self.tensors = 0
        self.reload_bytes = 0

    def pack(self, tensor):
        if tensor.device.type != 'cuda' or tensor.untyped_storage().data_ptr() in self.weight_storages:
            return ('resident', tensor.detach())
        cpu = tensor.detach().to('cpu')
        self.bytes += cpu.numel() * cpu.element_size()
        self.tensors += 1
        return ('cpu', str(tensor.device), cpu)

    def unpack(self, item):
        if item[0] == 'resident':
            return item[1]
        self.reload_bytes += item[2].numel() * item[2].element_size()
        return item[2].to(item[1])


def forward_group(model, packed, selected_rows, *, gradient=False, sign=1., offload=True):
    """Original full-vocabulary forward; one scalar mean-token VJP per row.

    sign=+1 for NS new NLL, -1 for NS true NLL. Summing both signed path
    derivatives gives the safety-margin derivative. No logical-batch averaging
    is inserted: each diagnostic scalar is one prompt's token-mean NLL.
    """
    require(sign in (-1., 1.), 'INVALID_MARGIN_PATH_SIGN')
    require(all(not p.requires_grad for p in model.parameters()), 'MODEL_PARAMETERS_NOT_FROZEN')
    require(len(set(selected_rows)) == len(selected_rows), 'DUPLICATE_SELECTED_ROW')
    require(all(0 <= i < packed['input_ids'].shape[0] for i in selected_rows), 'SELECTED_ROW_RANGE')
    captured = {}
    module = model.get_submodule(WEIGHT.rsplit('.', 1)[0])

    def capture(module, inputs, output):
        require(isinstance(output, torch.Tensor) and len(inputs) == 1, 'UNEXPECTED_L4_OUTPUT_SCHEMA')
        captured['keys'] = inputs[0].detach()
        if gradient:
            leaf = output.detach().requires_grad_(True)
            captured['leaf'] = leaf
            return leaf
        return output

    hook = module.register_forward_hook(capture)
    offloader = ActivationOffload(model)
    saved = torch.autograd.graph.saved_tensors_hooks(offloader.pack, offloader.unpack) if gradient and offload else nullcontext()
    started = time.perf_counter()
    backward_seconds = 0.
    rows = {}
    try:
        with torch.set_grad_enabled(gradient), saved:
            logits = model(input_ids=packed['input_ids'], attention_mask=packed['attention_mask'], use_cache=False).logits.float()
            log_probs = torch.log_softmax(logits, dim=-1)
            predictions = logits.argmax(dim=-1)
            if logits.device.type == 'cuda':
                torch.cuda.synchronize(logits.device)
            forward_seconds = time.perf_counter() - started
            for ordinal, row in enumerate(selected_rows):
                positions = packed['target_positions'][row]
                targets = torch.tensor(packed['target_ids'][row], device=logits.device)
                selected = log_probs[row, positions, :].gather(1, targets[:, None])
                nll = -selected.mean()
                require(bool(torch.isfinite(nll)), 'NONFINITE_NLL')
                predicted = predictions[row, positions]
                valid = packed['attention_mask'][row].bool()
                value = dict(nll=float(nll.detach()), token_predictions=predicted.detach().cpu().tolist(),
                    target_token_ids=targets.cpu().tolist(), token_correct=(predicted == targets).cpu().tolist(),
                    all_tokens_correct=bool(torch.all(predicted == targets)),
                    K=captured['keys'][row, valid].detach().cpu().contiguous(),
                    input_ids=packed['input_ids'][row, valid].cpu().tolist(),
                    valid_positions=valid.nonzero().flatten().cpu().tolist(), target_positions=positions)
                if gradient:
                    begin = time.perf_counter()
                    grad = torch.autograd.grad(sign * nll, captured['leaf'],
                        retain_graph=ordinal + 1 < len(selected_rows), create_graph=False)[0]
                    if logits.device.type == 'cuda':
                        torch.cuda.synchronize(logits.device)
                    backward_seconds += time.perf_counter() - begin
                    require(bool(torch.isfinite(grad).all()), 'NONFINITE_ACTIVATION_GRADIENT')
                    value['margin_gradient'] = grad[row, valid].detach().cpu().contiguous()
                    value['padding_gradient_norm'] = float(torch.linalg.vector_norm(grad[row, ~valid].double()))
                    other = [i for i in range(grad.shape[0]) if i != row]
                    value['other_batch_gradient_norm'] = float(torch.linalg.vector_norm(grad[other].double())) if other else 0.
                    del grad
                rows[row] = value
        require(all(p.grad is None for p in model.parameters()), 'UNEXPECTED_MODEL_PARAMETER_GRAD')
        ledger = dict(forward_calls=1, vjp_calls=len(selected_rows) if gradient else 0,
            selected_target_paths=len(selected_rows), processed_sequences=packed['input_ids'].shape[0],
            valid_input_tokens=int(packed['attention_mask'].sum()), padded_input_tokens=packed['input_ids'].numel(),
            full_vocabulary_size=logits.shape[-1], forward_seconds=forward_seconds,
            backward_seconds=backward_seconds, total_seconds=time.perf_counter()-started,
            activation_offload_bytes=offloader.bytes, activation_offload_tensors=offloader.tensors,
            activation_reload_bytes=offloader.reload_bytes, parameter_gradients=0)
        return rows, ledger
    finally:
        hook.remove()


def activation_statistics(entry_delta, interval_delta, keys, margin_gradient):
    """FP64 all-valid-token response, signed cross term, and linearization."""
    device = entry_delta.device
    e, d = entry_delta.double(), interval_delta.double()
    k = keys.to(device=device, dtype=torch.float64).T
    g = margin_gradient.to(device=device, dtype=torch.float64).T
    require(e.shape == d.shape and e.shape[1] == k.shape[0], 'ACTIVATION_DIMENSION_MISMATCH')
    baseline, response = e @ k, d @ k
    require(g.shape == response.shape, 'GRADIENT_POSITION_DIMENSION_MISMATCH')
    baseline2, response2 = baseline.square().sum(), response.square().sum()
    cross = 2 * (baseline * response).sum()
    change = (baseline + response).square().sum() - baseline2
    predicted = (g * response).sum()
    values = dict(EK_squared_norm=float(baseline2), DK_squared_norm=float(response2),
        EK_DK_signed_cross_term=float(cross), activation_energy_change=float(change),
        energy_identity_error=float(change - cross - response2),
        predicted_margin_change=float(predicted), gradient_squared_norm=float(g.square().sum()),
        valid_tokens=k.shape[1], reduction_dtype='float64')
    require(all(math.isfinite(v) for v in values.values() if isinstance(v, float)), 'NONFINITE_ACTIVATION_STATS')
    return values


def selected_population(kernel, locality, records, panel):
    """Join a preselected panel to exact original NS1000 order; no resampling."""
    pairs = kernel.counterfact_pairs(records)
    pairs['locality_target_new'] = locality.counterfact_locality_target_new_pairs(records)
    new, true = pairs['locality_target_new'], pairs['locality_target_true']
    require(len(new) == len(true) == 1000, 'FIRST100_NS1000_REQUIRED')
    inventory = {}
    for i, (a, b) in enumerate(zip(new, true, strict=True)):
        require((a.case_id, a.prompt_index, a.prompt) == (b.case_id, b.prompt_index, b.prompt), 'NEW_TRUE_PAIR_IDENTITY')
        ident = digest([a.case_id, a.prompt_index, a.prompt, a.target, b.target])
        require(ident not in inventory, 'DUPLICATE_NS_IDENTITY')
        inventory[ident] = i
    selected = {}
    for row in panel:
        require(row['population'] == 'FIRST100_NS1000' and row['metric_tag'] == 'NS', 'PANEL_POPULATION_CHANGED')
        require(row['identity'] in inventory, 'PANEL_IDENTITY_NOT_IN_FIRST100')
        index = inventory[row['identity']]
        require(index not in selected, 'DUPLICATE_PANEL_ROW')
        require(int(row['case_id']) == new[index].case_id and int(row['prompt_index']) == new[index].prompt_index, 'PANEL_CASE_PROMPT_MISMATCH')
        selected[index] = row
    return {'new': new, 'true': true}, selected


def save_path(path, value):
    """A key/gradient/token observation is not a restorable model delta."""
    require(set(value) <= {'K', 'margin_gradient', 'metadata'}, 'UNAPPROVED_ANALYSIS_TENSOR_FIELDS')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        torch.save(value, handle)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path),
                kind='ALL_VALID_TOKEN_KEY_AND_MARGIN_GRADIENT_NOT_CHECKPOINT')


def artifact_members(output, tensor_receipts):
    """Bind scalar/raw observations; reuse tensor hashes from their creation."""
    known={str(Path(m['path'])):m for m in tensor_receipts}
    members=[]
    for path in sorted(Path(output).rglob('*')):
        if not path.is_file() or path.name=='terminal.json':continue
        if path.suffix=='.pt':
            require(str(path) in known,'UNBOUND_OR_UNAPPROVED_TENSOR_ARTIFACT')
            member=known[str(path)]
            require(path.stat().st_size==member['bytes'],'TENSOR_ARTIFACT_SIZE_CHANGED')
            members.append(member)
        else:
            members.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha256(path)))
    return members


def archive_rows(batch, identities, mapping):
    obj = read(mapped(S4CELL + f'/B{batch:03d}/seen-full.json', mapping))
    selected = {r['identity']: r for r in obj['metrics']['NS']['rows'] if r['identity'] in identities}
    require(set(selected) == identities, 'ARCHIVE_PANEL_MISSING')
    return selected, obj['state']


def metric_parity(actual, reference, identities):
    # Already-computed scalar observations only; no threshold or extra forward.
    return dict(max_new_nll_difference=max(abs(actual[i]['new_nll']-reference[i]['new_nll']) for i in identities),
        max_true_nll_difference=max(abs(actual[i]['true_nll']-reference[i]['true_nll']) for i in identities),
        max_margin_difference=max(abs((actual[i]['new_nll']-actual[i]['true_nll'])-
            (reference[i]['new_nll']-reference[i]['true_nll'])) for i in identities),**EXECUTION_POLICY)


def run_interval(model, tok, w0, bindings, population, panel, interval, output):
    a, b = interval; output = Path(output); output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter(); mapping = source_map()
    kernel = importlib.import_module('project.run_scripts.alphaedit_strength_neutral_barrier.evaluator')
    locality = importlib.import_module('project.run_scripts.ordered_response_barrier_ode.counterfact_locality_evaluator')
    paths, selected = selected_population(kernel, locality, population, panel)
    if not selected:
        result = dict(status='SKIPPED', reason='PANEL_EMPTY_NO_POPULATION_EXPANSION', interval=[a,b])
        write_json(output/'terminal.json', result); return result
    require(sum(r['selection_role'] == 'lost' for r in panel) <= 16 and
            sum(r['selection_role'] == 'matched_retained' for r in panel) <= 16, 'PANEL_SIZE_EXCEEDED')
    identities = [selected[i]['identity'] for i in sorted(selected)]
    archive_a, state_a = archive_rows(a, set(identities), mapping)
    archive_b, state_b = archive_rows(b, set(identities), mapping)
    cpa, cpb = rt.checkpoint(a), rt.checkpoint(b)
    wa, wb = cpa['weights'][WEIGHT], cpb['weights'][WEIGHT]
    require(tensor_sha(wa) == state_a['weights'][WEIGHT] and tensor_sha(wb) == state_b['weights'][WEIGHT], 'CHECKPOINT_ARCHIVE_WEIGHT_IDENTITY')
    e, d = wa.double() - w0.double(), wb.double() - wa.double()
    device = model.get_parameter(WEIGHT).device
    e_gpu, d_gpu = e.to(device), d.to(device)
    before = rt.pointer_versions(model)
    all_rows, partial, ledgers, path_stats, key_hashes, endpoint_checks = [], {}, [], {}, {}, {}
    tensor_receipts=[]
    stage = 'RESTORE'
    try:
        for s in (0., .5, 1.):
            state = (wa.double() + s*d).float()
            require((s != 0 or torch.equal(state, wa)) and (s != 1 or torch.equal(state, wb)), 'INTERPOLATION_ENDPOINT_NOT_EXACT')
            rt.set_weight(model, state)
            state_hash = tensor_sha(model.get_parameter(WEIGHT))
            sub = output / ('s' + str(s).replace('.', 'p')); sub.mkdir()
            observed = {i: {} for i in selected}
            for target, sign in (('new', 1.), ('true', -1.)):
                pairs = paths[target]
                for offset in sorted({i // 16 * 16 for i in selected}):
                    group = pairs[offset:offset+16]
                    local_rows = sorted(i-offset for i in selected if offset <= i < offset+len(group))
                    packed = pack_original(kernel, tok, group, device)
                    label = f'{target}-group{offset:04d}'
                    token_meta = dict(category='locality_target_' + target, original_group_offset=offset,
                        case_prompt_order=[[r.case_id,r.prompt_index] for r in group],
                        input_ids=packed['input_ids'].cpu().tolist(), attention_mask=packed['attention_mask'].cpu().tolist(),
                        target_ids=packed['target_ids'], target_positions=packed['target_positions'],
                        position_ids='IMPLICIT_NOT_PASSED', microbatch_size=16,
                        input_ids_sha256=tensor_sha(packed['input_ids']), attention_mask_sha256=tensor_sha(packed['attention_mask']))
                    write_json(sub/(label+'-tokens.json'),token_meta)
                    stage = f's={s}/{label}'
                    got, ledger = forward_group(model, packed, local_rows, gradient=(s==0), sign=sign)
                    ledger.update(interval_start=a, interval_end=b, s=s, target=target, group_offset=offset)
                    ledger['validation_only_forward_calls']=0
                    ledgers.append(ledger)
                    group_scalars=[]
                    for local, value in got.items():
                        idx=offset+local; meta=selected[idx]; ident=meta['identity']
                        kh=tensor_sha(value['K']); key=(ident,target)
                        if s==0:
                            key_hashes[key]=kh
                            stats=activation_statistics(e_gpu,d_gpu,value['K'],value['margin_gradient'])
                            path_stats[key]=stats
                            artifact=save_path(sub/f'{ident}-{target}-activation.pt',dict(K=value['K'],margin_gradient=value['margin_gradient'],
                                metadata=dict(identity=ident,target=target,interval=[a,b],**{k:v for k,v in value.items() if k not in ('K','margin_gradient')})))
                            tensor_receipts.append(artifact)
                            write_json(sub/f'{ident}-{target}-statistics.json',dict(**stats,artifact=artifact))
                        else:
                            value['key_hash_matches_s0']=kh==key_hashes[key]
                        observed[idx][target+'_nll']=value['nll']
                        observed[idx][target+'_strict']=value['all_tokens_correct']
                        observed[idx][target+'_tokens']=len(value['target_token_ids'])
                        group_scalars.append(dict(identity=ident,case_id=int(meta['case_id']),prompt_index=int(meta['prompt_index']),
                            **{k:v for k,v in value.items() if k not in ('K','margin_gradient')},key_sha256=kh))
                    write_json(sub/(label+'-observations.json'),group_scalars)
                    del got,packed; gc.collect()
            current={selected[i]['identity']:v for i,v in observed.items()}
            for idx,v in observed.items():
                meta=selected[idx]; margin=v['new_nll']-v['true_nll']
                all_rows.append(dict(interval_start=a,interval_end=b,s=s,identity=meta['identity'],case_id=int(meta['case_id']),
                    prompt_index=int(meta['prompt_index']),selection_role=meta['selection_role'],safety_margin=margin,
                    success=margin>0,weight_sha256=state_hash,**v))
            partial[s]=current
            write_csv(sub/'selected_metrics.csv',[r for r in all_rows if r['s']==s])
            if s in (0.,1.):
                parity=metric_parity(current,archive_a if s==0 else archive_b,identities)
                endpoint_checks[str(s)]=parity
                write_json(sub/'archive-parity.json',parity)
            require(state_hash==tensor_sha(model.get_parameter(WEIGHT)), 'SELECTED_WEIGHT_MUTATION')
            require(before==rt.pointer_versions(model), 'NONSELECTED_PARAMETER_MUTATION')
        summary=[]
        for idx in sorted(selected):
            meta=selected[idx]; ident=meta['identity']; new=path_stats[ident,'new']; true=path_stats[ident,'true']
            ma=partial[0.][ident]['new_nll']-partial[0.][ident]['true_nll']
            mb=partial[1.][ident]['new_nll']-partial[1.][ident]['true_nll']
            predicted=new['predicted_margin_change']+true['predicted_margin_change']
            common=dict(interval_start=a,interval_end=b,identity=ident,case_id=int(meta['case_id']),
                prompt_index=int(meta['prompt_index']),selection_role=meta['selection_role'],
                entry_margin=ma,endpoint_margin=mb,actual_margin_change=mb-ma,predicted_margin_change=predicted,
                nonlinear_remainder=(mb-ma)-predicted,sign_match=((mb-ma)>0)==(predicted>0) if mb!=ma and predicted!=0 else None,
                derivative_zero=predicted==0,actual_change_zero=mb==ma,all_valid_tokens=True,
                population='FIRST100_NS1000',outcome_selected_not_population_estimator=True)
            for name in ('EK_squared_norm','DK_squared_norm','EK_DK_signed_cross_term','activation_energy_change'):
                common[name]=new[name]+true[name]
            common.update(D_energy=common['DK_squared_norm'],cross_term=common['EK_DK_signed_cross_term'],
                          energy_aggregation='SUM_OF_SEPARATE_NEW_AND_TRUE_TF_PATHS')
            common.update(new_target_valid_tokens=new['valid_tokens'],true_target_valid_tokens=true['valid_tokens'])
            summary.append(common)
        write_csv(output/'activation_margin.csv',summary)
        write_csv(output/'interpolation_metrics.csv',all_rows)
        write_csv(output/'compute.csv',ledgers)
        rt.set_weight(model,w0)
        require(tensor_sha(model.get_parameter(WEIGHT))==tensor_sha(w0), 'W0_RESTORE_FAILURE')
        require(before==rt.pointer_versions(model), 'NONSELECTED_RESTORE_MUTATION')
        result=dict(status='PASS',cell_id=f'G{a:03d}_{b:03d}',interval=[a,b],selected_rows=len(summary),
            roles={r:sum(m['selection_role']==r for m in panel) for r in ('lost','matched_retained')},
            scalar_margin_gradients=len(summary),target_path_vjp_calls=sum(r['vjp_calls'] for r in ledgers),
            new_model_forward_calls=sum(r['forward_calls']+r.get('parity_reference_forward_calls',0) for r in ledgers),
            W0_restore=True,nonselected_pointer_versions=True,nonselected_full_bytehash='NOT_CLAIMED',
            save_checkpoints=False,new_editing=0,z_optimization=0,history_append=0,
            endpoint_archive_parity=endpoint_checks,wall_seconds=time.perf_counter()-start,
            peak_ram_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_gpu_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0,
            members=artifact_members(output,tensor_receipts))
        result.update(EXECUTION_POLICY)
        write_json(output/'terminal.json',result)
        return result
    except BaseException as exc:
        restored=False
        try:
            rt.set_weight(model,w0);restored=tensor_sha(model.get_parameter(WEIGHT))==tensor_sha(w0)
        except BaseException:
            pass
        if ledgers:write_csv(output/'partial_compute.csv',ledgers)
        failure=dict(status='FAILED',cell_id=f'G{a:03d}_{b:03d}',stage=stage,error=repr(exc),traceback=traceback.format_exc(),
            W0_restore=restored,wall_seconds=time.perf_counter()-start,save_checkpoints=False,
            failure_is_not_zero_score=True,technical_or_numerical_status_preserved=True)
        write_json(output/'failure.json',failure)
        write_json(output/'terminal.json',dict(**failure,members=artifact_members(output,tensor_receipts)))
        return failure
    finally:
        del e_gpu,d_gpu,e,d,cpa,cpb
        gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()


def run(output, gate_path, archival_path, bookkeeping_path, interval='all'):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    archival_path=Path(archival_path)
    try:
        gate,archive,book=read(gate_path),read(archival_path/'archival-receipt.json'),read(bookkeeping_path)
        validate_dependencies(gate,archive,book)
        with (archival_path/'mechanism_panel.csv').open(newline='') as handle:panel=list(csv.DictReader(handle))
        for path in (archival_path/'mechanism_panel.csv',archival_path/'mechanism_parent_rp.parquet'):
            matching=[m for m in archive['outputs'] if Path(m['path']).name==path.name]
            require(len(matching)==1 and matching[0]['sha256']==sha256(path), 'SEALED_ARCHIVAL_PANEL_OR_PARENT_HASH')
        requested=INTERVALS if interval=='all' else (tuple(map(int,interval.split(':'))),)
        require(all(i in INTERVALS for i in requested),'UNAUTHORIZED_INTERVAL')
        write_json(output/'inputs.json',dict(gate=dict(path=str(gate_path),sha256=sha256(gate_path)),
            archival=dict(path=str(archival_path/'archival-receipt.json'),sha256=sha256(archival_path/'archival-receipt.json')),
            bookkeeping=dict(path=str(bookkeeping_path),sha256=sha256(bookkeeping_path)),
            parent_RP=dict(path=str(archival_path/'mechanism_parent_rp.parquet'),sha256=sha256(archival_path/'mechanism_parent_rp.parquet'),
                           reused_observations=True,new_GPU_evaluation=0),save_checkpoints=False))
        bindings=rt.bind_sources();population,_=rt.stream_and_probe()
        model,_,tok,w0,runtime=rt.load();write_json(output/'runtime.json',runtime)
        require(model.config.model_type=='llama','ORIGINAL_LLAMA_ARCHITECTURE_REQUIRED')
        results=[]
        for a,b in requested:
            selected=[r for r in panel if int(r['interval_start'])==a and int(r['interval_end'])==b]
            result=run_interval(model,tok,w0,bindings,population[:100],selected,(a,b),output/f'G{a:03d}_{b:03d}')
            results.append(result)
            print('ACTIVATION_INTERVAL_TERMINAL',a,b,result['status'],flush=True)
            # Continue independent interval only after exact restoration.
            if result.get('W0_restore') is False:break
        final=dict(status='PASS' if all(r['status']=='PASS' for r in results) and len(results)==len(requested) else 'FAILED',
            intervals=results,save_checkpoints=False,scientific_editing=0,source_map_sha256=sha256(ATTEMPT/'inputs/source-map.json'),
            slurm_job_id=os.environ.get('SLURM_JOB_ID'))
        final.update(EXECUTION_POLICY)
        write_json(output/'terminal.json',final)
        return final
    except BaseException as exc:
        write_json(output/'failure.json',dict(status='BLOCKED_OR_FAILED',error=repr(exc),traceback=traceback.format_exc(),
            save_checkpoints=False,new_editing=0))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True);parser.add_argument('--gate',required=True)
    parser.add_argument('--archival',required=True);parser.add_argument('--bookkeeping',required=True)
    parser.add_argument('--interval',default='all',choices=['all','1:10','50:100'])
    args=parser.parse_args()
    run(args.output,args.gate,args.archival,args.bookkeeping,args.interval)
