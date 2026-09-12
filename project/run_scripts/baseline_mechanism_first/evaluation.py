"""Thin read-only binding to historical BLUE NLL-pair evaluation.

Canonical observations retain historical microbatch16 and reducer bytes. The
gradient diagnostic deliberately uses one fixed two-sequence (new,true) batch;
it is not claimed bitwise equal to the historical16 layout on real hardware.
Caller owns model/weight/history transaction guards. No editor is imported.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import types

import torch


class EvaluationBindingError(RuntimeError):
    pass


_BOUND = None


def _namespace(name: str, path: Path):
    path = path.resolve()
    if name in sys.modules:
        if list(getattr(sys.modules[name], "__path__", ())) != [str(path)]:
            raise EvaluationBindingError(f"namespace already bound elsewhere: {name}")
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    sys.modules[name] = module


def _module(name: str, path: Path):
    if path.is_symlink() or not path.is_file():
        raise EvaluationBindingError(f"non-regular source: {path}")
    if name in sys.modules:
        if Path(sys.modules[name].__file__).resolve() != path.resolve():
            raise EvaluationBindingError(f"module already bound elsewhere: {name}")
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def bind_evaluation_sources(historical_root, *, helper_root=None) -> dict:
    """Bind only exact observation files; no package initializers or save calls.

    ``historical_root`` contains BLUE's evaluation.py/integrity.py. Asset SHA
    admission is caller-owned; returned identities permit exact runtime locking.
    Binding cannot switch source directories inside a process.
    """
    global _BOUND
    history = Path(historical_root).resolve()
    helpers = Path(helper_root).resolve() if helper_root else Path(__file__).resolve().parents[1]
    namespace = "project.run_scripts.baseline_mechanism_first._historical_observation"
    _namespace(namespace, history)
    integrity = _module(namespace+".integrity", history/"integrity.py")
    historical = _module(namespace+".evaluation", history/"evaluation.py")
    snb = "project.run_scripts.alphaedit_strength_neutral_barrier"
    orb = "project.run_scripts.ordered_response_barrier_ode"
    _namespace(snb, helpers/"alphaedit_strength_neutral_barrier")
    _namespace(orb, helpers/"ordered_response_barrier_ode")
    contracts = _module(snb+".contracts", helpers/"alphaedit_strength_neutral_barrier/contracts.py")
    evaluator = _module(snb+".evaluator", helpers/"alphaedit_strength_neutral_barrier/evaluator.py")
    locality = _module(orb+".counterfact_locality_evaluator", helpers/"ordered_response_barrier_ode/counterfact_locality_evaluator.py")
    sources = [history/"integrity.py", history/"evaluation.py",
               helpers/"alphaedit_strength_neutral_barrier/contracts.py",
               helpers/"alphaedit_strength_neutral_barrier/evaluator.py",
               helpers/"ordered_response_barrier_ode/counterfact_locality_evaluator.py"]
    receipt = {str(p):dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources}
    _BOUND = dict(integrity=integrity, historical=historical, contracts=contracts,
                  evaluator=evaluator, locality=locality, receipt=receipt)
    return dict(sources=receipt, canonical_microbatch=16, package_initializers_executed=0,
                source_modification=0, transaction_guard="CALLER_OWNED")


def _binding():
    if _BOUND is None:
        raise EvaluationBindingError("bind explicit historical sources before evaluation")
    return _BOUND


def _device(model):
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise EvaluationBindingError("cannot infer model input device") from exc


def evaluate_records(model, tok, records) -> dict:
    """Historical raw-free per-prompt rows, including canonical NS new+true."""
    modules = _binding()
    records = list(records)
    if not records or len({r["case_id"] for r in records}) != len(records):
        raise EvaluationBindingError("canonical evaluation requires nonempty unique case inventory")
    pairs = modules["evaluator"].counterfact_pairs(records)
    pairs["locality_target_new"] = modules["locality"].counterfact_locality_target_new_pairs(records)
    raw = {kind:modules["evaluator"].evaluate_pairs(model, tok, values,
              device=_device(model), microbatch_size=16) for kind,values in pairs.items()}
    return dict(requests=len(records),
                request_order=modules["integrity"].digest([r["case_id"] for r in records]),
                metrics=modules["historical"].reduce(raw),
                evaluator_controller_influence=0,
                transaction_guard="CALLER_OWNED_NOT_ASSERTED_BY_EVALUATOR",
                evaluator_layout="HISTORICAL_MICROBATCH16_MANUAL_LEFT_PADDING_NO_POSITION_OVERRIDE",
                source_binding=modules["receipt"])


def diagnostic_layout(tok, record, *, category="R", prompt_index=0, device=None) -> dict:
    """Two sequences, new then true, and prediction-position complete target masks."""
    modules = _binding()
    names = {"R":"rewrite", "P":"rephrase", "N":"locality"}
    if category not in names or not isinstance(prompt_index,int) or prompt_index < 0:
        raise EvaluationBindingError("invalid diagnostic prompt selector")
    all_pairs = modules["evaluator"].counterfact_pairs([record])
    all_pairs["locality_target_new"] = modules["locality"].counterfact_locality_target_new_pairs([record])
    try:
        pairs = [all_pairs[names[category]+"_target_"+side][prompt_index] for side in ("new","true")]
    except IndexError as exc:
        raise EvaluationBindingError("diagnostic prompt index unavailable") from exc
    encoded = [(modules["contracts"].prompt_token_ids(tok,p.prompt),
                modules["contracts"].target_token_ids(tok,p.target)) for p in pairs]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad is None:
        raise EvaluationBindingError("no padding/eos token")
    length = max(len(prompt)+len(target)-1 for prompt,target in encoded)
    inputs = torch.full((2,length),int(pad),dtype=torch.long,device=device)
    attention = torch.zeros_like(inputs)
    labels = torch.zeros_like(inputs)
    target_mask = torch.zeros_like(inputs,dtype=torch.bool)
    for i,(prompt,target) in enumerate(encoded):
        full = prompt+target
        start = length-(len(full)-1)
        inputs[i,start:] = torch.tensor(full[:-1],device=device)
        attention[i,start:] = 1
        target_start = start+len(prompt)-1
        labels[i,target_start:] = torch.tensor(target,device=device)
        target_mask[i,target_start:] = True
    return dict(input_ids=inputs,attention_mask=attention,labels=labels,
                target_prediction_mask=target_mask,target_token_ids=[x[1] for x in encoded],
                identity=modules["integrity"].digest([pairs[0].case_id,prompt_index,pairs[0].prompt,pairs[0].target,pairs[1].target]),
                category=category,layout="DIAGNOSTIC_NEW_TRUE_PAIR2_FIXED_LEFT_PADDING",
                canonical_microbatch16_equivalence_claim=False)


def diagnostic_margin(model, tok, record, category="R", prompt_index=0):
    """Differentiable desired-margin scalar; full causal model, all target tokens.

    No backward or .grad mutation is performed here. Use AllPositionContraction
    around this forward, then its autograd.grad contraction. Prefix activations
    remain in the graph even though only target predictions enter NLL.
    """
    batch = diagnostic_layout(tok,record,category=category,prompt_index=prompt_index,device=_device(model))
    logits = model(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],use_cache=False).logits.float()
    logp = torch.log_softmax(logits,dim=-1)
    selected = logp.gather(-1,batch["labels"].unsqueeze(-1)).squeeze(-1)
    mask = batch["target_prediction_mask"]
    nll = -(selected*mask).sum(-1)/mask.sum(-1)
    raw_margin = nll[1]-nll[0]
    return -raw_margin if category == "N" else raw_margin
