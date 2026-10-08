"""Frozen, passive first-trajectory GPT-J/native factual controls.

This is an owner-reviewed formula/packing control, not an independent public
evaluator oracle. It observes the FIRST forward that the official factual
scorer already performs. Extra LM forwards, fits, writes and pilots are zero.
No prompt/token/case identifiers or tensor payloads are emitted in the receipt.
"""
from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from official.baselines import registry
from official.baselines.easyedit.models.rome import repr_tools as easy_reprs
from official.baselines.blue.rome import repr_tools as blue_reprs
from official.baselines.easyedit.util import nethook
from official.evaluation import factual
from official.experiments import checkpoint
from official.experiments.prepare import digest, file_sha, ROOT

SCHEMA = 'official-server2-first-trajectory-native-parity-v1'
NLL_ATOL = 1e-4
NLL_RTOL = 0.0
AFFINE_ATOL = 1e-4
AFFINE_RTOL = 1e-5
HIDDEN = 4096
KEY_WIDTH = 16384


def require(value, code):
    if not value:
        raise ValueError(code)


def plan(manifest, dataset, method, records):
    """Persist the result BEFORE GPU measurements; no per-case raw in this plan."""
    require(dataset in ('cf','zsre') and method in registry.SPECS, 'PARITY_DATASET_METHOD')
    require(len(records) >= 100 and records[0]['occurrence_index'] == 1, 'PARITY_FIRST_NATIVE_COHORT')
    hparams = registry.hparams(method,'gptj')
    value = dict(schema=SCHEMA,status='PLAN_READY_OWNER_SOURCE_FORMULA_CONTROLS',actual_GPU=False,
        dataset=dataset,method=method,
        cohort=dict(ordered_occurrences=[1],first_record_sha256=digest(records[0]),
            selection='FIRST_CANONICAL_REWRITE_NEW_CANDIDATE; EXISTING_FIRST_SCORER_FORWARD_ONLY'),
        model_revision=manifest['model_revision'],tokenizer_sha256=manifest['tokenizer_sha256'],
        stream_sha256=manifest['streams'][dataset]['lock']['stream_sha256'],
        source=dict(code_commit=manifest['code_commit'],official_tree_sha256=manifest['official_tree_sha256'],
            hparams_sha256=file_sha(ROOT/'hparams'/method/'gptj.json'),native_hparams=asdict(hparams),
            factual_sha256=file_sha(Path(factual.__file__)),parity_sha256=file_sha(Path(__file__)),
            nethook_sha256=file_sha(Path(nethook.__file__)),
            native_easy_repr_sha256=file_sha(Path(easy_reprs.__file__)),
            native_BLUE_repr_sha256=file_sha(Path(blue_reprs.__file__))),
        tolerance=dict(candidate_nll=dict(atol=NLL_ATOL,rtol=NLL_RTOL),
            affine_readout=dict(atol=AFFINE_ATOL,rtol=AFFINE_RTOL),
            token_prediction='EXACT',candidate_token_boundary='EXACT',subject_lookup='EXACT'),
        independent_public_native_evaluator='NOT_AVAILABLE_IN_OFFICIAL_DISTRIBUTION',
        evidence_scope='OWNER_REVIEWED_NATIVE_FORMULA_PACKING_AND_SAME_CALL_MODEL_OPERATOR_CONTROLS',
        scientific_quality_gate=False,after_result_tolerance_relaxation=False,
        extra_LM_forward_calls=0,target_fit_calls=0,optimizer_calls=0,writer_calls=0,
        privacy='SCALAR_SHAPE_HASH_ONLY; NO_PROMPT_TOKEN_CASE_OR_TENSOR_PAYLOAD')
    value['plan_sha256'] = digest(value)
    return value


def _equal(left,right):
    if isinstance(left,torch.Tensor):
        return isinstance(right,torch.Tensor) and left.dtype == right.dtype and torch.equal(left,right)
    if isinstance(left,np.ndarray):
        return isinstance(right,np.ndarray) and left.dtype == right.dtype and np.array_equal(left,right)
    if isinstance(left,dict):
        return isinstance(right,dict) and left.keys() == right.keys() and all(_equal(left[k],right[k]) for k in left)
    if isinstance(left,(tuple,list)):
        return type(left) is type(right) and len(left) == len(right) and all(_equal(a,b) for a,b in zip(left,right))
    return left == right


def _state(model,engine):
    def identity(value):
        return (value.data_ptr(),value._version,tuple(value.shape),str(value.dtype),value.requires_grad)
    return dict(parameters={name:identity(value) for name,value in model.named_parameters()},
        buffers={name:identity(value) for name,value in model.named_buffers()},
        modules={name:(value.training,tuple(value._forward_hooks),tuple(value._forward_pre_hooks),
                       tuple(value._backward_hooks)) for name,value in model.named_modules()},
        config=deepcopy(model.config.to_dict()),native=deepcopy(engine.state_identity()))


def _close(actual,expected,code):
    require(actual.dtype == expected.dtype == torch.float32 and actual.shape == expected.shape
        and bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all()),code+'_SHAPE_DTYPE_FINITE')
    require(torch.allclose(actual,expected,atol=AFFINE_ATOL,rtol=AFFINE_RTOL),code+'_MISMATCH')
    return dict(shape=list(actual.shape),dtype='float32',max_abs_error=float((actual-expected).abs().max().item()),
                atol=AFFINE_ATOL,rtol=AFFINE_RTOL,close=True)


def _token_ids(tok,text,special):
    value = tok(text,add_special_tokens=special,truncation=False)['input_ids']
    require(type(value) is list and value and all(type(token) is int for token in value),'PARITY_NATIVE_TOKEN_IDS')
    return value


class Probe:
    """Temporary hooks capture only candidate 0 and its target-prediction sites."""
    def __init__(self,model,tok,engine,manifest,records,frozen_plan):
        self.model,self.engine,self.plan = model,engine,frozen_plan
        require(plan(manifest,frozen_plan['dataset'],engine.method,records) == frozen_plan,
                'PARITY_PREMEASUREMENT_PLAN_CHANGED')
        require(engine.batch == 1,'PARITY_FIRST_REAL_NATIVE_BATCH_REQUIRED')
        record = records[0]
        rewrite = record['requested_rewrite']
        prompt,target = rewrite['prompt'].format(rewrite['subject']),rewrite['target_new']['str']
        prefix,suffix = _token_ids(tok,prompt,True),_token_ids(tok,' '+target,False)
        full = _token_ids(tok,prompt+' '+target,True) if frozen_plan['dataset'] == 'cf' else prefix+suffix
        require(full == prefix+suffix,'PARITY_NATIVE_TARGET_PREFIX_BOUNDARY')
        self.query = dict(input_token_ids=full,target_start=len(prefix),target_token_ids=suffix)
        easy = easy_reprs.get_words_idxs_in_templates(tok,[rewrite['prompt']],[rewrite['subject']],'last')
        require(easy == blue_reprs.get_words_idxs_in_templates(tok,[rewrite['prompt']],[rewrite['subject']],'last'),
                'PARITY_NATIVE_SUBJECT_LAST_LOOKUP')
        self.positions = list(range(len(prefix)-1,len(prefix)+len(suffix)-1))
        self.layers = list(dict.fromkeys((engine.hparams.layers[0],engine.hparams.layers[-1])))
        self.linear_names = [f'transformer.h.{layer}.mlp.fc_out' for layer in self.layers]
        self.block_names = [f'transformer.h.{layer}' for layer in list(dict.fromkeys([*self.layers,27]))]
        self.handles,self.captured = [],{}
        self.phase = 'BEFORE'
        self.forward_calls_seen = 0
        self.receipt = None
        self.before,self.rng = _state(model,engine),checkpoint.rng_snapshot()
        self.started = time.monotonic()

    def _pre(self,module,args,kwargs):
        self.forward_calls_seen += 1
        if self.phase != 'BEFORE':
            return
        self.phase = 'CAPTURE'
        ids,mask = kwargs.get('input_ids'),kwargs.get('attention_mask')
        require(type(ids) is torch.Tensor and type(mask) is torch.Tensor and ids.shape == mask.shape,
                'PARITY_SCORER_INPUT_LAYOUT')
        length = len(self.query['input_token_ids'])
        require(ids.ndim == 2 and ids.shape[0] > 0 and ids.shape[1] >= length
            and ids[0,:length].detach().cpu().tolist() == self.query['input_token_ids']
            and bool((mask[0,:length] == 1).all()) and bool((mask[0,length:] == 0).all())
            and kwargs.get('use_cache') is False,'PARITY_SCORER_NATIVE_SUFFIX_RIGHT_PADDING')
        self.captured['candidate_width'] = ids.shape[1]

    def _hook(self,name):
        def hook(module,args,kwargs,output):
            if self.phase != 'CAPTURE':
                return
            positions = self.positions
            if name in self.linear_names:
                require(args and type(args[0]) is torch.Tensor and type(output) is torch.Tensor,
                        'PARITY_FC_OUT_TENSOR_INPUT_OUTPUT')
                self.captured[name] = (args[0][0,positions].detach().clone(),output[0,positions].detach().clone())
            elif name in self.block_names:
                hidden = nethook.get_hidden_state(output)
                require(isinstance(output,(torch.Tensor,tuple,list)) and hidden.ndim == 3
                    and hidden.shape[-1] == HIDDEN,'PARITY_GPTJ_BLOCK_TENSOR_TUPLE_LAYOUT')
                self.captured[name] = dict(container=type(output).__name__,hidden=hidden[0,positions].detach().clone())
        return hook

    def _post(self,module,args,kwargs,output):
        if self.phase != 'CAPTURE':
            return
        self.phase = 'CAPTURE_COMPLETE'
        logits = output.logits if hasattr(output,'logits') else output[0]
        self.captured['logits'] = logits[0,self.positions].detach().clone()

    def __enter__(self):
        try:
            self.handles.append(self.model.register_forward_pre_hook(self._pre,with_kwargs=True))
            for name in self.linear_names+self.block_names:
                self.handles.append(nethook.get_module(self.model,name).register_forward_hook(self._hook(name),with_kwargs=True))
            self.handles.append(self.model.register_forward_hook(self._post,with_kwargs=True))
        except BaseException:
            for handle in reversed(self.handles):
                handle.remove()
            raise
        return self

    def finish(self,observed):
        require(self.phase == 'CAPTURE_COMPLETE','PARITY_EXISTING_SCORER_FORWARD_NOT_OBSERVED')
        checks = {}
        with torch.no_grad():
            for name in self.linear_names:
                module = nethook.get_module(self.model,name)
                require(tuple(module.weight.shape) == (HIDDEN,KEY_WIDTH) and module.bias is not None
                    and tuple(module.bias.shape) == (HIDDEN,),'PARITY_GPTJ_NATIVE_LINEAR_BIAS_SHAPE')
                actual_input,actual_output = self.captured[name]
                checks[name] = _close(actual_output,F.linear(actual_input,module.weight,module.bias),'PARITY_NATIVE_FC_OUT')
            head = nethook.get_module(self.model,'lm_head')
            norm = nethook.get_module(self.model,'transformer.ln_f')
            require(head.bias is not None and head.weight.data_ptr() != self.model.get_input_embeddings().weight.data_ptr(),
                    'PARITY_GPTJ_UNTIED_BIASED_HEAD')
            manual = F.linear(norm(self.captured['transformer.h.27']['hidden']),head.weight,head.bias)
            logits = self.captured['logits']
            readout = _close(logits,manual,'PARITY_NATIVE_READOUT27_ONCE')
            require(torch.equal(logits.argmax(-1),manual.argmax(-1)),'PARITY_NATIVE_TOKEN_PREDICTIONS')
            gold = torch.tensor(self.query['target_token_ids'],device=logits.device)
            values = (-logits.float().log_softmax(-1).gather(1,gold[:,None]).squeeze(1)).detach().cpu().tolist()
        case = observed['cases'][0]
        require(case['occurrence_index'] == 1,'PARITY_STORED_ORDERED_FIRST_COHORT')
        if self.plan['dataset'] == 'cf':
            row = case['rewrite_observations'][0]['target_new']
        else:
            row = case['rewrite_observations'][0]
        require(all(row[key] == value for key,value in self.query.items()),'PARITY_STORED_NATIVE_SUFFIX_TOKEN_PREFIX')
        require(row['predicted_token_ids'] == logits.argmax(-1).detach().cpu().tolist(),
                'PARITY_STORED_NATIVE_TOKEN_PREDICTIONS')
        total = np.float32(0)
        for value in values:
            total = np.float32(total+np.float32(value))
        native_nll = float(np.float32(total/np.float32(len(values))))
        require(type(row['mean_nll']) in (int,float) and math.isfinite(row['mean_nll'])
            and math.isclose(row['mean_nll'],native_nll,abs_tol=NLL_ATOL,rel_tol=NLL_RTOL),
            'PARITY_NATIVE_FP32_CANDIDATE_NLL')
        require(len(row['nll_by_token']) == len(values) and all(math.isclose(a,b,abs_tol=NLL_ATOL,rel_tol=NLL_RTOL)
            for a,b in zip(row['nll_by_token'],values)),'PARITY_NATIVE_PER_TOKEN_NLL')
        self.receipt = dict(schema=SCHEMA,status='PASS_ACTUAL_OWNER_FORMULA_PARITY',
            actual_GPU=logits.device.type == 'cuda',plan_sha256=self.plan['plan_sha256'],
            code_commit=self.plan['source']['code_commit'],
            official_tree_sha256=self.plan['source']['official_tree_sha256'],
            factual_source_sha256=self.plan['source']['factual_sha256'],
            native_hparams_sha256=self.plan['source']['hparams_sha256'],
            method=self.engine.method,dataset=self.plan['dataset'],fc_out=checks,readout27=readout,
            block_output_schemas={name:dict(container=self.captured[name]['container'],
                hidden_shape=list(self.captured[name]['hidden'].shape)) for name in self.block_names},
            candidate_nll_close=True,candidate_nll_abs_error=abs(native_nll-row['mean_nll']),
            candidate_token_prefix_exact=True,token_predictions_exact=True,subject_lookup_exact=True,
            independent_public_native_evaluator='NOT_AVAILABLE_IN_OFFICIAL_DISTRIBUTION',
            independent_oracle_PASS=False,bitwise_full_evaluator_claim=False,
            evidence_scope=self.plan['evidence_scope'],scientific_quality_gate=False,
            extra_LM_forward_calls=0,existing_factual_forward_calls=self.forward_calls_seen,
            target_fit_calls=0,optimizer_calls=0,writer_calls=0,
            native_candidate_identity_sha256=digest(self.query),elapsed_seconds=time.monotonic()-self.started,
            observer_no_mutation='PENDING_EXIT_GUARD',RNG_restored='PENDING_EXIT_GUARD')
        return self.receipt

    def __exit__(self,error_type,error,traceback):
        for handle in reversed(self.handles):
            handle.remove()
        mutated = _state(self.model,self.engine) != self.before or not _equal(checkpoint.rng_snapshot(),self.rng)
        checkpoint.rng_restore(self.rng)
        if mutated:
            if error is None:
                raise ValueError('PARITY_NATIVE_W_H_CONTEXT_HOOK_RNG_MUTATED')
            if hasattr(error,'add_note'):
                error.add_note('PARITY_NATIVE_W_H_CONTEXT_HOOK_RNG_MUTATED; original error preserved')
        if self.receipt is not None and not mutated and error is None:
            self.receipt['observer_no_mutation'] = True
            self.receipt['RNG_restored'] = True
            if not self.receipt['actual_GPU']:
                self.receipt['status'] = 'PASS_CPU_FORMULA_FIXTURE_NOT_ACTUAL_GPU'
        self.captured.clear()
        return False


def observe(model,tok,engine,manifest,records,frozen_plan,evaluate_call):
    """Wrap ONE already planned actual factual evaluate, adding no LM forward."""
    with Probe(model,tok,engine,manifest,records,frozen_plan) as probe:
        observed = evaluate_call()
        receipt = probe.finish(observed)
    return observed,receipt
