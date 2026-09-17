"""Full physical L8-weight differentiation and streamed response measurements.

Only canonical rewrite records and fixed-W0 S64 enter this module.  No native
L8 target, key, projector, residual parameterization, or controller is used.
Returned request/token rows and tensors are LOCAL-ONLY scientific evidence.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import time

import torch

from project.run_scripts.bg_tw_reference.ep_tw.model_adapter import _token_contracts


class ResponseError(RuntimeError):
    pass


def finite(x, label):
    if not bool(torch.isfinite(x).all()):
        raise ResponseError('NONFINITE_' + label)


class RepairResponse:
    def __init__(self, model, tok, teacher, *,
                 weight_name='model.layers.8.mlp.down_proj.weight', microbatch=16,
                 expected_vocab=128256):
        self.model, self.tok, self.teacher = model, tok, teacher
        self.weight_name, self.microbatch = weight_name, int(microbatch)
        self.parameters = dict(model.named_parameters())
        self.weight = self.parameters[weight_name]
        self.device = self.weight.device
        self.expected_vocab = int(expected_vocab)
        if model.training or any(p.requires_grad or p.grad is not None for p in self.parameters.values()):
            raise ResponseError('FROZEN_EVAL_MODEL_REQUIRED')
        if any(p.dtype != torch.float32 for p in self.parameters.values()):
            raise ResponseError('FP32_PARAMETERS_REQUIRED')
        if self.microbatch < 1:
            raise ResponseError('MICROBATCH')

    @contextmanager
    def _guard(self):
        params = [(name, p, p.data_ptr(), p._version) for name, p in self.parameters.items()]
        buffers = [(name, b, b.data_ptr(), b._version) for name, b in self.model.named_buffers()]
        modules = [(m, m.training, tuple(m._forward_hooks), tuple(m._forward_pre_hooks),
                    tuple(m._backward_hooks)) for m in self.model.modules()]
        cpu_rng = torch.random.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(self.device).clone() if self.device.type == 'cuda' else None
        try:
            yield
        finally:
            current = dict(self.model.named_parameters())
            if any(current[n] is not p or p.data_ptr() != ptr or p._version != version
                   or p.requires_grad or p.grad is not None for n, p, ptr, version in params):
                raise ResponseError('RESPONSE_PARAMETER_MUTATION')
            if any(b.data_ptr() != ptr or b._version != version for _, b, ptr, version in buffers):
                raise ResponseError('RESPONSE_BUFFER_MUTATION')
            if any((m.training, tuple(m._forward_hooks), tuple(m._forward_pre_hooks),
                    tuple(m._backward_hooks)) != (mode, fh, ph, bh)
                   for m, mode, fh, ph, bh in modules):
                raise ResponseError('RESPONSE_MODE_OR_HOOK_MUTATION')
            if not torch.equal(cpu_rng, torch.random.get_rng_state()):
                raise ResponseError('RESPONSE_CPU_RNG_MUTATION')
            if cuda_rng is not None and not torch.equal(cuda_rng, torch.cuda.get_rng_state(self.device)):
                raise ResponseError('RESPONSE_CUDA_RNG_MUTATION')

    def _replacement(self, weight, *, gradient=False):
        source = self.weight if weight is None else weight
        if source.shape != self.weight.shape or source.dtype != torch.float32:
            raise ResponseError('SELECTED_WEIGHT_SCHEMA')
        finite(source, 'SELECTED_WEIGHT')
        # A detached real selected-weight leaf, never an EP residual/custom node.
        return source.detach().to(self.device).requires_grad_(gradient)

    def _forward(self, ids, attention, weight):
        kwargs = dict(input_ids=ids, attention_mask=attention, use_cache=False)
        if weight is None:
            out = self.model(**kwargs)
        else:
            out = torch.func.functional_call(self.model, {self.weight_name: weight}, (),
                                             kwargs, strict=False)
        logits = out.logits.float()
        if logits.shape[-1] != self.expected_vocab:
            raise ResponseError('FULL_VOCAB_SIZE')
        return logits

    def _groups(self, records, target_key='target_new'):
        encoded = []
        for record in records:
            rw = record['requested_rewrite']
            if target_key not in rw or not rw[target_key] or not rw[target_key].get('str'):
                if target_key == 'target_new':
                    raise ResponseError('MISSING_NEW_TARGET')
                continue
            prompt = rw['prompt'].format(rw['subject'])
            p = _token_contracts().prompt_token_ids(self.tok, prompt)
            t = _token_contracts().target_token_ids(self.tok, rw[target_key]['str'])
            encoded.append((record, prompt, p, t))
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        if pad is None:
            raise ResponseError('PAD_TOKEN')
        for begin in range(0, len(encoded), self.microbatch):
            group = encoded[begin:begin+self.microbatch]
            length = max(len(p)+len(t)-1 for _, _, p, t in group)
            ids = torch.full((len(group), length), int(pad), dtype=torch.long, device=self.device)
            mask = torch.zeros_like(ids)
            positions = []
            for row, (_, _, p, t) in enumerate(group):
                offset = length-len(p)-len(t)+1
                ids[row, offset:] = torch.tensor((p+t)[:-1], device=self.device)
                mask[row, offset:] = 1
                positions.append(list(range(offset+len(p)-1, length)))
            yield group, ids, mask, positions

    def _losses(self, logits, group, positions):
        logp = torch.log_softmax(logits, dim=-1)
        return torch.stack([-logp[i, positions[i], :].gather(1,
            torch.tensor(row[3], device=self.device)[:, None]).mean()
            for i, row in enumerate(group)])

    def panel(self, records, weight=None, gradient=False):
        records = list(records)
        case_ids = [int(r['case_id']) for r in records]
        if len(set(case_ids)) != len(case_ids):
            raise ResponseError('DUPLICATE_PANEL_REQUEST')
        if not records:
            return dict(E=None, L=None, denominator=0, rows=[], strict_ids=[], preference_ids=[],
                        preference_status='EMPTY_PANEL', counts=dict(forwards=0, backwards=0,
                        input_tokens=0, scored_tokens=0)), None
        started = time.monotonic()
        leaf = self._replacement(weight, gradient=gradient) if gradient or weight is not None else None
        grad = torch.zeros_like(self.weight, device='cpu') if gradient else None
        rows, old = [], {}
        counts = dict(forwards=0, backwards=0, input_tokens=0, scored_tokens=0,
                      new_forwards=0, old_forwards=0)
        with self._guard():
            for target_key in ('target_new', 'target_true'):
                backward = gradient and target_key == 'target_new'
                with torch.set_grad_enabled(backward):
                    for group, ids, mask, positions in self._groups(records, target_key):
                        logits = self._forward(ids, mask, leaf)
                        losses = self._losses(logits, group, positions)
                        finite(losses, 'PANEL_NLL')
                        if backward:
                            part = torch.autograd.grad(losses.sum()/len(records), leaf)[0]
                            finite(part, 'PANEL_GRADIENT')
                            grad.add_(part.detach().cpu()); del part
                            counts['backwards'] += 1
                        for i, (record, prompt, _, targets) in enumerate(group):
                            cid = int(record['case_id'])
                            if target_key == 'target_true':
                                old[cid] = float(losses[i].detach())
                                continue
                            scores = logits[i, positions[i], :].detach()
                            target = torch.tensor(targets, device=self.device)
                            predictions = scores.argmax(-1)
                            competitors = scores.clone()
                            competitors.scatter_(1, target[:, None], -torch.inf)
                            competitor_ids = competitors.argmax(-1)
                            margins = scores.gather(1, target[:, None]).squeeze(1)-scores.gather(
                                1, competitor_ids[:, None]).squeeze(1)
                            finite(margins, 'TOKEN_MARGIN')
                            correct = predictions.eq(target)
                            rows.append(dict(case_id=cid, prompt=prompt, target=record['requested_rewrite']['target_new']['str'],
                                new_nll=float(losses[i].detach()), nll=float(losses[i].detach()),
                                target_token_ids=targets, token_predictions=predictions.cpu().tolist(),
                                token_correct=correct.cpu().tolist(), all_tokens_correct=bool(correct.all()),
                                competitor_ids=competitor_ids.cpu().tolist(), token_margins=margins.cpu().tolist()))
                            del scores, competitors, margins
                        counts['forwards'] += 1
                        counts['new_forwards' if target_key == 'target_new' else 'old_forwards'] += 1
                        counts['input_tokens'] += int(mask.sum())
                        counts['scored_tokens'] += sum(len(row[3]) for row in group)
                        del logits, losses
        for row in rows:
            row['old_nll'] = old.get(row['case_id'])
            row['preference_margin'] = (None if row['old_nll'] is None else row['old_nll']-row['new_nll'])
            row['preference_success'] = row['preference_margin'] is not None and row['preference_margin'] > 0
        mean = math.fsum(r['new_nll'] for r in rows)/len(rows)
        result = dict(E=mean, L=mean, denominator=len(rows), rows=rows,
            strict_ids=[r['case_id'] for r in rows if r['all_tokens_correct']],
            preference_ids=[r['case_id'] for r in rows if r['preference_success']],
            preference_status='AVAILABLE' if len(old)==len(records) else 'PARTIAL' if old else 'NOT_AVAILABLE',
            preference_denominator=len(old), reduction='TOKEN_MEAN_CANONICAL_CONTEXT1_REQUEST_MEAN',
            layout='MICROBATCH16_MANUAL_LEFT_PADDING_NO_POSITION_OVERRIDE', counts=counts,
            seconds=time.monotonic()-started, parameter_hook_rng_nonmutation=True)
        return result, grad

    def base(self, role='S64', weight=None, gradient=False):
        if gradient and role != 'S64':
            raise ResponseError('OBSERVER_GRADIENT_FORBIDDEN')
        indices = self.teacher.indices(role)
        if not indices:
            raise ResponseError('EMPTY_TEACHER_PANEL')
        leaf = self._replacement(weight, gradient=gradient) if gradient or weight is not None else None
        grad = torch.zeros_like(self.weight, device='cpu') if gradient else None
        rows = []; started = time.monotonic(); io = 0.
        with self._guard(), torch.set_grad_enabled(gradient):
            for index in indices:
                before = time.monotonic()
                ids, logp0, source_id = self.teacher.document(index, self.device)
                io += time.monotonic()-before
                if ids.shape != (1,257) or logp0.shape != (1,128,self.expected_vocab):
                    raise ResponseError('TEACHER_SCORE_SCHEMA')
                logits = self._forward(ids, torch.ones_like(ids), leaf)[:,128:256,:]
                logp = torch.log_softmax(logits, -1)
                loss = (logp0.exp()*(logp0-logp)).sum(-1).mean()
                finite(loss, 'BASE_KL')
                if gradient:
                    part = torch.autograd.grad(loss/len(indices), leaf)[0]
                    finite(part, 'BASE_GRADIENT'); grad.add_(part.detach().cpu()); del part
                rows.append(dict(index=index, source_row_id=source_id, kl=float(loss.detach()),
                    natural_nll=float(-logp.gather(-1,ids[:,129:257,None]).mean().detach()),
                    teacher_normalizer_max_abs=float(torch.logsumexp(logp0,-1).abs().max()),
                    input_tokens=257, scored_positions=128))
                del logits, logp, loss, logp0, ids
        value = math.fsum(r['kl'] for r in rows)/len(rows)
        return dict(B=value,D=value,role=role,denominator=len(rows),rows=rows,
            reduction='VOCAB_SUM_POSITION128_MEAN_DOCUMENT_MEAN',
            counts=dict(forwards=len(rows),backwards=len(rows) if gradient else 0,
                        input_tokens=257*len(rows),scored_tokens=128*len(rows),teacher_reads=len(rows)),
            seconds=dict(total=time.monotonic()-started,teacher_read=io),
            parameter_hook_rng_nonmutation=True), grad

    def _panel_response(self, Q, records, anchor, label):
        records = list(records); m = len(Q)
        if not records:
            return [], [], [], dict(jvp_forwards=0,input_tokens=0,scored_tokens=0)
        expected = [int(r['case_id']) for r in records]
        if expected != [r['case_id'] for r in anchor['rows']]:
            raise ResponseError('ANCHOR_REQUEST_ORDER')
        anchors = {r['case_id']:r for r in anchor['rows']}
        derivatives = {}; counts = dict(jvp_forwards=0,input_tokens=0,scored_tokens=0)
        w = self._replacement(None)
        for key in ('target_new','target_true'):
            for group,ids,mask,positions in self._groups(records,key):
                def measured(weight):
                    logits = self._forward(ids,mask,weight)
                    losses = self._losses(logits,group,positions)
                    values = [losses]
                    if key == 'target_new':
                        for i,(record,_,_,targets) in enumerate(group):
                            row = anchors[int(record['case_id'])]
                            if row['all_tokens_correct']:
                                scores = logits[i,positions[i],:]
                                t = torch.tensor(targets,device=self.device)
                                comp = torch.tensor(row['competitor_ids'],device=self.device)
                                values.append(scores.gather(1,t[:,None]).flatten()-scores.gather(1,comp[:,None]).flatten())
                    return torch.cat(values)
                columns = []
                for q in Q:
                    primal,tangent = torch.func.jvp(measured,(w,),(q.to(self.device),))
                    finite(primal,'GUARD_PRIMAL'); finite(tangent,'GUARD_JVP')
                    columns.append(tangent.detach().double().cpu())
                    counts['jvp_forwards'] += 1
                    counts['input_tokens'] += int(mask.sum())
                    counts['scored_tokens'] += sum(len(row[3]) for row in group)
                    del primal,tangent
                matrix = torch.stack(columns,dim=1)
                offset = len(group)
                for i,(record,_,_,targets) in enumerate(group):
                    cid = int(record['case_id'])
                    derivatives[(cid,key)] = matrix[i]
                    if key=='target_new' and anchors[cid]['all_tokens_correct']:
                        derivatives[(cid,'margins')] = matrix[offset:offset+len(targets)]
                        offset += len(targets)
        A = [sum((derivatives[(cid,'target_new')] for cid in expected),torch.zeros(m,dtype=torch.float64))/len(records)]
        s = [0.]; meta = [dict(panel=label,kind='mean_nll',unit='nats/token',slack=0.)]
        for cid in expected:
            row = anchors[cid]
            if row['all_tokens_correct']:
                for token_index,margin in enumerate(row['token_margins']):
                    if margin < 0:
                        raise ResponseError('STRICT_SUCCESS_NEGATIVE_MARGIN')
                    A.append(-derivatives[(cid,'margins')][token_index]); s.append(margin)
                    meta.append(dict(panel=label,kind='strict_token_margin',case_id=cid,token_index=token_index,
                        target_token_id=row['target_token_ids'][token_index],competitor_id=row['competitor_ids'][token_index],
                        unit='logit',slack=margin))
            if row['preference_success']:
                A.append(derivatives[(cid,'target_new')]-derivatives[(cid,'target_true')])
                s.append(row['preference_margin'])
                meta.append(dict(panel=label,kind='new_old_nll_preference',case_id=cid,
                                 unit='nats/token',slack=row['preference_margin']))
        return A,s,meta,counts

    def response(self, Q, current, past, anchor_current, anchor_past, *, include_guards=True):
        """Return small FP64 H/A/s/b; full-vocab tangents live for ONE doc only.

        ``include_guards=False`` is the R-GD proposal: it measures only Base
        Fisher/b and does not compute dummy Current/Past directional sweeps.
        Actual endpoint quality is still the caller's mandatory acceptance.
        """
        Q = [q.detach().cpu() for q in Q]; m = len(Q)
        if not 1 <= m <= 3 or any(q.shape != self.weight.shape or q.dtype != torch.float32 for q in Q):
            raise ResponseError('DIRECTION_SCHEMA')
        for q in Q: finite(q,'DIRECTION')
        start = time.monotonic(); H = torch.zeros((m,m),dtype=torch.float64)
        b = torch.zeros(m,dtype=torch.float64); w = self._replacement(None)
        counts = dict(base_jvp_forwards=0,guard_jvp_forwards=0,input_tokens=0,scored_tokens=0,
                      teacher_reads=0,directions=m,full_vocab_peak_documents=1)
        indices = self.teacher.indices('S64')
        with self._guard(), torch.no_grad():
            for index in indices:
                ids,logp0,_ = self.teacher.document(index,self.device)
                if ids.shape != (1,257) or logp0.shape != (1,128,self.expected_vocab):
                    raise ResponseError('TEACHER_SCORE_SCHEMA')
                def measured(weight):
                    logits = self._forward(ids,torch.ones_like(ids),weight)[:,128:256,:]
                    logp = torch.log_softmax(logits,-1)
                    loss = (logp0.exp()*(logp0-logp)).sum(-1).mean()
                    return logits,loss
                tangents = []; probability = None
                for j,q in enumerate(Q):
                    (logits,loss),(tangent,d_loss) = torch.func.jvp(measured,(w,),(q.to(self.device),))
                    finite(tangent,'BASE_LOGIT_JVP');finite(loss,'BASE_PRIMAL');finite(d_loss,'BASE_JVP')
                    if probability is None: probability = logits.softmax(-1).detach().squeeze(0)
                    tangents.append(tangent.detach().squeeze(0)); b[j] += float(d_loss)/len(indices)
                    counts['base_jvp_forwards'] += 1; counts['input_tokens'] += 257; counts['scored_tokens'] += 128
                    del logits,loss,tangent,d_loss
                # FP64 contractions in 16-position chunks avoid a FP64 full-doc
                # m*vocab expansion; p_N (not p0) defines the Fisher.
                for begin in range(0,128,16):
                    p = probability[begin:begin+16].double()
                    J = torch.stack([t[begin:begin+16] for t in tangents],dim=0).double()
                    means = (J*p[None,:,:]).sum(-1)
                    h = torch.einsum('jtv,ktv,tv->jk',J,J,p)-means@means.T
                    H.add_(h.cpu()/(128*len(indices)))
                    del J,p,means,h
                counts['teacher_reads'] += 1
                del probability,tangents,ids,logp0
            rows=[]; slacks=[]; metadata=[]
            if include_guards:
                for records,anchor,label in ((current,anchor_current,'Current-R'),(past,anchor_past,'Past-R')):
                    a,s,meta,c = self._panel_response(Q,records,anchor,label)
                    rows.extend(a);slacks.extend(s);metadata.extend(meta)
                    counts['guard_jvp_forwards'] += c['jvp_forwards']
                    counts['input_tokens'] += c['input_tokens']; counts['scored_tokens'] += c['scored_tokens']
        A = torch.stack(rows) if rows else torch.zeros((0,m),dtype=torch.float64)
        s = torch.tensor(slacks,dtype=torch.float64)
        for x,label in ((H,'FISHER'),(A,'GUARD_MATRIX'),(s,'GUARD_SLACK'),(b,'BASE_DIRECTIONAL_GRADIENT')): finite(x,label)
        return dict(H=H,A=A,s=s,b=b,guard_rows=metadata,counts=counts,
            seconds=time.monotonic()-start,parameter_hook_rng_nonmutation=True,
            base_probability='ACTUAL_ANCHOR_pN',gradient_teacher='FIXED_W0_p0',
            derivative_route='torch.func.jvp_DIRECT_FULL_L8_WEIGHT_FUNCTIONAL_CALL',
            include_guards=include_guards)
