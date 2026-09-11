"""Joint full-sequence functional views and globally weighted prediction rows.

The live model is frozen. This module does not implement any controller or GGN
approximation: the shared functional kernel consumes ``PredictionBatch.logits``.
"""
from dataclasses import dataclass
import torch
from .contracts import Ledger, tensor_sha


class StateBoundary(RuntimeError):
    pass


class JointView:
    def __init__(self, model, names, ledger=None):
        self.model=model;self.names=tuple(names);self.ledger=ledger or Ledger()
        if not self.names or len(set(self.names))!=len(self.names):
            raise StateBoundary('DUPLICATE_OR_EMPTY_SUPPORT')
        self.parameters=dict(model.named_parameters())
        if any(n not in self.parameters for n in self.names):raise StateBoundary('UNKNOWN_WEIGHT')
        if any(p.requires_grad for p in self.parameters.values()):raise StateBoundary('LIVE_MODEL_NOT_FROZEN')
        if any(p.dtype!=torch.float32 for p in self.parameters.values()):raise StateBoundary('MODEL_NOT_FULL_FP32')
        self.signature=self._signature()
        selected=[self.parameters[n] for n in self.names]
        storage=[p.untyped_storage().data_ptr() for p in selected]
        if len(set(storage))!=len(storage):raise StateBoundary('ALIASED_SELECTED_WEIGHTS')
        self.entry=tuple(p.detach().clone() for p in selected)
        self.entry_sha=tuple(tensor_sha(p) for p in selected)

    def _signature(self):
        return tuple((n,p.data_ptr(),p._version,tuple(p.shape),str(p.dtype),str(p.device))
                     for n,p in self.model.named_parameters())

    def assert_live(self, *, bytes_check=False):
        if self._signature()!=self.signature:raise StateBoundary('LIVE_PARAMETER_IDENTITY_CHANGED')
        if bytes_check and tuple(tensor_sha(self.parameters[n]) for n in self.names)!=self.entry_sha:
            raise StateBoundary('LIVE_SELECTED_BYTES_CHANGED')

    def logits(self, weights, input_ids, attention_mask, positions):
        self.assert_live()
        if len(weights)!=len(self.names):raise StateBoundary('SUPPORT_LENGTH')
        for n,w in zip(self.names,weights):
            p=self.parameters[n]
            if w.shape!=p.shape or w.dtype!=p.dtype or w.device!=p.device:
                raise StateBoundary('WEIGHT_SHAPE_DTYPE_DEVICE')
        try:
            self.ledger.add('model_forward_invocations')
            self.ledger.add('forward_padded_tokens',input_ids.numel())
            self.ledger.add('forward_nonpadding_tokens',int(attention_mask.sum()))
            output=torch.func.functional_call(self.model,dict(zip(self.names,weights)),(),
              dict(input_ids=input_ids,attention_mask=attention_mask,use_cache=False),strict=False)
            all_logits=output.logits if hasattr(output,'logits') else output
            batch,pos=positions
            selected=all_logits[batch,pos]
            if selected.dtype!=torch.float32:raise StateBoundary('LOGITS_NOT_FP32')
            return selected
        finally:
            self.assert_live()

    def weights_for_delta(self,deltas):
        if len(deltas)!=len(self.entry):raise StateBoundary('DELTA_SUPPORT_LENGTH')
        return tuple(w+d.to(device=w.device,dtype=w.dtype) for w,d in zip(self.entry,deltas))


@dataclass
class PredictionBatch:
    view: JointView
    rows: tuple
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    positions: tuple
    target_ids: torch.Tensor
    context_index: torch.Tensor
    token_mean_weights: torch.Tensor
    context_weights: torch.Tensor

    def logits(self, weights):
        # PCG vectors may live on host. Device transfer remains differentiable;
        # model/forward precision is unchanged and transfer cost is explicit.
        transferred=tuple(w.to(device=self.input_ids.device) for w in weights)
        self.view.ledger.add('host_weight_transfer_bytes',sum(w.numel()*w.element_size()
            for w in weights if w.device!=self.input_ids.device))
        return self.view.logits(transferred,self.input_ids,self.attention_mask,self.positions)

    def context_nll(self,logits):
        logp=logits.log_softmax(-1)
        nll=-logp.gather(1,self.target_ids[:,None]).squeeze(1)
        return torch.zeros(len(self.rows),device=logits.device,dtype=logits.dtype).index_add(
            0,self.context_index,nll*self.token_mean_weights.to(logits.dtype))

    def current_loss(self,weights):
        return (self.context_nll(self.logits(weights)).double()*self.context_weights).sum()

    def metadata(self):
        return dict(target_ids=self.target_ids,context_index=self.context_index,
          token_mean_weights=self.token_mean_weights,context_weights=self.context_weights)


def pack(view,rows,pad_token_id,physical_microbatch=2):
    """Right padding is not a reduction: global row weights survive chunking."""
    if physical_microbatch<1:raise ValueError('INVALID_PHYSICAL_MICROBATCH')
    device=view.entry[0].device
    for start in range(0,len(rows),physical_microbatch):
        rr=tuple(rows[start:start+physical_microbatch]);width=max(len(r['input_ids']) for r in rr)
        ids=torch.full((len(rr),width),pad_token_id,device=device,dtype=torch.long)
        mask=torch.zeros_like(ids);batch=[];pos=[];targets=[];ci=[];tw=[]
        for j,r in enumerate(rr):
            if len(r['target_ids'])!=len(r['positions']) or not r['target_ids']:
                raise ValueError('PREDICTION_ALIGNMENT')
            if min(r['positions'])<0 or max(r['positions'])>=len(r['input_ids']):
                raise ValueError('PREDICTION_POSITION')
            length=len(r['input_ids']);ids[j,:length]=torch.tensor(r['input_ids'],device=device)
            mask[j,:length]=1;batch.extend([j]*len(r['positions']));pos.extend(r['positions'])
            targets.extend(r['target_ids']);ci.extend([j]*len(r['positions']))
            tw.extend([r['token_mean_weight']]*len(r['positions']))
        yield PredictionBatch(view,rr,ids,mask,
          (torch.tensor(batch,device=device),torch.tensor(pos,device=device)),
          torch.tensor(targets,device=device),torch.tensor(ci,device=device),
          torch.tensor(tw,device=device,dtype=torch.float64),
          torch.tensor([r['context_weight'] for r in rr],device=device,dtype=torch.float64))


def current_callback(view,rows,pad_token_id,physical_microbatch=2):
    """Logical full-batch gradient at one joint state; optimizer lives elsewhere."""
    def evaluate(deltas):
        weights=tuple(w.detach().requires_grad_(True) for w in view.weights_for_delta(deltas))
        grads=[torch.zeros_like(w) for w in weights];total=0.;chunks=0
        with view.ledger.time('current_forward_backward'):
            for batch in pack(view,rows,pad_token_id,physical_microbatch):
                loss=batch.current_loss(weights)
                gg=torch.autograd.grad(loss,weights)
                total+=float(loss.detach());chunks+=1
                for dest,g in zip(grads,gg):dest.add_(g.detach())
                view.ledger.add('current_backward_microbatches')
        if not torch.isfinite(torch.tensor(total)) or any(not torch.isfinite(g).all() for g in grads):
            raise FloatingPointError('CURRENT_NONFINITE')
        return total,tuple(grads),dict(physical_microbatches=chunks,logical_contexts=len(rows),
          physical_microbatch=physical_microbatch,context_weight_sum=sum(r['context_weight'] for r in rows),
          full_joint_weights=True,loss_reduction='token mean then global context/request weighted sum')
    return evaluate


def teacher(view,weights,rows,pad_token_id,physical_microbatch=2):
    """CPU-owned log probabilities, one row per exact sequence/context identity."""
    result=[]
    with torch.no_grad(),view.ledger.time('teacher_capture'):
        for batch in pack(view,rows,pad_token_id,physical_microbatch):
            logits=batch.logits(weights);lp=logits.log_softmax(-1);nll=batch.context_nll(logits)
            for j,row in enumerate(batch.rows):
                result.append(dict(identity=row['identity'],logp=lp[batch.context_index==j].cpu(),
                  reference_nll=nll[j].cpu(),target_ids=batch.target_ids[batch.context_index==j].cpu()))
    return result


def bind_teacher(batch,saved):
    lookup={r['identity']:r for r in saved}
    values=[lookup[r['identity']] for r in batch.rows]
    lp=torch.cat([r['logp'] for r in values]).to(batch.input_ids.device)
    target=torch.cat([r['target_ids'] for r in values]).to(batch.input_ids.device)
    if not torch.equal(target,batch.target_ids):raise StateBoundary('TEACHER_TARGET_MISMATCH')
    return dict(teacher_logp=lp,reference_nll=torch.stack([r['reference_nll'] for r in values]).to(lp.device))
