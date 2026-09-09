"""Full-token affine-weight NLL plus source-exact reverse essence KL."""
import torch
import torch.nn.functional as F
from .algebra import calibrated_eta,momentum_update

WEIGHT='model.layers.4.mlp.down_proj.weight'

def essence_kl(teacher_logp,student_logp):
    # Native compute_z: input=entry teacher, log_target=current student.
    return F.kl_div(teacher_logp,student_logp,log_target=True,reduction='batchmean')

class AffineForward:
    def __init__(self,model,entry,u,ledger):
        self.model,self.entry,self.u,self.ledger=model,entry,u,ledger
        self.parameter=dict(model.named_parameters())[WEIGHT]
    def __call__(self,a,**kw):
        pointer=self.parameter.data_ptr();version=self.parameter._version
        self.ledger.add('training_forward');self.ledger.add('training_sequences',kw['input_ids'].shape[0])
        self.ledger.add('training_tokens',int(kw['attention_mask'].sum()))
        w=self.entry+a@self.u.T
        out=torch.func.functional_call(self.model,{WEIGHT:w},(),dict(kw,use_cache=False),strict=False)
        assert self.parameter.data_ptr()==pointer and self.parameter._version==version
        return out.logits

class DirectObjective:
    def __init__(self,model,tok,requests,contexts,lookup_fn,entry,u,metric,j_native,ledger,microbatch=2):
        self.forward=AffineForward(model,entry,u,ledger)
        self.tok,self.requests,self.contexts=tok,requests,[c for group in contexts for c in group]
        self.metric,self.j_native,self.ledger=metric,j_native,ledger
        self.microbatch=microbatch;self.teacher=[];self.essence=[]
        self.training=[]
        assert tok.padding_side=='right'
        for request in requests:
            ids=tok(request['target_new']['str'],return_tensors='pt')['input_ids'][0].tolist()
            if ids[0] in (tok.bos_token_id,tok.unk_token_id):ids=ids[1:]
            assert ids
            # Native target-token IDs and context recipe, no activation intervention.
            rows=[]
            for c in self.contexts:
                prompt=(c.format(request['prompt'])+tok.decode(ids[:-1])).format(request['subject'])
                inp=tok(prompt)['input_ids']
                assert len(inp)>=len(ids)
                rows.append((inp,ids))
            self.training.append(rows)
            prompt='{} is a'
            idx=lookup_fn(prompt,request['subject'],tok,'subject_last',verbose=False)
            self.essence.append((tok(prompt.format(request['subject']))['input_ids'],idx))
        with ledger.time('essence_teacher'),torch.no_grad():
            for start in range(0,len(requests),microbatch):
                items=self.essence[start:start+microbatch]
                inputs=self.pack([x[0] for x in items])
                logits=model(**inputs,use_cache=False).logits
                lp=logits[torch.arange(len(items),device=logits.device),[x[1] for x in items]].log_softmax(-1)
                self.teacher.extend(lp.detach().unbind(0))
                ledger.add('teacher_forward');ledger.add('teacher_sequences',len(items));ledger.add('teacher_tokens',int(inputs['attention_mask'].sum()))

    def pack(self,sequences):
        device=self.forward.entry.device
        ids=torch.full((len(sequences),max(map(len,sequences))),self.tok.pad_token_id,device=device,dtype=torch.long)
        mask=torch.zeros_like(ids)
        for i,v in enumerate(sequences):ids[i,:len(v)]=torch.tensor(v,device=device);mask[i,:len(v)]=1
        return dict(input_ids=ids,attention_mask=mask)

    def evaluate(self,a,backward=True):
        if a.grad is not None:a.grad=None
        total_nll=total_kl=0.;n=len(self.requests);nc=len(self.contexts)
        with self.ledger.time('direct_objective'),torch.set_grad_enabled(backward):
            for start in range(0,n,self.microbatch):
                count=min(self.microbatch,n-start)
                for ci in range(nc):
                    items=[self.training[i][ci] for i in range(start,start+count)]
                    logits=self.forward(a,**self.pack([x[0] for x in items]))
                    loss=logits.new_zeros(())
                    for i,(inp,target) in enumerate(items):
                        selected=logits[i,len(inp)-len(target):len(inp)]
                        loss=loss+F.cross_entropy(selected,torch.tensor(target,device=logits.device))/n/nc
                    total_nll+=float(loss.detach())
                    if backward:loss.backward();self.ledger.add('backward')
                    del logits,loss
                items=self.essence[start:start+count]
                logits=self.forward(a,**self.pack([x[0] for x in items]))
                student=logits[torch.arange(count,device=logits.device),[x[1] for x in items]].log_softmax(-1)
                teacher=torch.stack(self.teacher[start:start+count])
                kl=essence_kl(teacher,student)*(count/n)
                total_kl+=float(kl.detach())
                if backward:(.0625*kl).backward();self.ledger.add('backward')
                del logits,student,teacher,kl
            j=((a@self.metric)*a).sum()/self.j_native
            j_value=float(j.detach())
            if backward:(.1*j).backward();self.ledger.add('penalty_backward')
        result=dict(edit_nll=total_nll,essence_kl_unweighted=total_kl,normalized_native_action=j_value,
                    objective=total_nll+.0625*total_kl+.1*j_value)
        if not all(torch.isfinite(torch.tensor(v)) for v in result.values()):raise FloatingPointError('NONFINITE_OBJECTIVE')
        if backward and not torch.isfinite(a.grad).all():raise FloatingPointError('NONFINITE_GRADIENT')
        return result
