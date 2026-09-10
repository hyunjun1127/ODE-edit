"""Single grouped edit-gradient pass, then essence and analytic full action."""
import torch
import torch.nn.functional as F
from project.run_scripts.single_layer_cumulative_risk.objective import DirectObjective,essence_kl

class ProgressObjective(DirectObjective):
    def grouped(self,x,groups,backward=True,edit_only=False):
        n=len(self.requests);nc=len(self.contexts)
        values=[0.]*n;gradients=[];total_kl=0.
        self.last_request_gradients=None
        selected_for_hook=[];projected_input=None
        request_gradients=torch.zeros((n,*x.shape),device=x.device,dtype=torch.float32) if backward else None
        module=self.forward.model.get_submodule('model.layers.4.mlp.down_proj')
        def observe_forward(mod,args,out):
            nonlocal projected_input
            if selected_for_hook:
                projected_input=args[0].detach()@self.forward.u
        def observe_backward(mod,grad_input,grad_output):
            if selected_for_hook:
                # Existing backward, per-sequence separable loss. Undo the
                # group mean only; context averaging is already in grad_output.
                per=torch.bmm(grad_output[0].detach().transpose(1,2),projected_input)*25
                request_gradients[selected_for_hook]+=per
                self.ledger.add('request_gradient_observer_bmm')
        handles=[module.register_forward_hook(observe_forward),module.register_full_backward_hook(observe_backward)] if backward else []
        self.forward.active_weight=(self.forward.entry+x.detach()@self.forward.u.T).detach().requires_grad_(backward)
        try:
            with self.ledger.time('grouped_objective'),torch.set_grad_enabled(backward):
                for group in groups:
                    if backward:self.forward.active_weight.grad=None
                    for start in range(0,len(group),self.microbatch):
                        selected=group[start:start+self.microbatch]
                        selected_for_hook=selected if backward else []
                        for ci in range(nc):
                            items=[self.training[i][ci] for i in selected]
                            logits=self.forward(x,**self.pack([t[0] for t in items]))
                            loss=logits.new_zeros(())
                            for ii,(inp,target) in enumerate(items):
                                each=F.cross_entropy(logits[ii,len(inp)-len(target):len(inp)],torch.tensor(target,device=logits.device))
                                values[selected[ii]]+=float(each.detach())/nc
                                loss=loss+each/len(group)/nc
                            if backward:loss.backward();self.ledger.add('edit_backward')
                            del logits,loss
                    if backward:
                        gradients.append((self.forward.active_weight.grad@self.forward.u).double())
                        self.ledger.add('coefficient_gradient_pullback')
                selected_for_hook=[];projected_input=None
                if backward:self.forward.active_weight.grad=None
                if not edit_only:
                    for start in range(0,n,self.microbatch):
                        items=self.essence[start:start+self.microbatch];count=len(items)
                        logits=self.forward(x,**self.pack([t[0] for t in items]))
                        student=logits[torch.arange(count,device=logits.device),[t[1] for t in items]].log_softmax(-1)
                        kl=essence_kl(torch.stack(self.teacher[start:start+count]),student)*(count/n)
                        total_kl+=float(kl.detach())
                        if backward:(.0625*kl).backward();self.ledger.add('essence_backward')
                        del logits,student,kl
                xd=x.detach().double()
                numerator=((xd@self.metric)*xd).sum()+2*(xd*self.penalty_cross).sum()+self.penalty_constant
                native_action=numerator/self.j_native
                terms=dict(edit_nll=sum(values)/n,essence_kl_unweighted=total_kl,
                           normalized_native_action=float(native_action),request_edit_nll=values,
                           group_edit_nll=[sum(values[i] for i in group)/len(group) for group in groups])
                terms['objective']=terms['edit_nll']+.0625*total_kl+.1*float(native_action)
                if backward:
                    js=torch.stack(gradients);gradient=js.mean(0)
                    self.last_request_gradients=request_gradients
                    terms['request_gradient_group_pullback_difference']=[float((request_gradients[group].double().mean(0)-js[k]).norm()) for k,group in enumerate(groups)]
                    if not edit_only:
                        gradient=gradient+(self.forward.active_weight.grad@self.forward.u).double()
                        gradient=gradient+.2*(xd@self.metric+self.penalty_cross)/self.j_native
                        self.ledger.add('analytic_action_gradient')
                    if not torch.isfinite(gradient).all():raise FloatingPointError('NONFINITE_GRADIENT')
                else:js=gradient=None
            if not all(torch.isfinite(torch.tensor(terms[k])) for k in ('objective','edit_nll','essence_kl_unweighted')):
                raise FloatingPointError('NONFINITE_OBJECTIVE')
            return terms,gradient,js
        finally:
            self.forward.active_weight=None
            for handle in handles:handle.remove()
