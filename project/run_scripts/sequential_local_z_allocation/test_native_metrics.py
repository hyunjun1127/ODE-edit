"""Small CPU fixtures only; no actual Llama/native-GPU parity claim."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from .native import GeneralizedNativeFitter, NativeTrace, FitBoundary, select_projector
from .metrics import OnlineMetrics
from project.run_scripts.low_cost_write_donor_pilot.fitting import NativeSingletonFitter


def fake_compute_z(model,tok,request,hparams,layer,context_templates):
    target_init=torch.ones(3);delta=torch.zeros(3,requires_grad=True)
    kl_distr_init=torch.zeros((1,3));lookup_idxs=[0]
    opt=torch.optim.Adam([delta],lr=.1)
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()
        nll_loss_each=((delta-4)**2).mean().reshape(1)
        nll_loss=nll_loss_each.mean();kl_loss=delta.sum()*0
        weight_decay=.1*delta.norm();loss=nll_loss+kl_loss+weight_decay
        if loss < 5e-2:break
        if it==hparams.v_num_grad_steps-1:break
        loss.backward();opt.step()
        max_norm=hparams.clamp_norm_factor*target_init.norm()
        if delta.norm()>max_norm:
            with torch.no_grad():delta[...] = delta*max_norm/delta.norm()
    return target_init+delta


class Tokens(dict):
    def to(self,device):return Tokens({k:v.to(device) for k,v in self.items()})


class Tok:
    bos_token_id=1;unk_token_id=0;eos_token_id=2;pad_token_id=2;padding_side='right'
    def __init__(self,bos=True):self.bos=bos
    def encode(self,text,add_special_tokens=False):
        return ([1] if add_special_tokens and self.bos else [])+[3+ord(c)%6 for c in str(text).strip()]
    def decode(self,ids):return ''.join(chr(97+int(i)%6) for i in ids)
    def __call__(self,text,add_special_tokens=True,return_tensors=None,padding=False):
        if return_tensors is None:return {'input_ids':self.encode(text,add_special_tokens)}
        texts=text if isinstance(text,list) else [text]
        rows=[self.encode(t,add_special_tokens) for t in texts];length=max(map(len,rows))
        ids=torch.full((len(rows),length),2,dtype=torch.long);mask=torch.zeros_like(ids)
        for i,row in enumerate(rows):ids[i,:len(row)]=torch.tensor(row);mask[i,:len(row)]=1
        return Tokens(input_ids=ids,attention_mask=mask)


class Layer(nn.Module):
    def __init__(self):
        super().__init__();self.mlp=nn.Module();self.mlp.down_proj=nn.Linear(5,5,bias=False)
    def forward(self,x):return (torch.tanh(self.mlp.down_proj(x)),)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.embed=nn.Embedding(9,5);self.model=nn.Module()
        self.model.layers=nn.ModuleList([Layer() for _ in range(9)])
        self.norm=nn.Identity();self.head=nn.Linear(5,9,bias=False);self.config=SimpleNamespace(vocab_size=9)
    def forward(self,input_ids,attention_mask,use_cache=False):
        x=self.embed(input_ids)*attention_mask[...,None]
        x=x.cumsum(1)/attention_mask.cumsum(1).clamp_min(1)[...,None]
        for layer in self.model.layers:x=layer(x)[0]
        return SimpleNamespace(logits=self.head(self.norm(x)))


class TraceDict(dict):
    def __init__(self,model,layers,**kwargs):
        super().__init__();self.handles=[]
        for name in layers:
            def hook(module,args,output,name=name):self[name]=SimpleNamespace(output=output)
            self.handles.append(model.get_submodule(name).register_forward_hook(hook))
    def __enter__(self):return self
    def __exit__(self,*args):
        for h in self.handles:h.remove()


def parameter(model,name):
    try:return model.get_parameter(name)
    except AttributeError as e:raise LookupError(name) from e


class Teacher:
    def __init__(self,model):
        self.ids=torch.tensor([[1]+[3+i%6 for i in range(256)]])
        with torch.no_grad():self.logp=model(self.ids,torch.ones_like(self.ids)).logits[:,128:256,:].log_softmax(-1)
    def indices(self,role):
        if role=='S64':return [0,1]
        if role=='Dev128':return [2]
        raise ValueError(role)
    def document(self,index,device):return self.ids.to(device),self.logp.to(device),str(index)


def hp(layers):
    return SimpleNamespace(layers=layers,blue=True,L2=1,v_num_grad_steps=25,v_lr=.1,v_weight_decay=.5,
        clamp_norm_factor=.75,kl_factor=.0625,rewrite_module_tmp='model.layers.{}.mlp.down_proj',
        layer_module_tmp='model.layers.{}',v_loss_layer=8,lm_head_module='head',ln_f_module='norm')


class NativeTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def test_generalized_allowlist_original_unchanged(self):
        model=Tiny().eval();tok=Tok();f=object.__new__(GeneralizedNativeFitter)
        f.torch=torch;f.contexts=[['{}']];f.module=SimpleNamespace(CONTEXT_TEMPLATES_CACHE=f.contexts)
        m=torch.zeros(1,5,5);p=torch.eye(5)[None]
        for l in range(4,9):self.assertEqual(f._check(model,tok,hp([l]),m,p,l)[0],f'model.layers.{l}.mlp.down_proj.weight')
        with self.assertRaises(FitBoundary):NativeSingletonFitter._check(f,model,tok,hp([5]),m,p,5)
        self.assertIs(GeneralizedNativeFitter._functions,NativeSingletonFitter._functions)
        self.assertIs(GeneralizedNativeFitter.finalize,NativeSingletonFitter.finalize)
    def test_mapping_and_negative_hparam(self):
        stack=torch.stack([torch.eye(5)*(i+1) for i in range(5)])
        for layer in range(4,9):
            p,receipt=select_projector(stack,layer)
            self.assertEqual(receipt['source_index'],layer-4);self.assertTrue(torch.equal(p[0],stack[layer-4]))
        f=object.__new__(GeneralizedNativeFitter);f.torch=torch;f.contexts=[['{}']]
        f.module=SimpleNamespace(CONTEXT_TEMPLATES_CACHE=f.contexts);h=hp([4]);h.v_lr=.2
        with self.assertRaises(FitBoundary):f._check(Tiny(),Tok(),h,torch.zeros(1,5,5),stack[:1],4)
    def test_readonly_trace_loss_adam_clamp_exact(self):
        h=hp([4]);h.clamp_norm_factor=.01;r={'case_id':4}
        plain=fake_compute_z(None,None,r,h,4,[])
        with NativeTrace(fake_compute_z) as trace:observed=fake_compute_z(None,None,r,h,4,[])
        self.assertTrue(torch.equal(plain,observed));self.assertEqual(len(trace.rows),1)
        row=trace.rows[0]
        self.assertEqual(row['adam_updates'],24);self.assertEqual(row['loss_evaluations'],25)
        self.assertEqual(len(row['losses']),25);self.assertEqual(row['clamp_hits'],24)
        self.assertAlmostEqual(row['final_loss'],row['final_nll']+row['final_kl']+row['final_decay'],places=5)
        self.assertEqual(row['stop_reason'],'LOSS_BUDGET_25')
    def test_failed_native_preserves_partial_trace_before_exception(self):
        f=object.__new__(GeneralizedNativeFitter);f.module=SimpleNamespace(compute_z=fake_compute_z)
        def failing_fit(*args,**kwargs):
            return fake_compute_z(None,None,{'case_id':7},hp([4]),4,[])
        with patch.object(NativeSingletonFitter,'fit',side_effect=failing_fit), \
             patch.object(torch.optim.Adam,'step',side_effect=RuntimeError('TOY_NATIVE_STEP_FAILURE')):
            with self.assertRaisesRegex(RuntimeError,'TOY_NATIVE_STEP_FAILURE') as ctx:
                f.fit_capture(None,None,hp([4]),None,None,[{'case_id':7}],layer=4)
        partial=ctx.exception.native_partial
        self.assertEqual(partial['failed_requests'][0]['case_id'],7)
        self.assertEqual(len(partial['failed_requests'][0]['losses']),1)
        self.assertEqual(partial['completed_requests'],[])


class MetricTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71);torch.set_num_threads(1);self.model=Tiny().float().eval()
        for p in self.model.parameters():p.requires_grad_(False)
        self.teacher=Teacher(self.model)
        nethook=SimpleNamespace(get_parameter=parameter,get_module=lambda m,n:m.get_submodule(n),TraceDict=TraceDict)
        self.metric=OnlineMetrics(self.model,Tok(False),Tok(),SimpleNamespace(nethook=nethook),hp([4]),[['{}'],['x {}','zz {}']],self.teacher)
        self.records=[dict(case_id=i,requested_rewrite=dict(prompt='{} writes',subject='A'+str(i),
            target_new={'str':'abc' if i%2 else 't'},target_true={'str':'old'})) for i in range(3)]
    def test_native_context_mass_and_state(self):
        before={n:p.clone() for n,p in self.model.state_dict().items()}
        result=self.metric.training_E(self.records)
        self.assertEqual(result['contexts'],9)
        expected=sum(sum(r['context_nll'])/3 for r in result['rows'])/3
        self.assertAlmostEqual(result['E'],expected,places=6)
        self.assertTrue(all(torch.equal(p,before[n]) for n,p in self.model.state_dict().items()))
        self.assertFalse(any(m._forward_hooks for m in self.model.modules()))
    def test_canonical_reuses_historical_new_nll_strict(self):
        old=self.metric._canonical.current(self.records);new=self.metric.canonical(self.records)
        self.assertEqual(old['E'],new['E']);self.assertEqual(old['strict_ids'],new['strict_ids'])
        self.assertEqual([r['nll'] for r in old['rows']],[r['new_nll'] for r in new['rows']])
        for r in new['rows']:
            self.assertEqual(r['pair_success'],r['new_nll']<r['old_nll'])
            self.assertEqual(new['token_margins'][r['case_id']],min(r['token_margins']))
    def test_controller_semantics_no_plateau(self):
        score=self.metric.score(self.records,[]);c=score['controller']
        self.assertEqual(c['training_e'],score['details']['training']['E'])
        self.assertEqual(c['canonical_e'],score['details']['current']['E'])
        self.assertIsNone(c['past_h']);self.assertEqual(c['past_strict'],frozenset())
        self.assertEqual(score['official_P_N_access'],0);self.assertEqual(score['Dev_access'],0)
        self.assertFalse(score['canonical_Current_mean_is_guard'])
    def test_missing_old_disables_only_pair(self):
        records=deepcopy(self.records)
        for r in records:del r['requested_rewrite']['target_true']
        result=self.metric.canonical(records)
        self.assertEqual(result['pair_status'],'NOT_AVAILABLE');self.assertFalse(result['pair_ids'])
        self.assertEqual(len(result['token_margins']),3)
        self.assertIsNone(self.metric.score(records,[])['controller']['current_pair'])
    def test_past_and_self_teacher(self):
        result=self.metric.score(self.records[:1],self.records[1:]);c=result['controller']
        self.assertEqual(c['past_h'],result['details']['past']['E'])
        self.assertAlmostEqual(c['base_kl'],0.,places=7)
        self.assertEqual(self.metric.base('Dev128')['denominator'],1)


if __name__=='__main__':unittest.main()
