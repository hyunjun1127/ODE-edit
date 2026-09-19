"""CPU fixtures only: these tests do not establish Llama/native parity."""
import types
import unittest
from unittest.mock import patch
import torch
from .z_hook import (ZHookConfig,ZHookBoundary,normalize_requests,prepare_batch,
                     freeze_and_clamp,capture_prefix,suffix_hidden,native_losses,
                     compute_z_batch,instrument_native_compute_z)


class Tokens(dict):
    def to(self,device):return Tokens({k:v.to(device) for k,v in self.items()})


class Tokenizer:
    padding_side='right';bos_token_id=1;unk_token_id=0
    def __call__(self,text,return_tensors=None,padding=False):
        texts=text if isinstance(text,list) else [text]
        rows=[[1]+[2+sum(map(ord,w))%11 for w in s.split()] for s in texts]
        width=max(map(len,rows))
        return Tokens(input_ids=torch.tensor([x+[0]*(width-len(x)) for x in rows]),
                      attention_mask=torch.tensor([[1]*len(x)+[0]*(width-len(x)) for x in rows]))
    def decode(self,ids):return ' '.join('v'+str(int(x)) for x in ids)


class Layer(torch.nn.Module):
    def __init__(self,width,index):
        super().__init__();self.linear=torch.nn.Linear(width,width,bias=False);self.index=index;self.calls=0
    def forward(self,hidden_states,attention_mask=None,position_ids=None,cache_position=None,
                position_embeddings=None,use_cache=False,past_key_value=None,output_attentions=False):
        self.calls+=1
        return (hidden_states+torch.tanh(self.linear(hidden_states)),)


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embed_tokens=torch.nn.Embedding(16,4);self.layers=torch.nn.ModuleList([Layer(4,i) for i in range(3)]);self.norm=torch.nn.LayerNorm(4)
    def forward(self,input_ids,attention_mask,**kwargs):
        hidden=self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden=layer(hidden,attention_mask=attention_mask,position_ids=torch.arange(input_ids.shape[1])[None],
                         cache_position=torch.arange(input_ids.shape[1]),position_embeddings=(torch.ones_like(hidden),torch.zeros_like(hidden)),use_cache=False)[0]
        return self.norm(hidden)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.model=Backbone();self.lm_head=torch.nn.Linear(4,16,bias=False)
        self.config=types.SimpleNamespace(model_type='llama',_attn_implementation='eager')
        self.eval()
        for p in self.parameters():p.requires_grad_(False)


def hp():return types.SimpleNamespace(layer_module_tmp='model.layers.{}',lm_head_module='lm_head',
    ln_f_module='model.norm',v_loss_layer=2,v_num_grad_steps=25,v_lr=.1,v_weight_decay=.5,
    clamp_norm_factor=.75,kl_factor=.0625,fact_token='subject_last')


def request(i=1,target=' one two'):return dict(case_id=i,prompt='{} likes',subject='name',target_new={'str':target})


def lookup(*args,**kwargs):return 1


def compute_z():
    """CPU-only arithmetic fixture for source-clone instrumentation layout."""
    delta=torch.tensor([1.],requires_grad=True)
    for it in range(2):
        nll_loss=delta.square().sum();kl_loss=delta.sum()*0;weight_decay=delta.sum()*0
        loss=nll_loss+kl_loss+weight_decay
        loss.backward()
    return delta


class ZHookTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9);self.model=Model();self.tok=Tokenizer()
        self.old=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32=self.old[0]
        torch.backends.cudnn.allow_tf32=self.old[1]
    def test_config_rejects_unregistered_batch(self):
        with self.assertRaises(ZHookBoundary):ZHookConfig(2)
    def test_normalize_does_not_mutate(self):
        x=[request(target='one')];y=normalize_requests(x)
        self.assertEqual(x[0]['target_new']['str'],'one');self.assertEqual(y[0]['target_new']['str'],' one')
    def test_variable_target_and_right_padding(self):
        batch=prepare_batch(self.tok,[request(),request(2,' one')],[['{}','a {}']],hp(),lookup,'cpu')
        self.assertEqual(batch['targets'].ne(-100).sum(1).tolist(),[2,2,1,1])
        self.assertEqual(batch['rw_rows'],[0,1,3,4]);self.assertEqual(batch['kl_rows'],[2,5])
    def test_capture_stops_and_removes_hooks(self):
        batch=prepare_batch(self.tok,[request()],[['{}']],hp(),lookup,'cpu')
        prefix=capture_prefix(self.model,hp(),0,batch['tokens'])
        self.assertEqual([x.calls for x in self.model.model.layers],[1,0,0])
        self.assertEqual(len(self.model.model.layers[0]._forward_hooks),0)
        self.assertIn('position_embeddings',prefix['kwargs'])
    def test_kl_uses_final_not_loss_layer(self):
        h=hp();h.v_loss_layer=0
        batch=prepare_batch(self.tok,[request()],[['{}']],h,lookup,'cpu')
        prefix=capture_prefix(self.model,h,0,batch['tokens']);delta=torch.zeros(1,4,requires_grad=True)
        initial=prefix['hidden'][[0],[1]]
        loss,final=suffix_hidden(self.model,h,0,prefix,delta,batch)
        values=native_losses(self.model,h,batch,loss,final,delta,initial,None)
        expected=self.model.lm_head(self.model.model.norm(final[batch['kl_rows'],batch['kl_cols']])).log_softmax(-1)
        wrong=self.model.lm_head(self.model.model.norm(loss[batch['kl_rows'],batch['kl_cols']])).log_softmax(-1)
        self.assertTrue(torch.equal(values[-1],expected));self.assertFalse(torch.equal(expected,wrong))
    def test_frozen_adam_momentum_is_undone(self):
        delta=torch.ones(2,4,requires_grad=True);opt=torch.optim.Adam([delta],lr=.1)
        delta.square().sum().backward();opt.step();frozen=delta.detach().clone()
        opt.zero_grad();delta[1].square().sum().backward();opt.step()
        self.assertFalse(torch.equal(delta[0],frozen[0]))
        freeze_and_clamp(delta,frozen,torch.tensor([True,False]),torch.tensor([100.,100.]))
        self.assertTrue(torch.equal(delta[0],frozen[0]))
    def test_25_loss_24_adam_and_one_prefix(self):
        targets,receipt=compute_z_batch(self.model,self.tok,[request()],hp(),0,[['{}']],lookup)
        self.assertEqual(receipt['loss_steps'],[25]);self.assertEqual(receipt['adam_steps'],[24])
        self.assertEqual(self.model.model.layers[0].calls,1);self.assertEqual(targets.shape,(4,1))
    def test_request_batches_do_not_average_independent_gradients(self):
        z,r=compute_z_batch(self.model,self.tok,[request(),request(2)],hp(),0,[['{}']],lookup,config=ZHookConfig(16,True))
        self.assertTrue(torch.equal(z[:,0],z[:,1]));self.assertTrue(torch.equal(r['gradients'][0][0],r['gradients'][1][0]))
        _,single=compute_z_batch(self.model,self.tok,[request()],hp(),0,[['{}']],lookup,config=ZHookConfig(1,True))
        self.assertTrue(torch.allclose(r['gradients'][0][0],single['gradients'][0][0],atol=1e-6,rtol=1e-6))
    def test_call_local_prefix_refresh(self):
        _,r1=compute_z_batch(self.model,self.tok,[request()],hp(),0,[['{}']],lookup)
        _,r2=compute_z_batch(self.model,self.tok,[request(target=' other')],hp(),0,[['{}']],lookup)
        self.assertNotEqual(r1['input_identity'],r2['input_identity']);self.assertEqual(self.model.model.layers[0].calls,2)
    def test_nonfinite_is_technical_failure(self):
        with torch.no_grad():self.model.lm_head.weight.fill_(float('nan'))
        with self.assertRaises(FloatingPointError):compute_z_batch(self.model,self.tok,[request()],hp(),0,[['{}']],lookup)
    def test_unknown_native_layout_rejected(self):
        with self.assertRaises(ZHookBoundary):instrument_native_compute_z(hp,lambda *a:None)
    def test_native_instrumentation_does_not_replace_arithmetic(self):
        events=[]
        instrumented=instrument_native_compute_z(compute_z,lambda *args:events.append(args[0]))
        self.assertTrue(torch.equal(compute_z(),instrumented()))
        self.assertEqual(events,['loss','gradient','loss','gradient'])


if __name__=='__main__':unittest.main()
