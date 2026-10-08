"""CPU-only config/production-kernel regressions; never target-model evidence."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import unittest
import torch
from official.ours.config import resolve,profile,plain,knobs,canonical
from official.ours.core.jlz_interference_l1.cap_controller import RequestController
from official.ours.core.jlz_interference_l1.cap_optimizer import EfficiencyAdamAbs
from official.ours.core.jlz_v12r.optimizer import analytic_norm

FROZEN=Path(os.environ.get('PRICE_FROZEN_REFERENCE',
    '/data/janghj/ODE-edit/local/qwen-ours-m1-2k-20261008/preparation-v1/source'))

def old_module(name,relative):
    path=FROZEN/relative
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

def controller(p,anchors,prices):
    return RequestController(anchors,anchors[-1],tuple(range(len(anchors))),prices,
        n_exp=p['n_exp'],grace=p['K_grace'],threshold=p['tau_F'],c=p['c'],
        beta_base=p['beta_base'],cap_mode=p['cap_mode'],
        beta_max_native_scale=p['beta_max_scale'],max_updates=p['max_updates'])

class ConfigTests(unittest.TestCase):
    def test_defaults_and_preset(self):
        for model in ('llama3','qwen25','gptj'):
            r=resolve(model);p=r['price']
            self.assertEqual((p['beta_base'],p['c'],p['beta_max_scale']),(.75,.75,.75))
            self.assertEqual(p['lr'],.5 if model=='gptj' else .1)
            self.assertEqual((p['lambda_N'],p['lambda_KL'],p['tau_F']),(.5,.0625,.05))
            self.assertEqual((p['K_grace'],p['n_exp'],p['max_updates'],r['K_eval']),(12,4,24,25))
            self.assertEqual(r['eligible_layers'],tuple(range(3 if model=='gptj' else 4,9)))
            if model!='qwen25':self.assertEqual(p,resolve(model,preset='native')['price'])
        self.assertEqual(resolve('qwen25',preset='native')['price'],resolve('qwen25','qwen25-Q7-native')['price'])

    def test_immutable_hash_and_binding(self):
        r=resolve('qwen25');p=profile(r,{'writer':'memit'})
        self.assertEqual(knobs(p),r['price'])
        with self.assertRaises(TypeError):r['price']['lr']=1
        with self.assertRaises(ValueError):profile(r,{'lr':2})
        bad=plain(p);bad['lambda_N']=3
        with self.assertRaises(ValueError):knobs(bad)
        self.assertEqual(r['sha256'],hashlib.sha256(canonical({k:v for k,v in r.items() if k!='sha256'})).hexdigest())

    def test_arm_table(self):
        names=['Q0','Q1-lamN0001','Q2-beta150','Q3-beta250','Q4-beta400',
               'Q5-beta400-lamN05','Q6-beta400-c075','Q7-native']
        expected=[(.75,.75,.1,.5),(.75,.75,.1,.001),(1.5,1.5,.1,.001),
            (2.5,2.5,.1,.001),(4,4,.1,.001),(4,4,.1,.5),(4,.75,.1,.001),(4,4,.5,.001)]
        base=resolve('qwen25')['price']
        for name,values in zip(names,expected):
            p=resolve('qwen25','qwen25-'+name)['price']
            self.assertEqual(tuple(p[k] for k in ('beta_base','c','lr','lambda_N')),values)
            self.assertEqual(p['beta_max_scale'],p['beta_base'])
            for k in set(p)-{'beta_base','c','beta_max_scale','lr','lambda_N'}:self.assertEqual(p[k],base[k])

    def test_invalid(self):
        for override in ({'beta_base':0},{'beta_base':float('nan')},{'lr':float('inf')},
            {'c':None},{'cap_mode':'none'},{'K_grace':24},{'n_exp':-1},{'n_exp':True},
            {'lambda_N':-1},{'lambda_KL':-1},{'max_updates':0},{'betas':[1,.9]},
            {'unknown':1},{'tau_F':0},{'beta_max_scale':-1}):
            with self.subTest(override=override),self.assertRaises(ValueError):resolve('qwen25',override=override)
        p=resolve('qwen25',override={'cap_mode':'none','c':None})['price']
        c=controller(p,torch.ones(2,3),torch.ones(2,3));self.assertIsNone(c.caps)

    def test_update_budget_and_reactivation(self):
        p=resolve('qwen25',override={'max_updates':3,'K_grace':1})['price']
        c=controller(p,torch.ones(2,2),torch.tensor([[1.,1.],[2.,3.]]))
        for i,F in enumerate(([.1,0],[0,.1],[0,.1],[0,.1])):
            active,terminal=c.observe(torch.tensor(F),i)
            self.assertEqual(active.tolist(),[True,i>0])
            self.assertEqual(terminal,i==3)
            if not terminal:c.before_update(torch.tensor(F));c.record_update(active)
        self.assertEqual(c.t.tolist(),[3,2])
        with self.assertRaises(RuntimeError):c.record_update(active)

class FrozenKernelParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not FROZEN.is_dir():raise unittest.SkipTest('Frozen CPU reference absent; NOT_VERIFIED')
        sys.path.insert(0,str(FROZEN))
        package=types.ModuleType('_price_frozen_cpu');package.__path__=[str(FROZEN/'project/run_scripts/jlz_interference_l1')]
        sys.modules[package.__name__]=package
        cls.oldc=old_module('_price_old_controller','project/run_scripts/jlz_interference_l1/cap_controller.py')
        cls.oldo=old_module('_price_frozen_cpu.cap_optimizer','project/run_scripts/jlz_interference_l1/cap_optimizer.py')

    def test_defaults_bit_exact_controller_optimizer_norm(self):
        generator=torch.Generator().manual_seed(730)
        for model in ('llama3','qwen25','gptj'):
            p=resolve(model)['price'];anchors=torch.rand(3,4,generator=generator)+2
            prices=torch.tensor([[1.,2.,1.,4.],[3.,1.,2.,1.],[2.,3.,3.,2.]])
            old=self.oldc.RequestController(anchors,anchors[-1],(0,1,2),prices)
            new=controller(p,anchors,prices)
            R={i:torch.zeros(7,4) for i in range(3)}
            o=self.oldo.EfficiencyAdamAbs(R,lr=p['lr'],eps=p['eps'],betas=p['betas'])
            n=EfficiencyAdamAbs(R,lr=p['lr'],eps=p['eps'],betas=p['betas'],max_updates=p['max_updates'])
            for k in range(25):
                F=torch.tensor([.1,.08 if k>=2 else 0,.01 if k>=8 else .1,0])
                oldactive,ot=old.observe(F,k);active,nt=new.observe(F,k)
                self.assertTrue(torch.equal(active,oldactive));self.assertEqual(nt,ot)
                ln,gn=analytic_norm(R,anchors[-1],active,lambda_N=p['lambda_N'])
                lo,go=self.oldo.analytic_norm(R,anchors[-1],active)
                self.assertTrue(torch.equal(ln,lo))
                for l in R:self.assertTrue(torch.equal(gn[l],go[l]))
                if nt:break
                self.assertTrue(torch.equal(new.before_update(F),old.before_update(F)))
                g={l:torch.randn(R[l].shape,generator=generator) for l in R}
                Ro,_=o.step(R,g,active,old.caps,old.weights,old.beta)
                Rn,_=n.step(R,g,active,new.caps,new.weights,new.beta)
                for l in R:
                    self.assertTrue(torch.equal(Ro[l],Rn[l]))
                    self.assertTrue(torch.equal(o.m[l],n.m[l]));self.assertTrue(torch.equal(o.v[l],n.v[l]))
                old.record_update(active);new.record_update(active)
                self.assertEqual(old.receipt(),new.receipt());R=Rn

    def test_subject_coefficients_bit_exact(self):
        from official.ours.core.jlz_v12r.subject import evaluate
        old=old_module('_price_old_subject','project/run_scripts/jlz_v12r/subject.py').evaluate
        def fixture():
            rows=[dict(kind='rewrite',request=0,global_row=0,lookup=0,target=torch.tensor([1])),
                  dict(kind='kl',request=0,global_row=1,lookup=0,target=torch.tensor([-100]))]
            return dict(pack={'n_requests':1,'n_rw':1},teachers={0:torch.tensor([-.7,-1.2,-1.6])},
                groups=[dict(rows=rows,tokens={'input_ids':torch.ones(2,1,dtype=torch.long),
                    'attention_mask':torch.ones(2,1,dtype=torch.long)},cache={'key':torch.zeros(1),'residual':torch.zeros(1)})])
        class A:
            device='cpu';sites=(0,)
            def masked(self,g,v,capture=False):
                h=torch.tensor([[[.1,.3,-.2]],[[.7,-.1,.2]]])+v[0][:,None,:]
                return h,h,None,{0:h[:,0]}
            def head(self,x):return x
        for model in ('llama3','qwen25','gptj'):
            a=A();a.profile=profile(resolve(model),{})
            built={'v':{0:torch.zeros(2,3)}}
            x=evaluate(a,fixture(),built,backward=True,capture=True)
            y=old(a,fixture(),built,backward=True,capture=True)
            for k in ('F','nll','kl','active_mask'):self.assertTrue(torch.equal(x[k],y[k]),k)
            self.assertTrue(torch.equal(x['adjoint'][0],y['adjoint'][0]))

if __name__=='__main__':unittest.main()
