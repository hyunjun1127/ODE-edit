"""CPU regression checks for CAKE port, sequential H, milestones and cap-2 DAG."""
import ast
import copy
from contextlib import redirect_stdout, ExitStack
import hashlib
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from official.baselines.cake import Cake_main as native
from official.baselines.registry import hparams, requests
from official.experiments import checkpoint as cp, milestones
from official.runners.server1.cake_submit import append_graph
from official.runners.server1.submit import graph_width
from official.runners.server1.cake_run import payload
from official.tracking.schema import config as tracking_config, metrics

ROOT = Path(__file__).resolve().parents[1]


class CakeTests(unittest.TestCase):
    def test_author_numeric_functions_unchanged(self):
        lock = json.loads((ROOT / 'tests/fixtures/cake-author-ast.json').read_text())
        functions = {n.name:n for n in ast.parse(Path(native.__file__).read_text()).body if isinstance(n, ast.FunctionDef)}
        for name, expected in lock['functions'].items():
            for node in ast.walk(functions[name]):
                if getattr(node, 'type_params', None) == []:
                    del node.type_params  # Python 3.12 adds an empty AST field.
            actual = hashlib.sha256(ast.dump(functions[name], include_attributes=False).encode()).hexdigest()
            self.assertEqual(actual, expected, name)

    def test_author_hparams_and_qwen_physical_layer_mapping(self):
        for model, layers, L2, clamp in [('llama3',list(range(4,9)),10,.5), ('gptj',list(range(3,9)),30,.75), ('qwen25',list(range(4,9)),1,4)]:
            h = hparams('CAKE', model)
            self.assertEqual((h.layers,h.L2,h.clamp_norm_factor), (layers,L2,clamp))
            self.assertEqual(sorted(map(int,h.causal_scores)), list(range(len(layers))))
        q = hparams('CAKE','qwen25')
        alpha = json.loads((ROOT/'hparams/ALPHAEDIT/qwen25.json').read_text())
        for k,v in alpha.items():
            if k not in ('alg_name','P_loc','stats_dir','device'):
                self.assertEqual(getattr(q,k),v,k)
        self.assertEqual(q.causal_scores['2'], .130932552514356)
        weights = native.compute_layer_weights(q.causal_scores,q.temperature)
        self.assertEqual(max(weights,key=weights.get),2)  # physical L6, not L2
        self.assertAlmostEqual(sum(weights.values()),1.)

    def test_requests_remain_nested_and_no_source_mutation(self):
        rows=[dict(case_id=0,requested_rewrite=dict(prompt='{} is',subject='A',target_new={'str':'B'},target_true={'str':'C'}))]
        saved=copy.deepcopy(rows); r=requests(rows,'CAKE','gptj')
        self.assertEqual(r[0]['target_new'],{'str':'B'})
        r[0]['target_new']['str']='changed'
        self.assertEqual(rows,saved)

    def test_two_batch_native_writer_matches_independent_linear_oracle(self):
        # Rectangular nn.Linear weights reproduce GPT-J's actual orientation.
        def model():
            m=torch.nn.Module();m.layers=torch.nn.ModuleList([torch.nn.Linear(3,2,bias=False) for _ in range(2)])
            with torch.no_grad():
                for p in m.parameters():p.zero_()
            for p in m.parameters():p.requires_grad_(False)
            return m
        m=model(); oracle=model(); H=torch.zeros(2,3,3); OH=H.clone(); P=torch.eye(3).repeat(2,1,1)
        hp=SimpleNamespace(layers=[0,1],rewrite_module_tmp='layers.{}',layer_module_tmp='layers.{}',
            fact_token='subject_last',clamp_norm_factor=.75,L2=2.,temperature=.1,causal_scores={'0':.3,'1':.2})
        req=[dict(case_id=i,prompt='{} is',subject=str(i),target_new={'str':'X'}) for i in range(2)]
        initial=copy.deepcopy(req); counts={'z':0,'keys':[0,0]}
        base=torch.tensor([[1.,.2,.3],[.5,.4,1.]])
        def keys(mm,layer):
            return base+float(layer)*.1+sum(p.sum() for p in mm.parameters())*.01
        def output(mm):
            return sum((p @ base.T for p in mm.parameters())).T
        def target(mm,request):
            return output(mm)[request['case_id']]+torch.tensor([.5,.2])*(request['case_id']+1)
        def get_keys(mm,tok,rs,hh,layer,ctx):
            counts['keys'][layer]+=1;return keys(mm,layer)
        def get_z(mm,tok,r,hh,layer,ctx):
            counts['z']+=1;return target(mm,r)
        def get_output(mm,*args,**kwargs):return None,output(mm)
        factory_tensor, factory_eye=torch.tensor,torch.eye
        def cpu_tensor(*args,**kwargs):kwargs.pop('device',None);return factory_tensor(*args,**kwargs)
        def cpu_eye(*args,**kwargs):kwargs.pop('device',None);return factory_eye(*args,**kwargs)
        with ExitStack() as stack:
            stack.enter_context(patch.object(torch.Tensor,'cuda',lambda self,*a,**k:self))
            stack.enter_context(patch.object(torch,'tensor',cpu_tensor));stack.enter_context(patch.object(torch,'eye',cpu_eye))
            stack.enter_context(patch.object(native,'compute_ks',get_keys));stack.enter_context(patch.object(native,'compute_z',get_z))
            stack.enter_context(patch.object(native,'get_module_input_output_at_words',get_output))
            stack.enter_context(patch.object(native,'CONTEXT_TEMPLATES_CACHE',[['{}']]))
            stack.enter_context(redirect_stdout(io.StringIO()))
            for batch in range(2):
                zs=torch.stack([target(oracle,r) for r in req],dim=1)
                allocation=torch.tensor(list(native.compute_layer_weights(hp.causal_scores,hp.temperature).values()))
                for i,w in enumerate(oracle.parameters()):
                    K=keys(oracle,i).T;resid=(zs-output(oracle).T)*allocation[i]/allocation[i:].sum()
                    delta=torch.linalg.solve(P[i]@(K@K.T+OH[i])+hp.L2*torch.eye(3),P[i]@K@resid.T)
                    w.add_(delta.T)
                for i in range(2):
                    K=keys(oracle,i).T;OH[i].add_(K@K.T)
                returned,h=native.apply_Cake_to_model(m,None,req,hp,cache_c=H,P=P)
                self.assertIs(returned,m);self.assertIs(h,H)
                for a,b in zip(m.parameters(),oracle.parameters()):torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)
                torch.testing.assert_close(H,OH,rtol=1e-6,atol=1e-7)
            self.assertEqual(counts,{'z':4,'keys':[4,4]})  # fit once; read keys + history once/layer/batch
        self.assertEqual(req,initial)
        self.assertGreater(float(H.norm()),0.)

    def test_milestones_survive_rollover_and_resume_rng_history_exactly(self):
        ident={k:'bound-'+k for k in cp.IDENTITY_FIELDS}
        W={'W':torch.zeros(2,3)};H={'4':torch.zeros(3,3)}
        random.seed(2);np.random.seed(2);torch.manual_seed(2)
        def step(w,h):
            w['W'].add_(torch.rand(2,3)+random.random()+np.random.random())
            h['4'].add_(w['W'].T@w['W'])
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            def save(b):
                return milestones.save(root,lock_path=root/'io.lock',batch=b,weights=W,cache_c=H,
                    contexts=[['{}']],evaluation_cursor={'edits':b*100},identity=ident,method='CAKE',evaluation_complete=True)
            save(0); expected={}
            for batch in range(1,21):
                step(W,H)
                if batch%5==0:
                    save(batch);expected[batch]=(W['W'].clone(),H['4'].clone())
            self.assertEqual(len(list((root/'checkpoint').glob('*.pt'))),1)
            self.assertEqual(len(list((root/'milestones').glob('edit-*'))),4)
            for b in (5,10,15,20):
                saved=cp.load(root/'milestones'/f'edit-{b*100:04d}',ident)
                self.assertTrue(torch.equal(saved['weights']['W'],expected[b][0]))
                self.assertTrue(torch.equal(saved['cache_c']['4'],expected[b][1]))
            saved=cp.load(root/'milestones/edit-0500',ident);cp.rng_restore(saved['rng'])
            for _ in range(15):step(saved['weights'],saved['cache_c'])
            self.assertTrue(torch.equal(saved['weights']['W'],W['W']))
            self.assertTrue(torch.equal(saved['cache_c']['4'],H['4']))
            with self.assertRaisesRegex(ValueError,'IDENTITY_MISMATCH'):
                cp.load(root/'milestones/edit-0500',dict(ident,code_commit='other'))

    def test_failed_checkpoint_retains_prior_milestone_and_lost_pin_is_recovered(self):
        ident={k:'bound-'+k for k in cp.IDENTITY_FIELDS}
        args=dict(weights={'W':torch.ones(2)},cache_c={'4':torch.eye(2)},contexts=[['{}']],
            evaluation_cursor={},identity=ident,method='CAKE',evaluation_complete=True)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);lock=root/'io.lock'
            milestones.save(root,lock_path=lock,batch=0,**args)
            # Simulate death between latest commit and milestone publication.
            cp.save(root/'checkpoint',batch=5,interval=5,**args)
            with patch.object(cp.torch,'save',side_effect=OSError('disk full')):
                with self.assertRaises(OSError):milestones.save(root,lock_path=lock,batch=10,**args)
            self.assertEqual(cp.load(root/'checkpoint',ident)['batch'],5)
            self.assertEqual(cp.load(root/'milestones/edit-0500',ident)['batch'],5)
            milestones.save(root,lock_path=lock,batch=10,**args)
            self.assertEqual(cp.load(root/'milestones/edit-0500',ident)['batch'],5)
            self.assertEqual(cp.load(root/'checkpoint',ident)['batch'],10)

    def test_two_lanes_wait_for_every_old_frontier_without_old_mutation(self):
        for width in (0,1,2,3):
            old=[dict(key=str(i),gpus=1,parents=[]) for i in range(width)]
            existing=dict(jobs=old,admitted_DAG_width=width);before=copy.deepcopy(existing)
            graph,combined=append_graph(existing)
            self.assertEqual(existing,before)
            self.assertEqual(graph['llama3-cf'],[str(i) for i in range(width)])
            self.assertEqual(graph['llama3-zsre'],graph['llama3-cf'])
            self.assertEqual(graph['gptj-cf'],['llama3-cf'])
            self.assertEqual(graph['qwen25-zsre'],['gptj-zsre'])
            self.assertEqual(graph_width([dict(key=k,gpus=1,parents=v) for k,v in graph.items()]),2)
            self.assertEqual(combined,max(2,width))

    def test_tracking_accepts_dataset_specific_endpoints_without_flucon(self):
        from official.tests.test_factual import retained_fixture
        for dataset in ('cf','zsre'):
            values=dict(server='server1',task_id='cake-six-2k-server1-20261011',model='gptj',model_family='gptj',
                writer='cake_native',baseline='CAKE',role='scientific',arm='gptj-'+dataset+'-cake',attempt='r1',
                source_sha='a'*40,config_sha='b'*64,dataset=dataset,metric_schema='official-baselines-scalar-v1',
                instruction_id='USER-OFFICIAL-BASELINES-20261008-R1')
            if dataset=='cf':values['generation_schedule']='DEFERRED_CHECKPOINT_EVALUATION'
            tracking_config(values)
            result=retained_fixture('cf')[0] if dataset=='cf' else dict(summary=dict(Efficacy=90.,Generalization=80.,Specificity=30.,Specificity_loc_ans=30.,requests=2000))
            if dataset == 'cf':
                result['cases'] = result['cases'] * 1000
                result['summary']['requests'] = 2000
            for prefix,edits in [('W0_first2000',0),('all_seen/post',2000)]:
                logged=payload(result,dataset,prefix,edits,values)
                metrics(logged,scientific=True,config_values=values)
                self.assertFalse(any('fluency' in k.lower() or 'consistency' in k.lower() for k in logged))


if __name__=='__main__':unittest.main()
