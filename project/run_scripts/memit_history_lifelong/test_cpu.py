"""Tiny fixtures execute the pinned upstream entrypoint, CPU only."""
import contextlib,importlib,io,json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch,numpy as np
from .method import observe,Proxy
from .io import save,signature,content,restore
from .metrics import summarize,paired,active_case_ids,FULL_BATCHES

BLUE=os.environ['MEMIT_BLUE_ROOT']
os.chdir(BLUE)
import sys
sys.path.insert(0,BLUE)
m=importlib.import_module('memit.memit_seq_main')

class Tests(unittest.TestCase):
    def fixture(self,hvalue=0.,fail=False):
        hp=SimpleNamespace(blue=False,layers=[4,5,6,7,8],rewrite_module_tmp='layers.{}',layer_module_tmp='layer.{}',fact_token='subject_last',mom2_dataset='wikipedia',mom2_n_samples=100000,mom2_dtype='float32',mom2_update_weight=15000,clamp_norm_factor=.75)
        w={f'layers.{l}.weight':torch.zeros((2,3)) for l in hp.layers}
        model=SimpleNamespace(named_parameters=lambda:list(w.items()))
        h=torch.eye(3).repeat(5,1,1)*hvalue
        req=[dict(case_id=i,prompt='{} test',subject='x',target_new={'str':' new'}) for i in range(2)]
        return hp,w,model,h,req
    def apply_fixture(self,hp,w,model,h,req,fail=False):
        prior=h.clone();before={k:v.clone() for k,v in w.items()};calls=[];solves=[];post=[]
        real_to=torch.Tensor.to;real_solve=torch.linalg.solve
        def to(x,*a,**kw):
            if a and a[0]=='cuda':a=('cpu',)+a[1:]
            return real_to(x,*a,**kw)
        def keys(model,tok,req,hp,layer,ctx):
            n=len(calls);calls.append(layer)
            if fail and n==7:raise RuntimeError('INJECTED_POST_KEY_FAILURE')
            k=torch.tensor([[1.,2.,.2],[.1,1.,.5]])+sum(float(x.sum()) for x in w.values())*.02
            if n>=5:post.append(k.clone())
            else:solves.append((k.T.clone(),layer))
            return k
        count=[0]
        def solve(a,b):
            i=count[0];count[0]+=1;k,_=solves[i]
            self.assertTrue(torch.equal(h,prior),'history appended before solve')
            expected=15000*torch.eye(3).double()+prior[i].double()+k.double()@k.double().T
            torch.testing.assert_close(a,expected,rtol=0,atol=0)
            return real_solve(a,b)
        with patch.object(torch.Tensor,'cuda',lambda x,*a,**kw:x),patch.object(torch.Tensor,'to',to),patch.object(torch.cuda,'synchronize',lambda:None),patch.object(m.nethook,'get_parameter',lambda model,k:w[k]),patch.object(m,'get_context_templates',lambda *a:[['{}']]),patch.object(m,'compute_z',lambda *a:torch.tensor([1.+a[2]['case_id'],2.])),patch.object(m,'compute_ks',keys),patch.object(m,'get_module_input_output_at_words',lambda *a,**kw:(None,torch.zeros((2,2)))),patch.object(m,'get_cov',lambda *a,**kw:torch.eye(3)),patch.object(m,'torch',Proxy(torch,linalg=Proxy(torch.linalg,solve=solve))),contextlib.redirect_stdout(io.StringIO()):
            with observe(m,hp,w,h,req,model) as receipt:
                returned,cache=m.apply_memit_seq_to_model(model,None,req,hp,cache_template=None,cache_c=h)
            self.assertIs(returned,model);self.assertIs(cache,h)
        self.assertEqual(calls,hp.layers*2)
        for i,k in enumerate(post):torch.testing.assert_close(h[i],prior[i]+k.T@k,rtol=0,atol=0)
        self.assertEqual(receipt['history_append_layers'],5)
        self.assertTrue(receipt['temporary_restore_exact'])
        return receipt
    def test_native_history_h0_and_two_batches(self):
        hp,w,model,h,req=self.fixture();self.apply_fixture(hp,w,model,h,req)
        endpoint=content(signature(w,h));self.assertGreater(float(h.norm()),0)
        self.assertEqual(content(signature(w,h)),endpoint)
        self.apply_fixture(hp,w,model,h,req);self.assertNotEqual(content(signature(w,h)),endpoint)
    def test_nonzero_history_changes_actual_writer(self):
        a=self.fixture(0);b=self.fixture(30000)
        self.apply_fixture(*a);self.apply_fixture(*b)
        self.assertFalse(torch.equal(a[1]['layers.4.weight'],b[1]['layers.4.weight']))
    def test_rollback_after_partial_history(self):
        hp,w,model,h,req=self.fixture();ew={k:v.clone() for k,v in w.items()};eh=h.clone();before=content(signature(w,h))
        with self.assertRaisesRegex(RuntimeError,'INJECTED'):
            try:self.apply_fixture(hp,w,model,h,req,fail=True)
            except BaseException:restore(w,ew,h,eh);raise
        self.assertEqual(content(signature(w,h)),before)
    def test_atomic_scalar_no_tensor_or_nonfinite_or_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'commit.json';save(p,{'ok':np.bool_(True)});self.assertTrue(json.loads(p.read_text())['ok'])
            with self.assertRaises(FileExistsError):save(p,{'changed':True})
            for i,x in enumerate([float('nan'),float('inf'),np.zeros(2),torch.zeros(2)]):
                q=Path(d)/str(i)
                with self.assertRaises((TypeError,ValueError)):save(q,{'x':x})
                self.assertFalse(q.exists())
    def test_metrics_token_vs_prompt_vs_strict_and_pairs(self):
        def row(i,c,n,success):return dict(identity=str(i),case_id=i,success=success,new_token_correct=c,new_token_count=n,new_strict=c==n,true_token_correct=0,true_token_count=1,true_strict=False,new_nll=1.,true_nll=2.,margin=1.)
        rows=[row(1,1,1,True),row(2,1,3,False)];r=summarize(rows,'RS')
        self.assertEqual(r['tf_token_micro'],.5);self.assertAlmostEqual(r['tf_prompt_macro'],2/3);self.assertEqual(r['tf_strict'],.5)
        self.assertEqual(summarize(rows,'NS')['tf_token_micro'],0)
        q=[dict(x,success=not x['success']) for x in rows];t=paired(rows,q);self.assertEqual((t['lost'],t['gained']),(1,1))
    def test_routing_and_no_cp(self):
        rows=list(range(10000));batches=[rows[i:i+100] for i in range(0,10000,100)]
        self.assertEqual(len(batches),100);self.assertEqual(sum(batches,[]),rows)
        src=Path(__file__).with_name('runner.py').read_text()
        self.assertIn('module.apply_memit_seq_to_model(',src);self.assertIn('returned_history is state',src)
        for forbidden in ['torch.save(','tensor_artifact(','checkpoint(','persist(']:self.assertNotIn(forbidden,src)
        self.assertEqual(FULL_BATCHES,[1,5,10,20,30,40,50,60,70,80,90,100])
    def test_canonical_observer_cpu_nonmutation(self):
        from project.run_scripts.blue_alphaedit_sequential_comparison.evaluation import bind_observation_only_package
        bind_observation_only_package()
        from project.run_scripts.alphaedit_strength_neutral_barrier.evaluator import evaluate_pairs,PromptTarget
        class Tok:
            pad_token_id=0;eos_token_id=0;bos_token_id=1;unk_token_id=None
            def encode(self,text,add_special_tokens=False):return [2] if text.strip()=='yes' else [3,2]
            def __call__(self,text,add_special_tokens=True):return {'input_ids':[1,3] if text=='short' else [1,3,3]}
        class Model:
            def __call__(self,input_ids,attention_mask,use_cache):
                logits=torch.zeros((*input_ids.shape,4));logits[:,:,2]=3.
                self.inputs=input_ids.clone();return SimpleNamespace(logits=logits)
        model=Model();weights={'x':torch.ones(2)};history=torch.zeros(2,2);before=signature(weights,history)
        rows=evaluate_pairs(model,Tok(),[PromptTarget(1,'rewrite',0,'short','yes'),PromptTarget(2,'rewrite',0,'long','no')],device=torch.device('cpu'),microbatch_size=16)
        self.assertEqual(signature(weights,history),before)
        self.assertEqual(rows[0]['token_correct'],[True]);self.assertEqual(rows[1]['token_correct'],[False,True])
        self.assertEqual(model.inputs[0,0].item(),0) # explicit left padding
        self.assertTrue(all(np.isfinite(r['nll']) for r in rows))

    def test_complete_reducer_without_forward(self):
        from .reducer import reduce_run
        # Mock only input storage, retain all 100 batch routing and real reduction.
        records=[{'case_id':i,'requested_rewrite':{'subject':str(i),'relation_id':'r','target_new':{'str':'new'}}} for i in range(10000)]
        def row(i,j):return dict(identity=f'{i}:{j}',case_id=i,success=i%2==0,new_token_correct=1,new_token_count=2,new_strict=False,true_token_correct=1,true_token_count=1,true_strict=True,new_nll=1.,true_nll=2.,margin=1.)
        def result(start,end):return {'metrics':{tag:{'rows':[row(i,j) for i in range(start,end) for j in range(n)]} for tag,n in [('RS',1),('PS',2),('NS',10)]}}
        original=Path.read_text
        def read(path,*a,**kw):
            if path.name in ('current.json','seen-full.json'):
                b=int(path.parent.name[1:]);return json.dumps(result((b-1)*100 if path.name=='current.json' else 0,b*100))
            return original(path,*a,**kw)
        with tempfile.TemporaryDirectory() as d,patch.object(Path,'read_text',read):
            out=reduce_run(d,records)
            self.assertEqual(out['final']['NS']['denominator'],100000)
            self.assertEqual(out['at_write']['RS']['lost'],0)
            self.assertEqual(out['first500_W5_to_W100']['PS']['denominator'],1000)
            self.assertFalse(out['save_checkpoints'])

    def test_submit_held_parser_and_no_postrelease_queries(self):
        from .submit import submit
        from .io import file_sha
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);script=root/'project/run_scripts/memit_history_lifelong/run.sbatch';script.parent.mkdir(parents=True);script.write_text('# fixture')
            lock=root/'execution.lock.json';lock.write_text(json.dumps({'source_root':str(root),'source_commit':'fixture','launcher_sha256':file_sha(script),'resource':{'cap':1,'memory_mib':121856}}))
            calls=[];submission=[]
            def command(argv):
                calls.append(argv)
                if argv[0]=='id':return 'janghj\n'
                if argv[0]=='squeue':return ''
                if argv[0]=='sbatch':submission.extend(argv);return '99999\n'
                if argv[:3]==['scontrol','show','job']:
                    return 'JobId=99999 JobName=odeedit_memit_history_10k_s3 UserId=janghj(1025) Priority=0 JobState=PENDING Dependency=(null) Requeue=0 TimeLimit=7-00:00:00 Partition=gpu AllocNode:Sid=ubuntu:1 ReqNodeList=ubuntu NumCPUs=8 CPUs/Task=8 ReqTRES=cpu=8,mem=119G,gres/gpu=1 MinMemoryNode=119G Command='+str(script)+' SubmitLine='+' '.join(submission)+' WorkDir='+str(root)+' TresPerNode=gres/gpu:1'
                if argv==['scontrol','release','99999']:return ''
                self.fail(str(argv))
            with patch('project.run_scripts.memit_history_lifelong.submit.run',command),contextlib.redirect_stdout(io.StringIO()):submit(lock)
            self.assertEqual(calls[-1],['scontrol','release','99999'])
            self.assertTrue(json.loads((root/'held-inspection.json').read_text())['all_pass'])
            self.assertEqual(json.loads((root/'release.json').read_text())['actual_initial'],'NOT_OBSERVED')

    def test_active_superseded_repeat(self):
        def r(i,t):return {'case_id':i,'requested_rewrite':{'subject':'s','relation_id':'r','target_new':{'str':t}}}
        self.assertEqual(active_case_ids([r(1,'a'),r(2,'a')]),{1,2})
        self.assertEqual(active_case_ids([r(1,'a'),r(2,'b')]),{2})

if __name__=='__main__':unittest.main()
