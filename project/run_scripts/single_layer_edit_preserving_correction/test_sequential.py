"""CPU-only S routing/state regression. Not model/numerical T validation."""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from contextlib import ExitStack
from unittest.mock import patch
import torch
from . import sequential_state as st
from .sequential_runtime import SequentialRuntime
from .sequential_control import authority_check, inspect
from .sequential_runner import validate_lock
from .common import tensor_sha, digest
from .binding import quality_ok

def record(i,subject=None,target='new'):
    return dict(case_id=i,requested_rewrite=dict(subject=str(i) if subject is None else subject,relation_id='R',
        prompt='{} is',target_new={'str':target},target_true={'str':'old'}))

class StateTests(unittest.TestCase):
    def test_received_all_and_latest_overwrite(self):
        ledger=st.receive([], [record(1,'x'),record(2,'y'),record(3,'x','newer')])
        self.assertEqual([x['case_id'] for x in st.past64(ledger,[])],
                         [x['case_id'] for x in st.past64(list(ledger),[])])
        self.assertEqual({r['case_id'] for r in st.past64(ledger,[])},{2,3})
        self.assertEqual({r['case_id'] for r in st.past64(ledger,[record(4,'x')])},{2})
        self.assertEqual([r['status'] for r in st.registry(ledger)],['SUPERSEDED','ACTIVE','ACTIVE'])
        self.assertEqual(len(ledger),3)
    def test_past_b1_and_hash_priority(self):
        self.assertEqual(st.past64([], [record(1)]),[])
        ledger=st.receive([], [record(i) for i in range(100)])
        a=st.past64(ledger,[])
        self.assertEqual(len(a),64)
        self.assertEqual(a,st.past64(list(reversed(ledger)),[]))
    def test_duplicate_event_fail(self):
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):st.receive([record(1)], [record(1)])
    def test_no_official_fields_in_received(self):
        r=record(1);r.update(paraphrase_prompts=['SECRET'],neighborhood_prompts=['SECRET'])
        self.assertNotIn('SECRET',json.dumps(st.receive([], [r])))
    def test_link_checks_every_state(self):
        state=dict(W='w',M='m',rng='r',ledger='l',context='c')
        st.require_next(state,copy.deepcopy(state))
        for key in state:
            wrong=dict(state);wrong[key]='other'
            with self.assertRaises(ValueError):st.require_next(state,wrong)
    def test_history_once_binding(self):
        r=dict(layer=4,history_append=1,compute_ks=1,before_sha256='m0',after_sha256='m1',weight_sha256='w')
        st.require_finalizer([r],'m0','w','m1')
        for rows in ([],[r,r],[dict(r,history_append=0)],[dict(r,layer=8)],[dict(r,compute_z=100)]):
            with self.assertRaises(ValueError):st.require_finalizer(rows,'m0','w','m1')
    def test_atomic_reload_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'checkpoint.pt';w=torch.ones(2,3);m=torch.ones(1,3,3)
            value=dict(weight=w,M4=m,RNG={'x':[1,2]},received_ledger=st.receive([],[record(1)]),next_batch=2)
            ref=st.atomic_tensor(p,value)
            self.assertGreater(ref['bytes'],0)
            saved=torch.load(p,weights_only=True,map_location='cpu')
            self.assertTrue(torch.equal(saved['weight'],w));self.assertTrue(torch.equal(saved['M4'],m))
            before=p.read_bytes()
            with self.assertRaises(FileExistsError):st.atomic_tensor(p,dict(weight=torch.zeros_like(w)))
            self.assertEqual(p.read_bytes(),before)
            self.assertEqual(list(Path(td).glob('*.partial-*')),[])
    def test_actual_IO_error_not_suppressed(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'bad.pt'
            with patch('torch.save',side_effect=OSError(28,'No space left on device')):
                with self.assertRaises(OSError):st.atomic_tensor(p,dict(weight=torch.ones(2,3)))
            self.assertFalse(p.exists())
    def test_atomic_commit_create_once(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'commit.json';st.atomic_json(p,dict(commit_id='unique',history=1))
            with self.assertRaises(FileExistsError):st.atomic_json(p,dict(commit_id='double',history=2))
            self.assertEqual(json.loads(p.read_text())['history'],1)

class RoutingTests(unittest.TestCase):
    def test_authorized_four_not_old_S_or_M(self):
        a=dict(arms=list(st.S_ARMS),requests_per_chain=1000,batches=10,start='W0_ZERO_M4',cap=2,
            M_dependency=None,R_L_allowed=False,T='SKIPPED_USER_DIRECTED',full_numerical_validation='NOT_ESTABLISHED')
        authority_check(a)
        for k,v in [('arms',['N4']),('batches',100),('M_dependency','afterok:50050'),('R_L_allowed',True),('cap',3)]:
            with self.assertRaises(ValueError):authority_check(dict(a,**{k:v}))
    def test_inspection_no_old_M_T_dependency(self):
        lock={'source_root':'/frozen'};path=Path('/lock.json')
        text='UserId=janghj JobState=PENDING JobHeldUser NumCPUs=8 Requeue=0 server4 '+\
          'Command=/frozen/project/run_scripts/single_layer_edit_preserving_correction/sequential.sbatch /lock.json '+\
          'ArrayTaskId=0-3 ArrayTaskThrottle=2 Dependency=(null) TimeLimit=3-00:00:00 mem=59G gres/gpu=1'
        inspect(text,lock,path,2)
        for a,b in [('Dependency=(null)','Dependency=afterok:50050'),('ArrayTaskThrottle=2','ArrayTaskThrottle=4'),('mem=59G','mem=64G')]:
            with self.assertRaises(ValueError):inspect(text.replace(a,b),lock,path,2)
    def test_nonzero_history_allowed_but_inner_mutation_fails(self):
        rt=SequentialRuntime.__new__(SequentialRuntime)
        rt.M=torch.ones(1,3,3);rt.P=torch.eye(3)[None];rt.memory_version=rt.M._version;rt.projector_version=rt.P._version
        rt.context=[['{}']];rt.module=types.SimpleNamespace(CONTEXT_TEMPLATES_CACHE=copy.deepcopy(rt.context),COV_CACHE={})
        rt.base_guard=((),());rt.model=types.SimpleNamespace(parameters=lambda:[])
        with patch('project.run_scripts.single_layer_edit_preserving_correction.sequential_runtime.model_guard',return_value=((),())):
            rt.guard();rt.M.add_(1)
            with self.assertRaisesRegex(RuntimeError,'INNER_HISTORY'):rt.guard()
    def test_no_w0_reset_after_first_commit(self):
        rt=SequentialRuntime.__new__(SequentialRuntime);rt.M=torch.ones(1,2,2)
        with self.assertRaisesRegex(RuntimeError,'COLD_RESET'):rt.reset()
    def test_canonical_past_not_native_context_or_PN(self):
        class Tokens:
            def prompt_token_ids(self,tok,s):return [1,2]
            def target_token_ids(self,tok,s):return [3,4]
        class Oracle:
            def __init__(self,model,packs):self.packs=packs
        rt=SequentialRuntime.__new__(SequentialRuntime);rt.etok=object();rt.model=object();rt.oracles=[]
        name='project.run_scripts.single_layer_edit_preserving_correction.sequential_runtime.'
        with patch(name+'_token_contracts',return_value=Tokens()),patch(name+'FullWeightLlamaOracle',Oracle):
            old,rows=rt.past_oracle([record(1)])
            self.assertEqual(len(rows),2);self.assertEqual({r['kind'] for r in rows},{'canonical'})
            self.assertEqual({r['branch'] for r in rows},{'old','new'})
            self.assertEqual(rows[0]['positions'],[1,2]);self.assertEqual(rows[0]['labels'],[3,4])
    def test_past_is_individual_ID_guard_not_mean(self):
        anchor={'1:canonical:new':dict(kind='canonical',branch='new',nll=.1,strict=True),
                '1:canonical:old':dict(kind='canonical',branch='old',nll=.2,strict=False)}
        self.assertTrue(quality_ok(anchor,anchor)[0])
        c=copy.deepcopy(anchor);c['1:canonical:new']['nll']=.1002
        self.assertFalse(quality_ok(c,anchor)[0])
        c=copy.deepcopy(anchor);c['1:canonical:new']['strict']=False
        self.assertFalse(quality_ok(c,anchor)[0])
        c=copy.deepcopy(anchor);c['1:canonical:old']['nll']=.05
        self.assertFalse(quality_ok(c,anchor)[0])
    def test_runtime_no_cold_loop_and_order(self):
        import ast,inspect as pyinspect
        from .sequential_runner import run
        source=pyinspect.getsource(run);ast.parse(source)
        self.assertNotIn('rt.reset(',source)
        self.assertIn('rt.native_batch(',source)
        self.assertLess(source.index('correct('),source.index('finalize_batch('))
        self.assertLess(source.index("'COMMIT.json'"),source.index('observe_batch('))
        self.assertIn('range(1,11)',source)

class LoopTests(unittest.TestCase):
    def loop(self, fail_batch=None):
        from . import sequential_runner as runner
        from .common import write
        from project.run_scripts.baseline_mechanism_first.fixtures import capture_rng,restore_rng
        instances=[]
        class FakeRuntime:
            def __init__(self,lock,root):
                self.lock=lock;self.W=torch.zeros(2,3);self.M=torch.zeros(1,3,3)
                self.context=[['{}']];self.records=[record(i) for i in range(1000)]
                self.identity={k:'same' for k in ('W0','M0','P4','contexts','context_tokens','rng','teacher',
                    'records_digest','torch','transformers','microbatch','physical_layer')}
                self.timing={};self.oracles=[];self.entries=[];instances.append(self)
            def byte_hash_nonselected(self):return {'frozen':'unchanged'}
            def guard(self):pass
            def copy_weight(self,w):self.W.copy_(w)
            def native_batch(self,records,directory,batch):
                self.entries.append(self.W.clone())
                if batch==fail_batch:
                    self.W.add_(55);self.M.add_(11)
                    raise RuntimeError('INJECTED_NATIVE_FAILURE')
                self.W.add_(1)
                return {'weight':self.W.clone()}
            def finalize_batch(self,records):
                before=tensor_sha(self.M);self.M.add_(len(records))
                return [dict(layer=4,history_append=1,compute_ks=1,before_sha256=before,
                    after_sha256=tensor_sha(self.M),weight_sha256=tensor_sha(self.W))]
            def rollback_entry(self,w,m,rng):self.W.copy_(w);self.M.copy_(m);restore_rng(rng)
        def correction(rt,records,past,native,path,arm,cov):
            w=native['weight']+.25
            seal=runner.selection_seal(path.name,arm,w,[r['case_id'] for r in records],'a'*64)
            return w,seal,write(path/'SELECTION_SEALED.json',seal),{'native_fallback':False}
        with tempfile.TemporaryDirectory() as td,ExitStack() as stack:
            td=Path(td);technical=td/'technical.json'
            keys=('W0','M0','P4','contexts','context_tokens','rng','teacher','records_digest','torch','transformers','microbatch','physical_layer')
            technical.write_text(json.dumps(dict(identity={k:'same' for k in keys},EN_COV_resolution=1e-6)))
            lock=dict(S_root=str(td/'arms'),technical_evidence={'path':str(technical)},attempt='mock',execution={'head':'h'},
                lock_identity='l',sample_order=list(range(1000)))
            for name,value in [('validate_lock',lambda x:None),('SequentialRuntime',FakeRuntime),('correct',correction),
                               ('observe_batch',lambda *args:None)]:
                stack.enter_context(patch.object(runner,name,value))
            stack.enter_context(patch('torch.cuda.max_memory_allocated',return_value=0))
            stack.enter_context(patch('torch.cuda.empty_cache'))
            stack.enter_context(patch('gc.collect'))
            if fail_batch:
                with self.assertRaisesRegex(RuntimeError,'INJECTED'):runner.run(lock,'EN-F')
            else:runner.run(lock,'EN-F')
            root=td/'arms/EN-F/attempt-v1';rt=instances[0]
            if fail_batch:
                failure=json.loads((root/'failure.json').read_text())
                self.assertEqual(failure['committed_prefix'],1)
                self.assertEqual(failure['rollback'],'ENTRY_W_M_CONTEXT_RNG_RESTORED')
                self.assertTrue(torch.equal(rt.W,torch.full((2,3),1.25)))
                self.assertTrue(torch.equal(rt.M,torch.full((1,3,3),100.)))
                self.assertFalse((root/'B002/COMMIT.json').exists())
            else:
                self.assertEqual(len(list(root.glob('B*/COMMIT.json'))),10)
                self.assertEqual(len(list(root.glob('B*/checkpoint.pt'))),10)
                self.assertEqual(json.loads((root/'TERMINAL.json').read_text())['history_appends'],10)
                self.assertTrue((root/'S_INITIAL_VALID.json').exists())
                for i,entry in enumerate(rt.entries):self.assertTrue(torch.equal(entry,torch.full((2,3),i*1.25)))
                self.assertTrue(torch.equal(rt.M,torch.full((1,3,3),1000.)))
    def test_ten_batch_own_state_not_cold_reset(self):self.loop()
    def test_failure_rolls_back_only_open_batch(self):self.loop(2)

if __name__=='__main__':unittest.main()
