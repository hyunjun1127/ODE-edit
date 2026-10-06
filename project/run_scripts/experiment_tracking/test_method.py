"""Production helper + SH4 producer, synthetic scalar rows, fake SDK only."""
import json
from pathlib import Path
import tempfile
import unittest
from . import schema
from .method import harmonic,AxisState,PREFIXES,NEIGHBOR_PREFIXES
from .identity import create
from .worker import session
from .test_tracking import CFG,SDK

CFG_METHOD=dict(CFG,config_sha='b'*64,model='gptj',model_family='GPTJ',writer='memit',
    role='scientific',metric_schema=schema.COMPARISON_SCHEMA)

class MethodSDK(SDK):
    def __init__(self):super().__init__();self.definitions=[];self.history=[];self.next_step=0
    def define_metric(self,*args,**kwargs):self.definitions.append((args,kwargs))
    def log(self,values,step=None):
        super().log(values,step)
        s=self.next_step if step is None else step;self.next_step=s+1
        self.history.append(dict(values,_step=s))
    def scan_history(self,**kw):
        return iter([r for r in self.history if all(k in r for k in kw.get('keys',[]))
            and kw.get('min_step',0)<=r['_step']<kw.get('max_step',10**9)])

def execute(payloads,sdk=None):
    sdk=sdk or MethodSDK();cfg=schema.bind_job_identity(CFG_METHOD,dict(SLURM_JOB_ID='42',SLURM_ARRAY_JOB_ID='40',SLURM_ARRAY_TASK_ID='0',SLURM_STEP_ID='-5'))
    request=dict(config=cfg,run_id='uniqueFixtureID',spool='/tmp/fake-only',smoke=False,base_url='https://api.wandb.ai')
    commands=[dict(op='log',values=p,step=None) for p in payloads]+[dict(op='finish',exit_code=0)]
    output=[];session(sdk,request,commands,output.append)
    return sdk,output,cfg

def synthetic_rows(n):
    rows=[]
    for kind,multiple in [('R',1),('P',2),('N',10)]:
        for i in range(n*multiple):
            rows.append(dict(identity=f'{kind}-{i}',case_id=i//multiple,kind=kind,
                new_nll=1. if kind!='N' else 3.,true_nll=2.,new_token_count=2,new_token_correct=1,
                new_strict=False,true_token_count=3,true_token_correct=3,true_strict=True))
    return rows

class Tests(unittest.TestCase):
    def test_installed_SDK_metric_and_readback_signature(self):
        import inspect
        try:import wandb
        except ImportError:self.skipTest('isolated SDK venv required')
        inspect.signature(wandb.Run.define_metric).bind(None,'current/pre/*',step_metric='edits',step_sync=False)
        inspect.signature(wandb.apis.public.Run.scan_history).bind(None,keys=['edits'],min_step=0,max_step=1,page_size=2)

    def test_parent_reader_persists_immutable_identity(self):
        import os,threading
        from .client import Tracker
        _,out,cfg=execute([])
        with tempfile.TemporaryDirectory() as d:
            t=Tracker.__new__(Tracker);t.spool=Path(d);t.config_values=cfg;t.run_id='uniqueFixtureID'
            t.job_identity=schema.job_identity(cfg);t.dropped=0;t.startup={};t.result={}
            t.ready=threading.Event();t.done=threading.Event()
            read,write=os.pipe()
            os.write(write,(json.dumps(out[0])+'\n'+json.dumps({'status':'LOGGING_ACCEPTED','delivery':'SDK_ASYNC_NOT_REMOTE_ACK'})+'\n').encode());os.close(write)
            t._read(read)
            identity=json.loads((Path(d)/'identity.json').read_text())
            self.assertEqual(identity['config'],cfg);self.assertEqual(identity['run_id'],'uniqueFixtureID')
            self.assertEqual(t.result['status'],'LOGGING_ACCEPTED');self.assertIn('url',identity)

    def test_queue_full_does_not_advance_axis(self):
        import queue,threading
        from .client import Tracker
        t=Tracker.__new__(Tracker);t.scientific=True;t.axis=AxisState();t.log_lock=threading.Lock()
        t.queue=queue.Queue(1);t.queue.put('full');t.closed=False;t.done=threading.Event();t.dropped=0
        self.assertFalse(t.log({'edits':100}));self.assertIsNone(t.axis.edits)
        t.queue.get();self.assertTrue(t.log({'edits':50}));self.assertEqual(t.axis.edits,50)

    def test_SH4_contract_and_B4_B5_production_mapping(self):
        from project.run_scripts.jlz_price_gptj.tracking import contract_ready,batch_values,w0_subset
        from project.run_scripts.jlz_realization.observe import reduce_rows
        contract_ready()
        current=reduce_rows(synthetic_rows(100));seen=reduce_rows(synthetic_rows(500))
        for b in (4,5):
            v=batch_values(current,current,seen if b==5 else None,b)
            schema.metrics(v,scientific=True)
            for kind,n in [('R',100),('P',200),('N',1000)]:
                self.assertEqual(v[f'current/post/{kind}/count'],n)
                self.assertEqual(v[f'current/pre/{kind}/count'],n)
                self.assertEqual(v[f'current/post/{kind}/token_acc_pct'],100 if kind=='N' else 50)
                self.assertEqual(v[f'current/post/{kind}/strict_acc_pct'],100 if kind=='N' else 0)
                if b==5:self.assertEqual(v[f'all_seen/post/{kind}/count'],5*n)
            self.assertEqual(v['edits'],b*100);self.assertEqual(v['pre_state_edits'],(b-1)*100)
            if b==4:self.assertFalse(any(k.startswith('all_seen/') for k in v))
            sdk,out,cfg=execute([v]);self.assertEqual(out[-1]['method_readback']['status'],'REMOTE_BOUNDED_ROWS_VERIFIED')
        subset=w0_subset(synthetic_rows(500),list(range(100)),'w0/current/N')
        self.assertEqual(subset['w0/current/N/count'],1000)
        self.assertEqual(subset['w0/current/N/token_acc_pct'],100)

    def test_W0_exact_scope(self):
        from project.run_scripts.jlz_interference_l1.comparison_bridge import metric_row
        from project.run_scripts.jlz_realization.observe import reduce_rows
        v=dict(edits=0,**metric_row('W0_first2000',reduce_rows(synthetic_rows(2000)),2000))
        schema.metrics(v,scientific=True)
        with self.assertRaises(ValueError):schema.metrics(dict(v,edits=100),scientific=True)
        v['W0_first2000/R/count']=10000
        with self.assertRaises(ValueError):schema.metrics(v,scientific=True)
        with self.assertRaises(ValueError):schema.metrics({'edits':0,'W0_first1000/R/count':1000})

    def test_harmonic_zero_missing_units(self):
        self.assertAlmostEqual(harmonic([50,60,75]),60)
        self.assertAlmostEqual(harmonic([.5,.6,.75],unit='fraction'),60)
        self.assertEqual(harmonic([0,60,75]),0)
        self.assertIsNone(harmonic([0,None,75]))
        with self.assertRaises(ValueError):schema.metrics({'edits':100,'post_state_edits':100,'current/post/success_harmonic_pct':0})

    def test_axes_and_readback(self):
        v={'edits':100,'post_state_edits':100,'current/post/R/count':100,'current/post/R/success_pct':75.}
        fit={'batch':1,'candidate':2,'fit/global_candidate':2,'fit/loss':1.5}
        sdk,out,_=execute([fit,v])
        for prefix in (*PREFIXES,*NEIGHBOR_PREFIXES):
            self.assertIn(((prefix+'/*',),dict(step_metric='edits',step_sync=False)),sdk.definitions)
        self.assertIn((('fit/*',),dict(step_metric='fit/global_candidate',step_sync=False)),sdk.definitions)
        self.assertEqual(out[-1]['method_readback']['rows'],2)
        self.assertFalse(out[-1]['scientific_completion_claim'])
        accepted=[r for r in out if r['status']=='LOGGING_ACCEPTED']
        self.assertTrue(all(r['delivery']=='SDK_ASYNC_NOT_REMOTE_ACK' for r in accepted))
        self.assertEqual(out[0]['run_name'],'server1-test-r1-job40_0')

    def test_missing_or_wrong_readback_is_not_verified(self):
        v={'edits':100,'post_state_edits':100,'current/post/N/count':1000}
        for mode in ('missing','wrong','unavailable'):
            sdk=MethodSDK()
            def scan(**kw):
                if mode=='unavailable':raise RuntimeError('PRIVATE_SDK_ERROR')
                return iter([] if mode=='missing' else [{'_step':0,**v,'edits':200}])
            sdk.scan_history=scan
            _,out,_=execute([v],sdk)
            self.assertEqual(out[-1]['method_readback']['status'],'UNVERIFIED_INCOMPLETE_OR_UNAVAILABLE')
            self.assertNotIn('PRIVATE_SDK_ERROR',json.dumps(out))

    def test_strict_config_and_privacy(self):
        self.assertEqual(schema.config(CFG_METHOD)['model'],'gptj')
        for changes in ({'metric_schema':'unknown'},{'model':'unregistered'},{'role':'anything'},{'prompt':'secret'},{'source_run_url':'https://wandb.ai/x/runs/id?token=secret'}):
            with self.assertRaises(ValueError):schema.config(dict(CFG_METHOD,**changes))
        for field in ('model','model_family','writer','role','metric_schema','config_sha'):
            c=dict(CFG_METHOD);c.pop(field)
            with self.assertRaises(ValueError):schema.config(c)
        for payload in ({'raw_prompt':'x'},{'edits':100,'current/post/R/token_acc_pct':101},
                        {'edits':100,'current/post/R/count':100.}, {'fit/loss':1.},
                        {'eval/RS':1.},{'current/pre/R/count':100},
                        {'edits':100,'pre_state_edits':101,'post_state_edits':100,'current/pre/R/count':100}):
            with self.assertRaises(ValueError):schema.metrics(payload,scientific=True)

    def test_margin_and_success_arithmetic(self):
        v={'edits':100,'post_state_edits':100,'current/post/N/count':1000,'current/post/N/success_count':500,
           'current/post/N/success_pct':50.,'current/post/N/true_nll':1.,'current/post/N/new_nll':2.,
           'current/post/N/margin_true_minus_new':-1.}
        schema.metrics(v,scientific=True)
        for changes in ({'current/post/N/margin_true_minus_new':1.},{'current/post/N/success_pct':.5}):
            with self.assertRaises(ValueError):schema.metrics(dict(v,**changes),scientific=True)

    def test_monotonic_fit_and_independent_eval(self):
        state=AxisState();state.accept({'fit/global_candidate':25});state.accept({'edits':100})
        state.accept({'fit/global_candidate':26});state.accept({'edits':100})
        with self.assertRaises(ValueError):state.accept({'fit/global_candidate':1})
        with self.assertRaises(ValueError):state.accept({'edits':0})

    def test_immutable_identity_separate_transport(self):
        _,out,cfg=execute([])
        with tempfile.TemporaryDirectory() as d:
            first=create(d,cfg,out[0],'uniqueFixtureID');path=Path(d)/'identity.json';original=path.read_bytes()
            (Path(d)/'receipt.json').write_text('{"status":"LOGGING_ACCEPTED"}')
            with self.assertRaises(FileExistsError):create(d,cfg,out[0],'uniqueFixtureID')
            self.assertEqual(path.read_bytes(),original);self.assertEqual(first['config']['writer'],'memit')
            self.assertFalse(first['scientific_completion_claim'])

if __name__=='__main__':unittest.main()
