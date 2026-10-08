"""CPU/fake SDK only: no auth, network, models, generation or scheduler work."""
import ast
import copy
import inspect
import io
import json
import os
from pathlib import Path
import queue
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from . import schema
from .client import Tracker
from .identity import create
from .method import AxisState, OFFICIAL_SCHEMA, harmonic
from .worker import session, settings


def config(dataset='cf'):
    value=dict(server='server1',task_id='official-baselines',arm='MEMIT_CF',
        attempt='fixture',source_sha='a'*40,config_sha='b'*64,model='llama3',
        model_family='llama',writer='memit',baseline='MEMIT',role='scientific',
        metric_schema=OFFICIAL_SCHEMA,instruction_id=schema.OFFICIAL_INSTRUCTION,
        dataset=dataset)
    if dataset=='cf':
        value.update(generation_metric_schema='counterfact-cake-generation-metrics-v1',
            generation_profile=schema.NATIVE_GENERATION_PROFILE,
            generation_eval_seed=20261007,reference_assets_sha256='c'*64,
            generation_source_sha='d'*40,
            generation_repair_instruction=schema.OFFICIAL_INSTRUCTION,
            generation_schedule=schema.OFFICIAL_GENERATION_SCHEDULE)
    return value


def generation(prefix='W0_first2000'):
    edits=0 if prefix=='W0_first2000' else 2000
    return dict(edits=edits,post_state_edits=edits,**{
        prefix+'/generation/planned_count':2000,
        prefix+'/generation/fluency_count':2000,
        prefix+'/generation/consistency_count':1999,
        prefix+'/generation/generation_prompt_count':20000,
        prefix+'/generation/generated_token_count':123456,
        prefix+'/generation/missing_zero_reference_vector_count':1,
        prefix+'/fluency/ngram_entropy':3.25,
        prefix+'/consistency/reference_score':.125})


def official(prefix='all_seen/post',edits=2000):
    rates=(99.1,90.2,75.3)
    requests=2000 if prefix=='W0_first2000' else edits if prefix=='all_seen/post' else 100
    return dict(edits=edits,pre_state_edits=max(0,edits-100),post_state_edits=edits,**{
        'official/'+prefix+'/requests':requests,
        'official/'+prefix+'/Efficacy':rates[0],
        'official/'+prefix+'/Generalization':rates[1],
        'official/'+prefix+'/Specificity':rates[2],
        'official/'+prefix+'/Score':harmonic(rates),
        'official/'+prefix+'/Score_AlphaEdit_display':harmonic([round(x,2) for x in rates])})


class SDK:
    __version__=schema.SDK_VERSION
    def __init__(self,auth=True,log_error=False):
        self.auth=auth;self.calls=[];self.points=[];self.definitions=[];self.next=0
        self.log_error=log_error;self.Settings=lambda **kw:kw
    def setup(self,**kw):self.settings=kw['settings']
    def login(self,**kw):
        assert kw['prompt'] is False and kw['verify'] is True
        return self.auth
    def init(self,**kw):
        self.calls.append(kw);self.id=kw['id'];self.name=kw['name'];self.config=kw['config']
        self.offline=False
        self.url='https://wandb.ai/wkdguswns2256/layer%20allocation/runs/'+self.id
        return self
    def define_metric(self,*args,**kw):self.definitions.append((args,kw))
    def Api(self,**kw):return self
    def run(self,path):return self
    def log(self,values,step=None):
        if self.log_error:raise RuntimeError('PRIVATE_SDK_ERROR')
        actual=self.next if step is None else step;self.next=actual+1
        self.points.append(dict(values,_step=actual))
    def finish(self,**kw):self.finished=kw
    def scan_history(self,**kw):
        return iter([row for row in self.points if all(k in row for k in kw.get('keys',[]))
            and kw.get('min_step',0)<=row['_step']<kw.get('max_step',10**9)])
    def files(self):return []


def execute(payloads,*,cfg=None,sdk=None):
    sdk=sdk or SDK()
    cfg=schema.bind_job_identity(cfg or config(),{'SLURM_JOB_ID':'42',
        'SLURM_ARRAY_JOB_ID':'40','SLURM_ARRAY_TASK_ID':'0','SLURM_STEP_ID':'-5'})
    request=dict(config=cfg,run_id='officialFixture',spool='/tmp/fake-sdk-no-write',
        smoke=False,base_url='https://api.wandb.ai')
    commands=[dict(op='log',values=p,step=None) for p in payloads]
    commands.append(dict(op='finish',exit_code=0));out=[]
    session(sdk,request,commands,out.append)
    return sdk,out,cfg


class OfficialTransport(unittest.TestCase):
    def test_cf_exact_authority_profile_schedule_and_models(self):
        for model in ('llama3','gptj','qwen25'):
            cfg=config();cfg['model']=model
            self.assertEqual(schema.config(cfg)['generation_schedule'],'W0_AND_W20_FIRST2000')
        for changes in ({'instruction_id':'WRONG'}, {'dataset':'other'},
                        {'model':'gpt2xl'},{'generation_repair_instruction':'USER-DIRECT-NATIVE-FLUCON-REPAIR-20261008-R1'},
                        {'generation_schedule':'W20_ONLY_FIRST2000'},
                        {'generation_profile':'cf-cake-prompt-inclusive-total100-eos-corrected-v1'}):
            with self.assertRaises(ValueError):schema.config(dict(config(),**changes))

    def test_missing_cf_binding_is_not_accepted(self):
        for key in ('instruction_id','dataset','generation_profile','generation_source_sha',
                    'reference_assets_sha256','generation_repair_instruction','generation_schedule'):
            cfg=config();cfg.pop(key)
            with self.assertRaises(ValueError):schema.config(cfg)

    def test_zsre_generation_config_and_metrics_forbidden(self):
        cfg=config('zsre');schema.config(cfg)
        for key in ('generation_schedule','generation_source_sha','reference_assets_sha256'):
            with self.assertRaises(ValueError):schema.config(dict(cfg,**{key:config()[key]}))
        for values in (generation(),{'phase':'W0_generation','generation_progress/step':1}):
            with self.assertRaisesRegex(ValueError,'ZSRE_GENERATION_FORBIDDEN'):
                schema.metrics(values,scientific=True,config_values=cfg)
        value=official();value.pop('official/all_seen/post/Score')
        value.pop('official/all_seen/post/Score_AlphaEdit_display')
        value['official/all_seen/post/Specificity_loc_ans']=87.2
        schema.metrics(value,scientific=True,config_values=cfg)

    def test_official_request_macro_not_prompt_pair_relabeling(self):
        value=official();sdk,out,cfg=execute([value])
        self.assertEqual(sdk.points[0]['official/all_seen/post/requests'],2000)
        self.assertFalse(any(k.startswith('all_seen/post/R/') for k in sdk.points[0]))
        self.assertEqual(out[-1]['method_readback']['status'],'REMOTE_BOUNDED_ROWS_VERIFIED')
        bad=dict(value);bad['official/all_seen/post/Score']=1.
        with self.assertRaisesRegex(ValueError,'OFFICIAL_REQUEST_MACRO_SCORE_MISMATCH'):
            schema.metrics(bad,scientific=True,config_values=cfg)
        with self.assertRaisesRegex(ValueError,'OFFICIAL_METRIC_CONFIG_REQUIRED'):
            schema.metrics(value,scientific=True)

    def test_variable_cf_W0_prompt_denominators_do_not_inherit_PRICE(self):
        value={'edits':0,'W0_first2000/R/count':2000,'W0_first2000/P/count':4013,
               'W0_first2000/N/count':19972}
        schema.metrics(value,scientific=True,config_values=config())
        with self.assertRaisesRegex(ValueError,'W0_EXACT_FIRST2000_COUNTS'):
            schema.metrics(value,scientific=True)

    def test_zsre_actual_W0_token_counts_not_CF_counts(self):
        value={'edits':0,'W0_first2000/R/count':3915,'W0_first2000/P/count':8219,
               'W0_first2000/N/count':6441}
        schema.metrics(value,scientific=True,config_values=config('zsre'))

    def test_CF_generation_W0_W20_only_with_full2000_coverage(self):
        for p in ('W0_first2000','all_seen/post'):
            schema.metrics(generation(p),scientific=True,config_values=config())
        for prefix in ('current/pre','current/post','w0/current','w0/all_seen'):
            with self.assertRaisesRegex(ValueError,'OFFICIAL_GENERATION_ENDPOINT_ONLY'):
                schema.metrics(generation(prefix),scientific=True,config_values=config())
        for changes in ({'edits':500},{'all_seen/post/generation/planned_count':100}):
            with self.assertRaisesRegex(ValueError,'OFFICIAL_GENERATION_EXACT_ENDPOINT'):
                schema.metrics(dict(generation('all_seen/post'),**changes),scientific=True,config_values=config())

    def test_W20_current100_stays_separate_from_generation2000(self):
        value=generation('all_seen/post')
        value.update({'current/post/R/count':100,'current/post/P/count':203,
                      'current/post/N/count':998})
        schema.metrics(value,scientific=True,config_values=config())
        current=official('current/post');current['official/current/post/requests']=2000
        with self.assertRaisesRegex(ValueError,'OFFICIAL_CURRENT_BATCH100'):
            schema.metrics(current,scientific=True,config_values=config())

    def test_official_cumulative_milestones_no_imputed_intermediates(self):
        for edits in (500,1000,1500,2000):
            schema.metrics(official(edits=edits),scientific=True,config_values=config())
        with self.assertRaisesRegex(ValueError,'OFFICIAL_ALL_SEEN_MEASURED_ENDPOINT'):
            schema.metrics(official(edits=600),scientific=True,config_values=config())

    def test_progress_is_not_partial_endpoint(self):
        for phase in ('W0_generation','W20_generation'):
            p={'phase':phase,'generation_progress/step':1,
               'generation_progress/completed_cases':42,'generation_progress/total_cases':2000}
            schema.metrics(p,scientific=True,config_values=config())
            for key in ('edits','fit/global_candidate'):
                with self.assertRaises(ValueError):
                    schema.metrics(dict(p,**{key:1}),scientific=True,config_values=config())
        with self.assertRaisesRegex(ValueError,'OFFICIAL_GENERATION_PHASE'):
            schema.metrics({'phase':'generation_evaluation','generation_progress/step':1},
                scientific=True,config_values=config())

    def test_native_W20_progress_callback_mapping_is_shared_and_nonmutating(self):
        from . import official_generation_progress
        original={'phase':'generation_evaluation','generation_progress/step':7,
                  'generation_progress/completed_cases':42}
        mapped=official_generation_progress(original,endpoint='W20')
        self.assertEqual(original['phase'],'generation_evaluation')
        self.assertEqual(mapped,dict(original,phase='W20_generation'))
        schema.metrics(mapped,scientific=True,config_values=config())
        w0=dict(original,phase='W0_generation')
        self.assertEqual(official_generation_progress(w0,endpoint='W0'),w0)

    def test_native_progress_callback_wrong_endpoint_or_phase_fails(self):
        for endpoint,phase in (('W5','generation_evaluation'),('W0','generation_evaluation'),
                               ('W20','W0_generation')):
            with self.assertRaises(ValueError):
                schema.official_generation_progress({'phase':phase,'generation_progress/step':1},
                                                    endpoint=endpoint)

    def test_fake_SDK_axes_job_zero_index_signed_step_and_readback(self):
        fit={'batch':1,'candidate':1,'fit/global_candidate':1,'fit/loss':2.}
        progress={'phase':'W20_generation','generation_progress/step':2,
                  'generation_progress/completed_cases':2000}
        sdk,out,cfg=execute([generation(),fit,official(),generation('all_seen/post'),progress])
        self.assertEqual(cfg['array_task_id'],'0');self.assertEqual(cfg['step_id'],'-5')
        self.assertTrue(sdk.name.endswith('-job40_0'))
        self.assertIn((('official/all_seen/post/*',),dict(step_metric='edits',step_sync=False)),sdk.definitions)
        self.assertIn((('fit/*',),dict(step_metric='fit/global_candidate',step_sync=False)),sdk.definitions)
        self.assertIn((('generation_progress/*',),dict(step_metric='generation_progress/step',step_sync=False)),sdk.definitions)
        self.assertEqual(out[-1]['method_readback']['rows'],3)
        self.assertEqual(out[-1]['method_readback']['status'],'REMOTE_BOUNDED_ROWS_VERIFIED')
        self.assertFalse(out[-1]['scientific_completion_claim'])
        self.assertTrue(all(x['delivery']=='SDK_ASYNC_NOT_REMOTE_ACK' for x in out if x['status']=='LOGGING_ACCEPTED'))

    def test_no_auth_means_no_run(self):
        sdk,out,_=execute([],sdk=SDK(auth=False))
        self.assertFalse(sdk.calls);self.assertEqual(out[-1]['status'],'SETUP_READY_NEEDS_USER_LOGIN')

    def test_wrong_remote_job_config_is_not_ready(self):
        sdk=SDK();run=sdk.run
        def mismatch(path):
            result=run(path);result.config=dict(result.config,job_id='43');return result
        sdk.run=mismatch
        with self.assertRaisesRegex(ValueError,'REMOTE_JOB_CONFIG_MISMATCH'):execute([],sdk=sdk)

    def test_missing_or_wrong_readback_is_explicit(self):
        for mode in ('empty','changed','error'):
            sdk=SDK()
            def scan(**kw):
                if mode=='error':raise RuntimeError('PRIVATE_SDK_ERROR')
                return iter([] if mode=='empty' else [{'edits':1,'_step':0}])
            sdk.scan_history=scan
            _,out,_=execute([generation()],sdk=sdk)
            self.assertEqual(out[-1]['method_readback']['status'],'UNVERIFIED_INCOMPLETE_OR_UNAVAILABLE')
            self.assertNotIn('PRIVATE_SDK_ERROR',json.dumps(out))

    def test_immutable_identity_cannot_be_overwritten(self):
        _,out,cfg=execute([])
        with tempfile.TemporaryDirectory() as root:
            identity=create(root,cfg,out[0],'officialFixture')
            before=(Path(root)/'identity.json').read_bytes()
            with self.assertRaises(FileExistsError):create(root,cfg,out[0],'officialFixture')
            self.assertEqual(before,(Path(root)/'identity.json').read_bytes())
            self.assertFalse(identity['scientific_completion_claim'])

    def test_scalar_only_never_converts_tensors_or_strings(self):
        class Tensor:
            def __float__(self):raise AssertionError('must not synchronize')
        for value in (Tensor(),[1],{'text':'PRIVATE'},'PRIVATE',float('nan'),float('inf')):
            with self.assertRaises(ValueError):schema.metrics({'fit/loss':value})
        for field in ('raw_prompt','token_ids','activation','api_key','code','full_environment'):
            with self.assertRaises(ValueError):schema.metrics({field:1})
        with self.assertRaises(ValueError):schema.config(dict(config(),api_key='PRIVATE'))

    def test_no_code_console_machine_stats_or_job_artifact_upload(self):
        opts=settings(SDK(),'https://api.wandb.ai')
        for key in ('disable_code','disable_git','disable_job_creation','x_disable_meta',
                    'x_disable_stats','x_disable_machine_info'):
            self.assertTrue(opts[key])
        self.assertFalse(opts['save_code']);self.assertFalse(opts['x_save_requirements'])
        self.assertEqual(opts['console'],'off')
        sdk,_,_=execute([])
        self.assertFalse(sdk.calls[0]['save_code'])

    def test_parent_sidecar_uses_official_module_and_allowlisted_env(self):
        captured={}
        def capture(args,**kwargs):
            captured.update(args=args,kwargs=kwargs)
            raise OSError('fixture stops before any real process or auth')
        local={'WANDB_ENTITY':schema.ENTITY,'WANDB_PROJECT':schema.PROJECT,
               'WANDB_BASE_URL':'https://api.wandb.ai','ODEEDIT_WANDB_PYTHON':'/bin/false'}
        with tempfile.TemporaryDirectory() as root,patch.dict(os.environ,{'PRIVATE_FULL_ENV':'PRIVATE'}),\
             patch('official.tracking.client.load_env',return_value=local),\
             patch('official.tracking.client.subprocess.Popen',side_effect=capture):
            with self.assertRaises(OSError):Tracker(env_file='ignored',spool=Path(root)/'new',config_values=config())
        self.assertEqual(captured['args'][3],'official.tracking.worker')
        self.assertNotIn('PRIVATE_FULL_ENV',captured['kwargs']['env'])
        self.assertEqual(captured['kwargs']['env']['CUDA_VISIBLE_DEVICES'],'')
        self.assertEqual(Path(captured['kwargs']['env']['PYTHONPATH']),Path(__file__).resolve().parents[2])
        self.assertNotIn('PRIVATE',repr(captured))

    def test_module_has_no_external_task_imports(self):
        for path in Path(__file__).parent.glob('*.py'):
            tree=ast.parse(path.read_text())
            for node in ast.walk(tree):
                names=[x.name for x in node.names] if isinstance(node,ast.Import) else \
                    [node.module] if isinstance(node,ast.ImportFrom) and node.level==0 else []
                self.assertFalse(any(x and x.startswith(('project.','scripts.','easyeditor.')) for x in names),path.name)

    def test_axes_monotonic_and_failed_queue_does_not_advance(self):
        state=AxisState();state.accept({'fit/global_candidate':25});state.accept({'edits':100})
        state.accept({'fit/global_candidate':26});state.accept({'edits':100})
        with self.assertRaises(ValueError):state.accept({'edits':0})
        with self.assertRaises(ValueError):state.accept({'fit/global_candidate':1})
        tracker=Tracker.__new__(Tracker);tracker.scientific=True;tracker.config_values=config()
        tracker.axis=AxisState();tracker.log_lock=threading.Lock();tracker.queue=queue.Queue(1)
        tracker.queue.put('full');tracker.closed=False;tracker.done=threading.Event();tracker.dropped=0
        self.assertFalse(tracker.log({'edits':100}));self.assertIsNone(tracker.axis.edits)

    def test_finish_timeout_does_not_mask_science_exception(self):
        tracker=Tracker.__new__(Tracker);tracker.closed=False;tracker.queue=queue.Queue(1)
        tracker.done=threading.Event();tracker.result={};tracker.dropped=0
        tracker._stop=Mock();tracker._receipt=Mock()
        self.assertEqual(tracker.finish(timeout=.001)['local_status'],'LOGGING_DEGRADED_FINISH_TIMEOUT')
        tracker._stop.assert_called_once();tracker.finish=Mock(side_effect=OSError('PRIVATE'))
        self.assertFalse(tracker.__exit__(RuntimeError,RuntimeError('science'),None))

    def test_legacy_PRICE_and_native_W20_optional_compatibility(self):
        cfg=config();cfg['metric_schema']=schema.COMPARISON_SCHEMA
        cfg.pop('dataset');cfg.pop('instruction_id')
        cfg.update(generation_repair_instruction='USER-DIRECT-NATIVE-FLUCON-REPAIR-20261008-R1',
                   generation_schedule='W20_ONLY_FIRST2000')
        schema.config(cfg)
        with self.assertRaises(ValueError):schema.config(dict(cfg,generation_schedule=schema.OFFICIAL_GENERATION_SCHEDULE))
        exact={'edits':0,'W0_first2000/R/count':2000,'W0_first2000/P/count':4000,'W0_first2000/N/count':20000}
        schema.metrics(exact,scientific=True,config_values=cfg)
        with self.assertRaisesRegex(ValueError,'W0_EXACT_FIRST2000_COUNTS'):
            schema.metrics(dict(exact,**{'W0_first2000/P/count':4001}),scientific=True,config_values=cfg)


if __name__=='__main__':unittest.main()
