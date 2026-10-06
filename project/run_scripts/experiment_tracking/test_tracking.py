import json
import inspect
from pathlib import Path
import queue
import tempfile
import threading
import unittest
from unittest.mock import Mock
from .schema import config,metrics,load_env,endpoint,SDK_VERSION,bind_job_identity
from .worker import session,settings
from .client import Tracker

CFG=dict(server='server1',task_id='fixture',arm='test',attempt='r1',source_sha='a'*40)


class SDK:
    __version__=SDK_VERSION
    def __init__(self,login=True,log_error=False,finish_error=False):
        self.auth=login;self.calls=[];self.points=[];self.log_error=log_error;self.finish_error=finish_error
        self.Settings=lambda **kw:kw
        self.files_list=[]
    def setup(self,**kw):self.options=kw['settings']
    def login(self,**kw):
        assert kw['prompt'] is False and kw['verify'] is True
        return self.auth
    def init(self,**kw):
        self.calls.append(kw);self.id=kw['id'];self.offline=False
        self.name=kw['name'];self.config=kw['config']
        self.url='https://wandb.ai/wkdguswns2256/layer%20allocation/runs/'+self.id
        return self
    def Api(self,**kw):return self
    def run(self,path):return self
    def log(self,values,step=None):
        if self.log_error:raise RuntimeError('SECRET_SENTINEL_MUST_NOT_ESCAPE')
        self.points.append(values)
    def finish(self,**kw):
        if self.finish_error:raise RuntimeError('SECRET_SENTINEL_MUST_NOT_ESCAPE')
        self.finished=kw
    def scan_history(self,**kw):return iter(self.points)
    def files(self):return [type('File',(),dict(name=n)) for n in self.files_list]


def exercise(sdk,smoke=True):
    request=dict(config=bind_job_identity(CFG,{}),run_id='fixture123',spool='/tmp/no-write-fake',smoke=smoke,base_url='https://api.wandb.ai')
    commands=[dict(op='log',values=dict(setup_ok=1,step=i),step=i) for i in range(3)]
    commands.append(dict(op='finish',exit_code=0));out=[]
    session(sdk,request,commands,out.append)
    return out


class Tests(unittest.TestCase):
    def test_installed_sdk_surface(self):
        try:import wandb
        except ImportError:self.skipTest('isolated SDK venv required for actual signature check')
        self.assertEqual(wandb.__version__,SDK_VERSION)
        settings(wandb,'https://api.wandb.ai')
        class SignatureSDK(SDK):
            def finish(self,**kw):
                inspect.signature(wandb.Run.finish).bind(None,**kw)
                return super().finish(**kw)
        result=exercise(SignatureSDK())
        self.assertEqual(result[-1]['status'],'READY_ONLINE_VERIFIED')
    def test_no_auth_no_run(self):
        sdk=SDK(login=False);out=exercise(sdk)
        self.assertEqual(out[-1]['status'],'SETUP_READY_NEEDS_USER_LOGIN');self.assertFalse(sdk.calls)
    def test_three_points_remote(self):
        sdk=SDK();out=exercise(sdk)
        self.assertEqual(out[-1]['status'],'READY_ONLINE_VERIFIED');self.assertEqual(out[-1]['remote_points'],3)
        self.assertEqual(len(sdk.points),3);self.assertEqual(len(sdk.calls),1)
        self.assertEqual(sdk.calls[0]['project'],'layer allocation');self.assertEqual(sdk.calls[0]['mode'],'online')
    def test_privacy_flags(self):
        v=settings(SDK(),'https://api.wandb.ai')
        self.assertEqual(v['console'],'off');self.assertFalse(v['save_code']);self.assertFalse(v['x_save_requirements'])
        for k in ('disable_code','disable_git','disable_job_creation','x_disable_meta','x_disable_stats','x_disable_machine_info'):
            self.assertTrue(v[k])
    def test_forbidden_remote_file(self):
        sdk=SDK();sdk.files_list=['code/main.py']
        with self.assertRaisesRegex(ValueError,'REMOTE_UPLOAD_BOUNDARY'):exercise(sdk)
    def test_logging_failure_not_science_failure(self):
        out=exercise(SDK(log_error=True),smoke=False)
        self.assertEqual(out[-1]['status'],'FINISHED_UNVERIFIED')
        self.assertNotIn('SECRET_SENTINEL',json.dumps(out))
    def test_finish_failure_isolated(self):
        out=exercise(SDK(finish_error=True),smoke=False)
        self.assertEqual(out[-1]['status'],'LOGGING_DEGRADED_FINISH')
        self.assertNotIn('SECRET_SENTINEL',json.dumps(out))
    def test_missing_readback_not_pass(self):
        sdk=SDK();sdk.scan_history=lambda **kw:iter([])
        with self.assertRaisesRegex(ValueError,'READBACK_NOT_THREE_POINTS'):exercise(sdk)
    def test_offline_not_pass(self):
        sdk=SDK();original=sdk.init
        def init(**kw):r=original(**kw);r.offline=True;return r
        sdk.init=init
        with self.assertRaisesRegex(ValueError,'NOT_ONLINE'):exercise(sdk)
    def test_scalar_only_no_conversion(self):
        class Tensor:
            def __float__(self):raise AssertionError('must not synchronize')
        for value in (Tensor(),[1],{'a':1},'prompt',float('nan'),float('inf')):
            with self.assertRaises(ValueError):metrics({'fit/loss':value})
    def test_secret_and_unknown_metrics(self):
        for key in ('api_key','prompt','activation','fit/api_key','weight','arbitrary'):
            with self.assertRaises(ValueError):metrics({key:1})
        self.assertEqual(metrics({'fit/loss':1.5}),{'fit/loss':1.5})
    def test_config_only_identity(self):
        with self.assertRaises(ValueError):config(dict(CFG,api_key='SECRET_SENTINEL'))
        with self.assertRaises(ValueError):config(dict(CFG,source_sha='not-a-sha'))
        with self.assertRaises(ValueError):config(dict(CFG,arm='secret prose cannot pass'))
    def test_config_not_shell(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'wandb.env'
            base="WANDB_ENTITY=wkdguswns2256\nWANDB_PROJECT='layer allocation'\nWANDB_MODE=online\nWANDB_CONSOLE=off\nWANDB_SAVE_CODE=false\nODEEDIT_WANDB_PYTHON=/bin/true\n"
            p.write_text(base);self.assertEqual(load_env(p)['WANDB_PROJECT'],'layer allocation')
            for addition in ('WANDB_API_KEY=SECRET_SENTINEL\n','WANDB_BASE_URL=$(touch /tmp/forbidden)\n'):
                p.write_text(base+addition)
                with self.assertRaises(ValueError):load_env(p)
    def test_ui_not_endpoint(self):
        with self.assertRaises(ValueError):endpoint('https://forge.coreweave.com/wandb/wkdguswns2256/layer%20allocation')
        with self.assertRaises(ValueError):endpoint('https://user:secret@api.wandb.ai')
    def test_runtime_reject_is_nonthrowing(self):
        t=Tracker.__new__(Tracker);t.closed=False;t.done=threading.Event();t.queue=queue.Queue(1);t.dropped=0
        self.assertFalse(t.log({'prompt':'not allowed'}));self.assertTrue(t.log({'step':0}))
        self.assertFalse(t.log({'step':1}));self.assertEqual(t.dropped,2)
    def test_finish_bound_and_exception_preserved(self):
        t=Tracker.__new__(Tracker);t.closed=False;t.queue=queue.Queue(1);t.done=threading.Event()
        t.result={};t.dropped=0;t._stop=Mock();t._receipt=Mock()
        result=t.finish(timeout=.001)
        self.assertEqual(result['local_status'],'LOGGING_DEGRADED_FINISH_TIMEOUT');t._stop.assert_called_once()
        t=Tracker.__new__(Tracker);t.finish=Mock(return_value={})
        self.assertFalse(t.__exit__(RuntimeError,RuntimeError('science'),None))
    def test_cleanup_error_cannot_mask_science(self):
        t=Tracker.__new__(Tracker);t.closed=False;t.queue=queue.Queue(1);t.done=threading.Event();t.done.set()
        t.result={};t.dropped=0;t._stop=Mock(side_effect=OSError('private path'));t._receipt=Mock()
        self.assertEqual(t.finish()['local_status'],'LOGGING_DEGRADED_STOP')
        t.finish=Mock(side_effect=OSError('private path'))
        self.assertFalse(t.__exit__(RuntimeError,RuntimeError('original scientific exception'),None))


if __name__=='__main__':unittest.main()
