"""Task-local scalar transport derived from shared client SHA5a60aa234694.

No shared edits/monkeypatch. New worker/profile, explicit rejection evidence,
immutable reference-only identity, original safe credential isolation retained.
"""
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import tempfile
import threading
import uuid
from .schema import config, bind_job_identity, job_identity, metrics, load_env, require, AxisState


class LoggingBlocked(RuntimeError):
    pass


def chosen_run_id(value=None):
    """New create-once task UUID, optionally sealed before Slurm submission."""
    result=uuid.uuid4().hex[:16] if value is None else value
    require(type(result) is str and re.fullmatch(r'[a-f0-9]{16}',result), 'W0_NEW_UUID_REQUIRED')
    return result


def create_identity(spool,cfg,startup,run_id):
    chosen_run_id(run_id)
    cfg = config(cfg)
    require(startup['run_id']==run_id and startup['config']==cfg
            and startup['job_identity']==job_identity(cfg), 'W0_STARTUP_IDENTITY')
    value = dict(run_id=run_id,url=startup['url'],run_name=startup['run_name'],config=cfg,
        job_identity=job_identity(cfg),startup_remote_identity_verified=True,
        scientific_completion_claim=False,reference_only=True,evaluation_model_state='W0',
        edits_axis_semantics='reference_cohort_progress',transport_receipt='receipt.json')
    root = Path(spool)
    fd,temp = tempfile.mkstemp(prefix='.identity-',dir=root)
    try:
        with os.fdopen(fd,'w') as stream:
            json.dump(value,stream,sort_keys=True,allow_nan=False)
            stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.link(temp,root/'identity.json')
    finally:
        os.unlink(temp)
    return value


class Tracker:
    def __init__(self,*,env_file,spool,config_values,startup_timeout=50,run_id=None):
        cfg = bind_job_identity(config_values)
        settings = load_env(env_file)
        self.config_values=cfg;self.axis=AxisState();self.log_lock=threading.Lock()
        self.job_identity=job_identity(cfg)
        self.spool=Path(spool).resolve();self.spool.mkdir(parents=True,exist_ok=False,mode=0o700)
        self.run_id=chosen_run_id(run_id);self.status='STARTING';self.result={};self.startup={}
        self.ready=threading.Event();self.done=threading.Event();self.queue=queue.Queue(maxsize=1024)
        self.dropped=0;self.closed=False;self.last_rejection=None
        keys=('PATH','HOME','USER','LANG','LC_ALL','SSL_CERT_FILE','REQUESTS_CA_BUNDLE',
              'NETRC','WANDB_API_KEY','WANDB_IDENTITY_TOKEN_FILE','WANDB_CREDENTIALS_FILE','WANDB_CONFIG_DIR')
        env={key:os.environ[key] for key in keys if key in os.environ}
        env.update(WANDB_ENTITY=settings['WANDB_ENTITY'],WANDB_PROJECT=settings['WANDB_PROJECT'],
            WANDB_MODE='online',WANDB_CONSOLE='off',WANDB_SAVE_CODE='false',WANDB_DISABLE_CODE='true',
            WANDB_DISABLE_GIT='true',WANDB_BASE_URL=settings['WANDB_BASE_URL'],CUDA_VISIBLE_DEVICES='',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
            GOMAXPROCS='2',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(Path(__file__).resolve().parents[4]))
        read_fd,write_fd=os.pipe()
        self.proc=subprocess.Popen([settings['ODEEDIT_WANDB_PYTHON'],'-u','-m',
            'project.run_scripts.base_model_eval.gpt2xl_server1_cohort.worker',str(write_fd)],
            stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,env=env,
            cwd=self.spool,pass_fds=(write_fd,),start_new_session=True,text=True,bufsize=1)
        os.close(write_fd)
        self.reader=threading.Thread(target=self._read,args=(read_fd,),daemon=True);self.reader.start()
        self.proc.stdin.write(json.dumps(dict(config=cfg,run_id=self.run_id,base_url=settings['WANDB_BASE_URL'],
            spool=str(self.spool)))+'\n');self.proc.stdin.flush()
        if not self.ready.wait(startup_timeout) or self.status!='READY_ONLINE':
            self._stop();self._receipt()
            raise LoggingBlocked(self.status if self.status!='STARTING' else 'LOGGING_BLOCKED_STARTUP_TIMEOUT')
        self.sender=threading.Thread(target=self._send,daemon=True);self.sender.start()

    def _receipt(self):
        try:
            value=dict(run_id=self.run_id,status=self.status,rejected_points=self.dropped,
                dropped_points=self.dropped,last_rejection=self.last_rejection,result=self.result,
                exact_model_resume='NOT_IMPLIED',credential_saved=False,job_identity=self.job_identity,
                startup_readback=self.startup,reference_only=True,evaluation_model_state='W0')
            temp=self.spool/'receipt.tmp'
            temp.write_text(json.dumps(value,allow_nan=False)+'\n');os.replace(temp,self.spool/'receipt.json')
        except Exception:
            pass

    def _read(self,fd):
        try:
            with os.fdopen(fd) as stream:
                for line in stream:
                    result=json.loads(line);self.result=result;self.status=result['status']
                    if self.status=='READY_ONLINE':
                        self.identity=create_identity(self.spool,self.config_values,result,self.run_id)
                        self.startup=dict(result);self.ready.set()
                    elif self.status not in ('LOGGING_ACCEPTED','LOGGING_DEGRADED','LOGGING_REJECTED'):
                        self.ready.set();self.done.set()
                    self._receipt()
        except Exception:
            self.status='LOGGING_DEGRADED_CONTROL'
        finally:
            self.ready.set();self.done.set()

    def _send(self):
        try:
            with (self.spool/'accepted-scalars.jsonl').open('x') as journal:
                while True:
                    message=self.queue.get()
                    journal.write(json.dumps(message,allow_nan=False)+'\n');journal.flush()
                    self.proc.stdin.write(json.dumps(message,allow_nan=False)+'\n');self.proc.stdin.flush()
                    if message['op']=='finish':return
        except Exception:
            self.status='LOGGING_DEGRADED_TRANSPORT';self.done.set();self._receipt()

    def log(self,values,*,step=None):
        try:
            if self.closed or self.done.is_set():raise ValueError('LOGGER_CLOSED')
            value=metrics(values,self.config_values)
            require(step is None or type(step) is int and step>=0,'INVALID_STEP')
            with self.log_lock:
                self.axis.check(value)
                self.queue.put_nowait(dict(op='log',values=value,step=step))
                self.axis.accept(value)
            return True
        except Exception as error:
            self.dropped+=1;self.status='LOGGING_DEGRADED_REJECTED_POINT'
            self.last_rejection=dict(code='SCHEMA_OR_QUEUE_REJECTED',error_type=type(error).__name__,
                point_not_enqueued=True,curve_axis_advanced=False)
            self._receipt()
            return False

    def _stop(self):
        if self.proc.poll() is None:
            try:os.killpg(self.proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:os.killpg(self.proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                self.proc.wait(timeout=2)

    def finish(self,*,exit_code=0,timeout=45):
        if self.closed:return self.result
        self.closed=True
        try:
            self.queue.put(dict(op='finish',exit_code=0 if exit_code==0 else 1),timeout=1)
            if not self.done.wait(timeout):self.status='LOGGING_DEGRADED_FINISH_TIMEOUT'
        except Exception:
            self.status='LOGGING_DEGRADED_FINISH'
        finally:
            try:self._stop()
            except Exception:self.status='LOGGING_DEGRADED_STOP'
            self._receipt()
        return dict(self.result,local_status=self.status,rejected_points=self.dropped,
            dropped_points=self.dropped,last_rejection=self.last_rejection)

    def __enter__(self):return self
    def __exit__(self,kind,value,traceback):
        try:self.finish(exit_code=1 if kind else 0)
        except Exception:pass
        return False


def init(*,env_file,spool,config,**kwargs):
    return Tracker(env_file=env_file,spool=spool,config_values=config,**kwargs)
