"""Nonblocking scalar-only API; SDK stays out of the scientific Python env."""
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import uuid
from .schema import bind_job_identity, job_identity, metrics, load_env


class LoggingBlocked(RuntimeError):
    pass


class Tracker:
    def __init__(self, *, env_file, spool, config_values, smoke=False, startup_timeout=50):
        cfg=bind_job_identity(config_values); settings=load_env(env_file)
        self.job_identity=job_identity(cfg)
        self.spool=Path(spool).resolve();self.spool.mkdir(parents=True,exist_ok=False,mode=0o700)
        self.run_id=uuid.uuid4().hex[:16];self.status='STARTING';self.result={};self.startup={}
        self.ready=threading.Event();self.done=threading.Event();self.queue=queue.Queue(maxsize=1024)
        self.dropped=0;self.closed=False
        # Do not forward the full experiment environment or secret-rich argv to SDK.
        keys=('PATH','HOME','USER','LANG','LC_ALL','SSL_CERT_FILE','REQUESTS_CA_BUNDLE',
              'NETRC','WANDB_API_KEY','WANDB_IDENTITY_TOKEN_FILE','WANDB_CREDENTIALS_FILE','WANDB_CONFIG_DIR')
        env={k:os.environ[k] for k in keys if k in os.environ}
        env.update(WANDB_ENTITY=settings['WANDB_ENTITY'],WANDB_PROJECT=settings['WANDB_PROJECT'],
            WANDB_MODE='online',WANDB_CONSOLE='off',WANDB_SAVE_CODE='false',WANDB_DISABLE_CODE='true',
            WANDB_DISABLE_GIT='true',WANDB_BASE_URL=settings['WANDB_BASE_URL'],CUDA_VISIBLE_DEVICES='',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
            GOMAXPROCS='2',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(Path(__file__).resolve().parents[3]))
        read_fd,write_fd=os.pipe()
        self.proc=subprocess.Popen([settings['ODEEDIT_WANDB_PYTHON'],'-u','-m',
            'project.run_scripts.experiment_tracking.worker',str(write_fd)],stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,env=env,cwd=self.spool,
            pass_fds=(write_fd,),start_new_session=True,text=True,bufsize=1)
        os.close(write_fd)
        self.reader=threading.Thread(target=self._read,args=(read_fd,),daemon=True);self.reader.start()
        request=dict(config=cfg,run_id=self.run_id,base_url=settings['WANDB_BASE_URL'],spool=str(self.spool),smoke=smoke)
        self.proc.stdin.write(json.dumps(request)+'\n');self.proc.stdin.flush()
        if not self.ready.wait(startup_timeout) or self.status!='READY_ONLINE':
            self._stop();self._receipt()
            raise LoggingBlocked(self.status if self.status!='STARTING' else 'LOGGING_BLOCKED_STARTUP_TIMEOUT')
        self.sender=threading.Thread(target=self._send,daemon=True);self.sender.start()

    def _receipt(self):
        try:
            data=dict(run_id=self.run_id,status=self.status,dropped_points=self.dropped,result=self.result,
                      exact_model_resume='NOT_IMPLIED',credential_saved=False,job_identity=self.job_identity,
                      startup_readback=self.startup)
            tmp=self.spool/'receipt.tmp'
            tmp.write_text(json.dumps(data,allow_nan=False)+'\n');os.replace(tmp,self.spool/'receipt.json')
        except Exception:
            pass # Telemetry/IO failure must not replace a scientific exception.

    def _read(self,fd):
        try:
            with os.fdopen(fd) as stream:
                for line in stream:
                    result=json.loads(line);self.result=result
                    self.status=result['status']
                    if self.status=='READY_ONLINE':
                        self.startup=dict(result);self.ready.set()
                    elif self.status not in ('LOGGING_ACCEPTED','LOGGING_DEGRADED'):
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
            values=metrics(values)
            if step is not None and (type(step) is not int or step<0):raise ValueError('INVALID_STEP')
            self.queue.put_nowait(dict(op='log',values=values,step=step));return True
        except Exception:
            self.dropped+=1;self.status='LOGGING_DEGRADED_REJECTED_POINT';return False

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
        return dict(self.result,local_status=self.status,dropped_points=self.dropped)

    def __enter__(self):return self
    def __exit__(self,kind,value,traceback):
        try:self.finish(exit_code=1 if kind else 0)
        except Exception:pass
        return False


def init(*,env_file,spool,config,**kwargs):
    return Tracker(env_file=env_file,spool=spool,config_values=config,**kwargs)
