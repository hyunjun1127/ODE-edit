"""USER-authorized initial-W0 telemetry through the existing sole SDK writer.

No model imports/calls, SDK init/sync/finish, signals, job/source/config edits or
agent polling. Immutable observation-file events supply progress, not scores.
The bridge exits as soon as the original producer starts its scalar journal.
"""
import argparse
import ctypes
import fcntl
import json
import os
from pathlib import Path
import re
import resource
import select
import stat
import struct
import subprocess
import sys
import time

from .common import LOCAL,TASK,read,write,member,sha,require
from project.run_scripts.experiment_tracking.schema import metrics,load_env
from project.run_scripts.jlz_price_gpt2xl.tracking import log_w0

ATTEMPT=LOCAL/'attempt-register-r1'
SCIENCE='6bc51602632b5a2dfb4c832479002b30b604b8eb'
CONTROL=LOCAL/'live-progress-r1'
MASK=0x00000100|0x00000002|0x00000080  # IN_CREATE,IN_MODIFY,IN_MOVED_TO


def frame(values):
    payload=metrics(values,scientific=True)
    data=(json.dumps(dict(op='log',values=payload,step=None),separators=(',',':'),allow_nan=False)+'\n').encode()
    require(len(data)<=4096,'ATOMIC_PROTOCOL_FRAME_BOUND')
    return data


def w0_values(out):
    value=read(out/'W0/summary.json');runtime=read(out/'runtime.json')
    require(value['requests']==2000 and value['row_count']==26000 and value['no_mutation'] is True
        and value['optimizer_feedback'] is False and value['state']==runtime['cold_state'],'EXISTING_W0_IDENTITY')
    class Capture:
        def log(self,payload,step=None):self.values=metrics(payload,scientific=True);return True
    target=Capture();require(log_w0(target,value['summary']),'PRODUCTION_W0_MAPPING')
    frame(target.values)
    return target.values


def first_parent_frames_atomic(values):
    """Only first W0 and generation-W0 frames can race before journal handoff.

    Both have small closed key sets; a generous maximal generation payload also
    fits PIPE_BUF. No bridge writes are allowed after the parent journal starts.
    """
    frame(values)
    prefix='W0_first2000'
    generation={prefix+'/generation/'+k:2000 for k in (
        'planned_count','fluency_count','consistency_count','generation_prompt_count','generated_token_count',
        'missing_missing_generation_prompts_count','missing_missing_reference_count','missing_zero_generated_vector_count',
        'missing_zero_reference_vector_count','missing_nonfinite_score_count','missing_length_cap_no_continuation_count')}
    generation.update({prefix+'/fluency/ngram_entropy':10.,prefix+'/consistency/reference_score':.5,
        'edits':0,'pre_state_edits':0,'post_state_edits':0})
    frame(generation)


def proc(pid):
    root=Path('/proc')/str(pid);require(root.stat().st_uid==os.getuid(),'EXACT_PROCESS_OWNER')
    tail=(root/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,parent=int(tail[1]),start=int(tail[19]),
        argv=(root/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0'),cwd=Path(os.readlink(root/'cwd')),
        rss=int(tail[21])*os.sysconf('SC_PAGE_SIZE'))


def locate(job,out,c):
    result=subprocess.run(['scontrol','listpids',job],capture_output=True,text=True,timeout=10)
    require(result.returncode==0,'OWN_JOB_PID_BINDING')
    ids={int(x.split()[0]) for x in result.stdout.splitlines()[1:] if x.split() and x.split()[0].isdigit()
        and len(x.split())>1 and x.split()[1]==job}
    sdk=load_env(c['tracking']['env_file'])['ODEEDIT_WANDB_PYTHON'];found=[]
    for pid in ids:
        try:p=proc(pid)
        except (OSError,ValueError):continue
        if p['cwd']!=out/'tracking':continue
        if len(p['argv'])!=5 or p['argv'][:4]!=[sdk,'-u','-m','project.run_scripts.experiment_tracking.worker']:continue
        parent=proc(p['parent'])
        require(parent['pid'] in ids and parent['cwd']==ATTEMPT/'source'
            and parent['argv'][-4:]==['--attempt',str(ATTEMPT),'--arm','BASE_MEMIT']
            and 'project.run_scripts.gpt2xl_generation_baselines.run' in parent['argv'],'OWN_SCIENTIFIC_PARENT')
        pipe=os.readlink(Path('/proc')/str(pid)/'fd/0')
        require(pipe.startswith('pipe:['),'EXISTING_SDK_STDIN_PIPE')
        require(any(os.readlink(p)==pipe for p in (Path('/proc')/str(parent['pid'])/'fd').iterdir()),
                'PARENT_WRITES_SAME_SDK_PIPE')
        found.append((p,parent,pipe))
    require(len(found)==1,'SINGLE_EXISTING_SDK_WRITER')
    return found[0]


def parent_started(out):
    p=out/'tracking/accepted-scalars.jsonl'
    return p.stat().st_size>0


def progress_values(count,parent):
    # step is completed immutable initial-W0 case files, not edits or fit.
    require(type(count) is int and 0<=count<=2000,'INITIAL_W0_PROGRESS_COUNT')
    elapsed=max(0.,time.clock_gettime(time.CLOCK_BOOTTIME)-parent['start']/os.sysconf('SC_CLK_TCK'))
    return dict(step=count,phase_id=2,status_code=1,**{
        'time/elapsed_seconds':elapsed,'memory/host_rss_bytes':parent['rss']})


def active(worker,parent,out,identity,pipe=None):
    w,p=proc(worker['pid']),proc(parent['pid'])
    require(w['start']==worker['start'] and p['start']==parent['start']
        and w['parent']==p['pid'] and w['cwd']==out/'tracking'
        and w['argv']==worker['argv'] and p['argv']==parent['argv'],'PID_REUSE_OR_OWNER_CHANGE')
    if pipe is not None:
        require(os.readlink(Path('/proc')/str(worker['pid'])/'fd/0')==pipe,'PIPE_INODE_CHANGED')
    receipt=read(out/'tracking/receipt.json')
    require(receipt['run_id']==identity['run_id'] and receipt['status'] in ('READY_ONLINE','LOGGING_ACCEPTED'),
            'EXISTING_LOGGER_NOT_HEALTHY')
    require(not (out/'terminal.json').exists(),'SCIENCE_TERMINAL_NO_LIVE_ATTACH')
    return p


def bridge(job):
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1])
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    require(job=='60928','THIS_EXACT_ALREADY_RUNNING_JOB_ONLY')
    out=ATTEMPT/'BASE_MEMIT';c=read(ATTEMPT/'config.json');lock=read(ATTEMPT/'execution.lock.json')
    identity=read(out/'tracking-identity.json')
    require(lock['source_commit']==SCIENCE and sha(ATTEMPT/'config.json')==lock['config_sha256']
        and identity['source_sha']==SCIENCE and identity['config_sha']==lock['config_sha256']
        and identity['config']['job_id']==job and identity['run_id']=='83b85657a45c4673','EXACT_EXISTING_RUN_SOURCE')
    require(not parent_started(out),'NATIVE_LOGGER_ALREADY_ACTIVE_NO_ATTACH')
    values=w0_values(out);first_parent_frames_atomic(values)
    worker,parent,pipe=locate(job,out,c);active(worker,parent,out,identity)
    CONTROL.mkdir(exist_ok=True,mode=0o700)
    singleton=(CONTROL/'bridge.lock').open('a');fcntl.flock(singleton,fcntl.LOCK_EX|fcntl.LOCK_NB)
    require(not (CONTROL/'attachment.json').exists(),'ONE_USER_AUTHORIZED_ATTACH_NO_RETRY')
    fd=os.open(Path('/proc')/str(worker['pid'])/'fd/0',os.O_WRONLY|os.O_NONBLOCK)
    require(stat.S_ISFIFO(os.fstat(fd).st_mode) and os.readlink(Path('/proc')/str(worker['pid'])/'fd/0')==pipe,
            'EXACT_EXISTING_PIPE_OPENED')
    limit=os.fpathconf(fd,'PC_PIPE_BUF');require(limit>=4096,'ATOMIC_PIPE_LIMIT')
    proof=dict(user_exact='실시간 기록 진행해',task_id=TASK,job_id=job,run_id=identity['run_id'],
        scientific_source=SCIENCE,scientific_config_sha256=lock['config_sha256'],source=member(Path(__file__)),
        SDK_writer_pid=worker['pid'],SDK_writer_start=worker['start'],scientific_pid=parent['pid'],
        original_SDK_and_run_reused=True,new_SDK_init_or_sync=0,new_fit_or_forward=0,science_job_source_mutations=0,
        telemetry_semantics=dict(phase_id_2='INITIAL_W0_GENERATION',status_code_1='LIVE_NONTERMINAL',
            step='completed atomic generation observation case files; not edits/fit/full-endpoint score'),
        cadence='new immutable case-file events, coalesced max one progress point per30s; no heartbeat or agent polling',
        end='original producer first scalar journal write / existing worker or science terminal / original48h wall bound')
    write(CONTROL/'attachment.json',proof)
    eventlog=(CONTROL/'scalar-requests.jsonl').open('x',buffering=1)
    def send(payload):
        active(worker,parent,out,identity,pipe)
        require(not parent_started(out),'HANDOFF_ORIGINAL_PRODUCER_ACTIVE')
        data=frame(payload);require(len(data)<=limit,'ATOMIC_FRAME')
        eventlog.write(json.dumps(dict(values=payload,step=None,delivery='REQUEST_NOT_REMOTE_ACK',time=time.time()),allow_nan=False)+'\n')
        eventlog.flush();os.fsync(eventlog.fileno())
        require(os.write(fd,data)==len(data),'ATOMIC_WRITE_COMPLETE')
    libc=ctypes.CDLL(None,use_errno=True)
    watch=libc.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC);require(watch>=0,'LOCAL_EVENT_WATCH')
    observations=out/'generation/observations'
    wd=libc.inotify_add_watch(watch,os.fsencode(observations),MASK)
    journal_wd=libc.inotify_add_watch(watch,os.fsencode(out/'tracking/accepted-scalars.jsonl'),MASK)
    require(wd>=0 and journal_wd>=0,'EXACT_LOCAL_WATCH_PATHS')
    names={p.name for p in observations.glob('*.json') if re.fullmatch(r'[0-9a-f]{64}\.json',p.name)
        and p.is_file() and not p.is_symlink()}
    count=len(names);status='UNKNOWN';sent=0
    elapsed=progress_values(count,parent)['time/elapsed_seconds']
    deadline=time.monotonic()+max(0,48*3600-elapsed)
    try:
        send(values);sent+=1
        send(progress_values(count,active(worker,parent,out,identity,pipe)));sent+=1
        last=time.monotonic();last_count=count
        while time.monotonic()<deadline:
            if parent_started(out):status='HANDOFF_ORIGINAL_PRODUCER_ACTIVE';break
            active(worker,parent,out,identity,pipe)
            timeout=max(.01,30-(time.monotonic()-last)) if count!=last_count else 30
            ready,_,_=select.select([watch],[],[],timeout)
            if ready:
                data=os.read(watch,65536);offset=0;handoff=False
                while offset<len(data):
                    which,mask,cookie,length=struct.unpack_from('iIII',data,offset);offset+=16
                    name=data[offset:offset+length].split(b'\0',1)[0].decode();offset+=length
                    if which==journal_wd:handoff=True
                    if which==wd and re.fullmatch(r'[0-9a-f]{64}\.json',name):
                        p=observations/name
                        if p.is_file() and not p.is_symlink():names.add(name)
                if handoff or parent_started(out):status='HANDOFF_ORIGINAL_PRODUCER_ACTIVE';break
                count=len(names)
            if count!=last_count and time.monotonic()-last>=30:
                send(progress_values(count,active(worker,parent,out,identity,pipe)));sent+=1
                last=time.monotonic();last_count=count
        else:status='WALL_BOUND_STOPPED_NO_RETRY'
    except Exception as error:
        status='HANDOFF_ORIGINAL_PRODUCER_ACTIVE' if parent_started(out) else 'LOGGING_BRIDGE_STOPPED'
        write(CONTROL/'error.json',dict(error_type=type(error).__name__,state=status,no_retry=True))
    finally:
        os.close(watch);os.close(fd);eventlog.close();singleton.close()
        write(CONTROL/'terminal.json',dict(status=status,requests_sent=sent,last_completed_case_count=count,
            scientific_completion_claim=False,remote_ACK_not_implied=True,no_retry=True,science_job_mutations=0))


def start(job):
    require(job=='60928' and not CONTROL.exists(),'CREATE_ONCE_BACKGROUND_UPLOADER')
    CONTROL.mkdir(mode=0o700)
    # This process is a bounded runtime telemetry writer, not an agent/scheduler
    # monitor. It has no SDK/auth/network client and cannot start science.
    keys=('PATH','LANG','LC_ALL','TMPDIR')
    env={k:os.environ[k] for k in keys if k in os.environ}
    env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(Path(__file__).resolve().parents[3]))
    log=(CONTROL/'controller.log').open('x')
    child=subprocess.Popen([sys.executable,'-u','-m',
        'project.run_scripts.gpt2xl_generation_baselines.live_progress','--job',job],
        env=env,cwd=Path(__file__).resolve().parents[3],stdin=subprocess.DEVNULL,
        stdout=log,stderr=log,start_new_session=True)
    log.close()
    proof=dict(pid=child.pid,job_id=job,source=member(Path(__file__)),CPU_max=1,memory_ceiling_bytes=4*1024**3,
        GPU=0,new_Slurm=0,new_SDK_or_remote_run=0,scientific_job_mutations=0,agent_monitor=False,
        lifecycle='initial W0 file events only; stop when original scalar producer takes over or task ends; no retry')
    write(CONTROL/'controller.json',proof)
    return proof


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);parser.add_argument('--start',action='store_true')
    args=parser.parse_args()
    if args.start:print(json.dumps(start(args.job)))
    else:bridge(args.job)
