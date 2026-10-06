"""Task caller of SH1's unmodified online scalar helper."""
from project.run_scripts.experiment_tracking import init
from .cap_common import require

def start(c,lock,out,cell):
    import os
    return init(env_file=c['tracking']['env_file'],spool=out/'tracking',config={
        'server':'server4','task_id':c['task_id'],'arm':cell,'attempt':c['run_instance']['attempt'],
        'source_sha':lock['source_commit'],'config_sha':lock['config_sha256'],
        'job_id':os.environ['SLURM_JOB_ID']})

class TrackedEvents:
    def __init__(self,events,tracker):self.events=events;self.tracker=tracker
    def __getattr__(self,name):return getattr(self.events,name)
    def emit(self,event,payload):
        self.events.emit(event,payload)
        if event=='candidate':
            B=len(payload['F']);self.tracker.log({'batch':self.events.batch,'candidate':payload['candidate'],
                'phase_id':1,'fit/loss':payload['J_mean'],'fit/nll':sum(payload['nll'])/B,
                'fit/kl':sum(payload['KL'])/B,'fit/norm':sum(payload['norm'])/B,
                'optimizer/calls':sum(payload['controller']['update_counts'])})

def log_endpoint(tracker,metrics,batch,edits):
    rates=[metrics[k]['rate'] for k in ('R','P','N')]
    values={'batch':batch,'edits':edits,'phase_id':2,
        'eval/RS':rates[0],'eval/PS':rates[1],'eval/NS':rates[2],
        'eval/harmonic':0. if min(rates)==0 else 3/sum(1/r for r in rates)}
    for kind in ('R','P','N'):
        for field in ('numerator','denominator'):values['eval/'+kind+'_'+field]=metrics[kind][field]
    tracker.log(values)
    for i,kind in enumerate(('R','P','N')):
        m=metrics[kind]
        tracker.log({'batch':batch,'edits':edits,'phase_id':3+i,'eval/TF_token_micro':m['token_micro'],
            'eval/TF_prompt_macro':m['prompt_macro'],'eval/TF_strict':m['strict_numerator']/m['strict_denominator'],
            'eval/true_nll':m['true_nll_mean'],'eval/new_nll':m['new_nll_mean']})
