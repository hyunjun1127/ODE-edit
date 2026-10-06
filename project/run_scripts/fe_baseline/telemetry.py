"""FE scalar mapping only; transport/auth/SDK remain SH1's shared helper."""
from project.run_scripts.experiment_tracking import init

def start(attempt,c,lock,job):
    return init(env_file=c['tracking']['env_file'],spool=attempt/'tracking',config=dict(
        server='server2',task_id=c['task_id'],arm='FE_MEMIT',attempt=attempt.name,
        source_sha=lock['source_commit'],config_sha=lock['config_sha256'],job_id=job))

def fit(tracker,index,values):
    if tracker is None:return
    payload={'phase_id':2,'candidate':index+1,'step':values['iteration'],
             'fit/loss':values['total'],'fit/nll':values['nll'],'fit/kl':values['kl'],
             'fit/norm':values['delta_norm']}
    tracker.log(payload)

def evaluation(tracker,result,batch):
    if tracker is None:return
    summary=result['summary'];payload={'phase_id':5,'batch':batch,'time/phase_seconds':result['seconds']}
    for kind in ('R','P','N'):
        v=summary[kind];payload['eval/'+kind+'S']=v['rate']
        payload['eval/'+kind+'_numerator']=v['numerator'];payload['eval/'+kind+'_denominator']=v['denominator']
    rates=[summary[k]['rate'] for k in ('R','P','N')]
    payload['eval/harmonic']=0. if min(rates)==0 else 3/sum(1/v for v in rates)
    tracker.log(payload)
    for i,kind in enumerate(('R','P','N'),1):
        v=summary[kind]
        tracker.log({'phase_id':5,'batch':batch,'retention/cohort_id':i,
            'eval/TF_token_micro':v['token_micro'],'eval/TF_prompt_macro':v['prompt_macro'],
            'eval/TF_strict':v['strict_numerator']/v['strict_denominator'],
            'eval/TF_token_correct':v['desired_token_correct'],'eval/TF_token_count':v['desired_token_count'],
            'eval/TF_strict_numerator':v['strict_numerator'],'eval/TF_strict_denominator':v['strict_denominator'],
            'eval/new_nll':v['new_nll_mean'],'eval/true_nll':v['true_nll_mean']})
