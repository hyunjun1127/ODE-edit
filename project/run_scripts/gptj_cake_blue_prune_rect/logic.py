"""Pure profile identities/counts; no model execution or performance gate."""
ARMS=('CAKE','ALPHAEDIT_BLUE','PRUNE','RECT')
WRITERS=dict(CAKE='cake',ALPHAEDIT_BLUE='alphaedit-blue',PRUNE='memit-prune-base-fix',RECT='memit-rect')
def writer_identity(arm):
    if arm not in ARMS:raise ValueError('UNKNOWN_ARM')
    return WRITERS[arm]
def expected_counts(arm):
    if arm not in ARMS:raise ValueError('UNKNOWN_ARM')
    sites=2 if arm=='ALPHAEDIT_BLUE' else 6
    history=sites if arm in ('CAKE','ALPHAEDIT_BLUE') else 0
    return dict(native_z=200 if arm=='ALPHAEDIT_BLUE' else 100,
        write_keys=sites,history_keys=history,solves=sites,history_appends=history)
def lane_dependencies(arm,frontier,jobs,cap=2):
    if arm not in ARMS:raise ValueError('UNKNOWN_ARM')
    if cap not in (1,2):raise ValueError('TASK_CAP')
    if cap==1:
        index=ARMS.index(arm)
        return list(frontier) if index==0 else [jobs[ARMS[index-1]]]
    if arm in ('CAKE','ALPHAEDIT_BLUE'):return list(frontier)
    return [jobs['CAKE' if arm=='PRUNE' else 'ALPHAEDIT_BLUE']]

def fit_payload(value,last_trace=None):
    """Whitelist existing native scalars; no tensor reads or candidate changes."""
    data=dict(value)
    if last_trace and last_trace.get('request_index')==value.get('request_index'):
        for field in ('loss','nll_loss','kl_loss'):
            if field in last_trace:data.setdefault(field,last_trace[field])
    result={'batch':data['batch'],'candidate':data.get('request_index',data['native_z']),
        'fit/global_candidate':data['fit_global_candidate'],'optimizer/calls':data['fit_updates']}
    for field,key in (('loss','fit/loss'),('nll_loss','fit/nll'),('kl_loss','fit/kl')):
        if field in data:result[key]=data[field]
    if 'native_z_seconds' in data:result['time/phase_seconds']=data['native_z_seconds']
    return result
