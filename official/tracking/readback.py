"""Bounded one-shot finish evidence for last measured eval/fit rows, not all history."""
import itertools
import math
from .schema import ENTITY,PROJECT,require

def verify_last_rows(sdk,base_url,run_id,cfg,name,expected):
    scope='last logged evaluation and last fit row only; not full history or scientific completion'
    if not expected:return dict(status='NOT_MEASURED',scope=scope,rows=0)
    try:
        api=sdk.Api(overrides={'base_url':base_url},timeout=10)
        remote=api.run(ENTITY+'/'+PROJECT+'/'+run_id)
        require(remote.id==run_id and remote.name==name,'IDENTITY')
        require(all(remote.config.get(k)==v for k,v in cfg.items()),'CONFIG')
        checked=[]
        for kind,row in expected.items():
            values=row['values'];step=row['step']
            rows=list(itertools.islice(remote.scan_history(keys=['_step',*values],
                min_step=step,max_step=step+1,page_size=2),3))
            require(len(rows)==1 and rows[0].get('_step')==step,'ROW_MISSING_OR_DUPLICATE')
            require(all((k=='phase' and rows[0].get(k)==v and v in ('W0_generation','generation_evaluation','W20_generation')) or
                (k!='phase' and type(rows[0].get(k)) in (int,float,bool) and
                 math.isclose(rows[0][k],v,rel_tol=1e-9,abs_tol=1e-8)) for k,v in values.items()),'VALUE_MISMATCH')
            checked.append(dict(kind=kind,transport_step=step,edits=values.get('edits'),
                global_candidate=values.get('fit/global_candidate'),keys=sorted(values)))
        return dict(status='REMOTE_BOUNDED_ROWS_VERIFIED',scope=scope,checked=checked,rows=len(checked))
    except Exception:
        # Never expose remote SDK exception text or retry/modify the scientific run.
        return dict(status='UNVERIFIED_INCOMPLETE_OR_UNAVAILABLE',scope=scope,rows=None)
