"""Explicit horizon override; FE mathematics and default2k stay unchanged."""
from . import require

def profile(requests=2000):
    require(type(requests) is int and requests in (2000,10000),'AUTHORIZED_HORIZON')
    batches=requests//100
    milestones=list(range(5,batches+1,5))
    # W0 + current pre/post + milestone extra seen. B1 pre reuses W0 rows.
    stored_rows=13*(requests+200*batches+sum(100*(b-1) for b in milestones))
    return dict(requests=requests,batches=batches,milestones=milestones,
        task_id=f'fe-sequential-{requests//1000}k',target_table_bytes=5*4096*requests*4,
        layer_solves=5*batches,history_appends=5*batches,own_joins=batches-1,
        planned_eval_rows=stored_rows,new_eval_rows=stored_rows-1300,
        final_denominators=dict(R=requests,P=2*requests,N=10*requests))
