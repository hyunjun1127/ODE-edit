"""CPU-only ordered W0 cohort reductions; never additional model forwards."""
from project.run_scripts.jlz_realization.common import require
from project.run_scripts.jlz_realization.observe import reduce_rows
from project.run_scripts.jlz_interference_l1.comparison_bridge import metric_row
from .gptj_server2_cohort_tracking.schema import metrics, AxisState, FIELDS

def curves(rows):
    require(len(rows)==26000,'FULL_ROWS')
    require([r['ordinal'] for r in rows]==[i for i in range(2000) for _ in range(13)],'ORDERED_OCCURRENCES')
    result=[];axis=AxisState()
    for x in range(0,2001,100):
        p=dict(edits=x,reference_cohort_edits=x,actual_model_edits=0,actual_applied_edits=0,
               pre_state_edits=0,post_state_edits=0)
        if x==0:p.update(metric_row('W0_first2000',reduce_rows(rows),2000))
        else:
            p.update(metric_row('current/post',reduce_rows(rows[(x-100)*13:x*13]),100))
            p.update({'w0/current/N/'+f:p['current/post/N/'+f] for f in FIELDS})
            if x%500==0:
                p.update(metric_row('all_seen/post',reduce_rows(rows[:x*13]),x))
                p.update({'w0/all_seen/N/'+f:p['all_seen/post/N/'+f] for f in FIELDS})
        metrics(p,scientific=True);axis.accept(p);result.append(p)
    return result
