"""One bounded history scan; verify all 21 curve payloads, no polling."""
import itertools
import math
from .schema import ENTITY,PROJECT,require

def verify_last_rows(sdk,base_url,run_id,cfg,name,expected):
    try:
        require(set(expected)=={str(x) for x in range(0,2001,100)},'LOCAL_CURVES_INCOMPLETE')
        remote=sdk.Api(overrides={'base_url':base_url},timeout=10).run(ENTITY+'/'+PROJECT+'/'+run_id)
        require(remote.id==run_id and remote.name==name and all(remote.config.get(k)==v for k,v in cfg.items()),'IDENTITY')
        rows=list(itertools.islice(remote.scan_history(page_size=100),101))
        require(len(rows)<=100,'BOUNDED_HISTORY')
        curves=[r for r in rows if r.get('reference_cohort_edits') is not None]
        require(len(curves)==21,'REMOTE_CURVE_COUNT')
        for got,want in zip(curves,expected.values()):
            require(got.get('_step')==want['step'],'STEP')
            require(all(type(got.get(k)) in (int,float) and math.isclose(got[k],v,rel_tol=1e-9,abs_tol=1e-8)
                for k,v in want['values'].items()),'VALUE')
        return dict(status='REMOTE_COHORT_CURVES_VERIFIED',full_W0=1,current=20,all_seen=4,
                    rows=21,scientific_completion_claim=False)
    except Exception:
        return dict(status='UNVERIFIED_INCOMPLETE_OR_UNAVAILABLE',scientific_completion_claim=False)
