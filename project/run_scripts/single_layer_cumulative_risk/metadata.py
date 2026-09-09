"""Outcome-independent overwrite/duplicate strata; no raw text in publication."""
from .panels import ENTRIES,select
from .records import digest

def panel_metadata(records,entry):
    _,offset=ENTRIES[entry];panel=select(records,entry)
    def pair(r):
        rw=r['requested_rewrite'];return digest([rw['subject'],rw.get('relation_id')])
    def target(r):return digest(r['requested_rewrite']['target_new']['str'])
    seen={};current={};rewrite_hashes=set()
    for i,r in enumerate(records[:offset+100]):
        rw=r['requested_rewrite'];rewrite_hashes.add(digest(rw['prompt'].format(rw['subject'])))
        (seen if i<offset else current).setdefault(pair(r),[]).append((i,target(r)))
    output=[]
    for name,indices in panel['panels'].items():
        for i in indices:
            r=records[i];identity=pair(r);new=target(r)
            subsequent=[(ordinal,t) for ordinal,t in seen.get(identity,[]) if ordinal>i]
            output.append(dict(entry=entry,panel=name,ordinal=i,case_id=r['case_id'],
                 subject_relation_cluster=identity,target_sha=new,
                 later_conflicting_seen_target=any(t!=new for _,t in subsequent),
                 current_conflicting_target=any(t!=new for _,t in current.get(identity,[])),
                 exact_neighbor_rewrite_overlap_count=sum(digest(p) in rewrite_hashes for p in r['neighborhood_prompts']),
                 semantic_conflict_detection=False,denominator_exclusion=0))
    return output
