"""Conservative maximum antichain of exact owned pending/allocation DAG."""
import itertools,re
from .common import require

def width(jobs):
    require(len(jobs)<=20 and all(j['gpus']==1 for j in jobs),'RESOURCE_GRAPH_UNSUPPORTED')
    ids={j['job'] for j in jobs};parents={}
    for j in jobs:
        dep=re.search(r'\bDependency=([^ ]+)',j['resource_detail'])
        require(dep is not None,'RESOURCE_DEPENDENCY_UNOBSERVED')
        text=dep[1]
        if text=='(null)':parents[j['job']]=set();continue
        # Unknown dependency syntax gives no concurrency credit, hence safely
        # increases the width and triggers a complete resource barrier.
        parents[j['job']]=set(re.findall(r'(?:^|:)(\d+)(?=\(|:|\?|$)',text))&ids if text.startswith('afterany:') and '?' not in text else set()
    reach={k:set(v) for k,v in parents.items()}
    for _ in ids:
        for k in ids:
            reach[k]|=set().union(*(reach[p] for p in tuple(reach[k]))) if reach[k] else set()
    require(all(k not in v for k,v in reach.items()),'RESOURCE_DAG_CYCLE')
    for n in range(len(ids),0,-1):
        if any(all(a not in reach[b] and b not in reach[a] for a,b in itertools.combinations(group,2))
               for group in itertools.combinations(ids,n)):return n
    return 0
