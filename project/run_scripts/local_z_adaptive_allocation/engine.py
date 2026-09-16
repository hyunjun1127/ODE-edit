"""Episode-local fit sharing; no case-only or cross-arm cache exists."""
import copy
import math
import time
import torch
from .common import EXPECTED, digest, save, tensor_save
from .policy import materialized, choose, strict_shadow
from .model import tensor_sha, restore_rng

def generate(rt,records,arm,root,*,technical=False,a4_order=None):
    root.mkdir(parents=True,exist_ok=False)
    entry=rt.snapshot();before=rt.state();candidates={};fits={};meta={};targets=solves=0
    def record_fit(key,result):
        nonlocal targets,solves
        receipt=result['receipt'];targets+=receipt.get('compute_z',0);solves+=receipt.get('solve',0)
        receipt['cache_identity']=digest(dict(entry=receipt.get('input_state',rt.state()),layer=receipt['layer'],contexts=rt.context,
            request=[r['case_id'] for r in records],source=rt.lock['editor_sha256'],hp=vars(rt.hp[receipt['layer']]),
            target_mode='TERMINAL_FIXED8' if 'readout_layer' in receipt else 'LOCAL_FRESH'))
        fits[key]=result
        tensor_save(root/(key+'.pt'),result)
        save(root/(key+'.json'),receipt)
    def add(name,w4,w8,a4,a8,mode):
        candidates[name]={4:w4,8:w8}
        action=math.sqrt(sum(float((candidates[name][l].double()-entry['W'][l].double()).square().sum()) for l in (4,8)))
        meta[name]=dict(candidate_id=name,a4=a4,a8=a8,target_mode=mode,action_norm=action,
            L8_zero=torch.equal(w8,entry['W'][8]),weights={str(l):tensor_sha(w) for l,w in candidates[name].items()})
    try:
        if arm not in ('T75',):
            rt.restore(entry);native=rt.fit(records,4);record_fit('own-N4',native)
            n4=native['weight']
            if arm in ('N4','L4D','LD','TD'):add('N4',n4,entry['W'][8],1.,0.,'LOCAL')
            if arm=='REFIT4':
                partial=materialized(entry['W'][4],n4,.75);rt.apply({4:partial,8:entry['W'][8]},entry['rng'])
                second=rt.fit(records,4);record_fit('refit4',second)
                add('REFIT4',second['weight'],entry['W'][8],.75,0.,'FRESH_SAME_LAYER_SECOND_FIT')
            if arm in ('L4D','LD','L75'):
                order = [.75,1.] if arm=='LD' else [.75]
                if a4_order is not None:
                    assert technical and arm=='LD' and sorted(a4_order)==[.75,1.]
                    order=list(a4_order)
                for a4 in order:
                    partial=materialized(entry['W'][4],n4,a4)
                    if arm in ('L4D','LD'):add(f'L{a4:g}-0',partial,entry['W'][8],a4,0.,'LOCAL')
                    if arm=='L4D':continue
                    rt.apply({4:partial,8:entry['W'][8]},entry['rng'])
                    second=rt.fit(records,8);record_fit(f'local8-a4-{a4:g}',second)
                    for a8 in ([.5,1.] if arm=='LD' else [1.]):
                        add(f'L{a4:g}-{a8:g}',partial,materialized(entry['W'][8],second['weight'],a8),a4,a8,'LOCAL')
                # LOCAL (1,0) is common N4, not an extra declared candidate.
                candidates.pop('L1-0',None);meta.pop('L1-0',None)
        if arm in ('T75','TD'):
            rt.restore(entry);z,zreceipt,observations=rt.targets8(records);targets+=zreceipt['compute_z']
            tensor_save(root/'terminal-Z8.pt',dict(Z8=z,observations=observations,receipt=zreceipt))
            first=rt.terminal_fit(records,4,z);record_fit('terminal4',first)
            for a4 in ([.75,1.] if arm=='TD' else [.75]):
                partial=materialized(entry['W'][4],first['weight'],a4)
                if arm=='TD':add(f'T{a4:g}-0',partial,entry['W'][8],a4,0.,'TERMINAL')
                rt.apply({4:partial,8:entry['W'][8]},entry['rng'])
                second=rt.terminal_fit(records,8,z);record_fit(f'terminal8-a4-{a4:g}',second)
                for a8 in ([.5,1.] if arm=='TD' else [1.]):
                    add(f'T{a4:g}-{a8:g}',partial,materialized(entry['W'][8],second['weight'],a8),a4,a8,'TERMINAL')
        assert (targets,solves,len(candidates))==EXPECTED[arm],('PROPOSAL_INVENTORY',arm,targets,solves,len(candidates))
        rt.restore(entry);assert rt.state()==before
        save(root/'generation.json',dict(arm=arm,entry=before,targets=targets,solves=solves,declared_candidates=list(meta.values()),
            inner_history_append=0,candidate_rng='RESTORE_BATCH_ENTRY_BEFORE_EACH_FIT_AND_EVAL; COMMIT_BATCH_ENTRY_RNG',
            fit_sharing='EPISODE_LOCAL_ONLY_SAME_A4_A8_VARIANTS',technical=technical))
        return candidates,meta,entry,fits
    except BaseException:
        rt.restore(entry);raise

def score_and_select(rt,records,past,arm,candidates,meta,entry,root,*,order=None):
    rows=[];cache={};state0=rt.state();start=time.monotonic()
    for name in (order or list(candidates)):
        m=meta[name]; key=digest(m['weights'])
        rt.apply(candidates[name],entry['rng'])
        if key in cache:
            scores=copy.deepcopy(cache[key][1]);reuse=cache[key][0]
        else:
            scores=rt.score(records,past);cache[key]=(name,copy.deepcopy(scores));reuse=None
        row=dict(m,**scores,score_reused_from=reuse)
        save(root/('candidate-'+name+'.json'),row);rows.append(row)
    if arm in ('L4D','LD','TD'):
        selection=choose(rows)
        selection['shadow_no_plateau']=choose(rows,plateau=False)['selected']
        selection['shadow_strict_only']=strict_shadow(rows)
    else:
        assert len(rows)==1; selection=dict(selected=rows[0]['candidate_id'],quality_screen='NOT_APPLIED_FIXED_POLICY')
    rt.restore(entry);assert rt.state()==state0
    save(root/'selection.json',dict(selection,seconds=time.monotonic()-start,all_declared_candidates_scored=True,
        distinct_materializations=len(cache),declared=len(rows),no_P_N_Dev_inputs=True))
    return selection,rows
