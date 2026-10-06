"""Readonly MEMIT endpoint arithmetic; no scheduler or target-model access."""
import json
from pathlib import Path
from .common import *
from .w0 import RUNTIME_FIELDS
from .predecessor import PREVIOUS,SOURCE
from project.run_scripts.jlz_realized_writer_sequential.review_completed import paired,reduce_rows

def existing_memit(c,attempt,alpha_prefix,reader):
    result=[];old=json.loads(verify(c['models']['LLAMA']['predecessor_observation_config']).read_text())
    lock=reader.json(PREVIOUS/'execution.lock.json')
    require(lock['source_commit']==SOURCE and sha(PREVIOUS/'config.json')==lock['config_sha256'],'MEMIT_COMPARISON_SOURCE')
    from project.run_scripts.jlz_interference_l1.cap_w0 import chunks
    for model in MODELS:
        for arm in ARMS:
            cell=model+'_AE_'+arm;other=model+'_'+arm;root=PREVIOUS/other
            for k in MILESTONES:
                record=dict(cell=cell,reference=other,endpoint=f'W{k}',writer_difference='alphaedit vs native MEMIT',
                    planner_difference='writer-dependent response/price and subsequent own trajectories',status='NOT_AVAILABLE')
                if k not in alpha_prefix[cell]:result.append(record);continue
                commitpath=root/f'batch-{k:02d}/commit.json';folder=commitpath.parent/'post'
                if not commitpath.is_file() or not (folder/'summary.json').is_file():result.append(record);continue
                try:
                    current_runtime=reader.json(attempt/cell/'runtime.json');previous_runtime=reader.json(root/'runtime.json')
                    require(all(current_runtime[f]==previous_runtime[f] for f in RUNTIME_FIELDS),'HISTORICAL_RUNTIME_DIFFERENCE')
                    require(old['models'][model]['observation_identity']==c['models'][model]['observation_identity'],
                        'HISTORICAL_OBSERVATION_DIFFERENCE')
                    commit=reader.json(commitpath)
                    require(commit['source']==SOURCE and commit['arm']==other and commit['batch']==k
                        and commit['seen_requests']==100*k and commit['post_scope']=='ALL_SEEN','MEMIT_PREFIX_ENDPOINT_IDENTITY')
                    rows=[]
                    for path in sorted(folder.glob('chunk-*.json')):
                        chunk=reader.json(path);require(chunk['state']==commit['after'] and not chunk['optimizer_feedback'],'MEMIT_RAW_STATE')
                        rows.extend(chunk['rows'])
                    identities=reader.json(verify(c['models'][model]['observer_identity']))['rows']
                    ids=[i for p in c['models'][model]['packs'][:k] for i in p['ids']]
                    validate_rows(rows,identities,ids,f'W{k}')
                    from .collect import independent_rows,compare_summary
                    independent_rows(rows,expected_rows(identities,ids),f'W{k}')
                    metrics=reduce_rows(rows);compare_summary(metrics,commit['post'])
                    record.update(status='MATCHED_MODEL_COHORT_EVALUATOR_ENDPOINT',metrics=metrics,
                        paired=paired(rows,alpha_prefix[cell][k]),source=SOURCE,raw_files=[member(p) for p in sorted(folder.glob('chunk-*.json'))],
                        runtime_identity_verified=True,no_new_evaluation=True)
                except Exception as error:
                    record.update(status='HISTORICAL_REFERENCE_OR_UNVERIFIED',reason=str(error))
                result.append(record)
    return result
