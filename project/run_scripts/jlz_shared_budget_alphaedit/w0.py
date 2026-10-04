"""Static completed W0 identity reuse, no old experiment monitoring."""
import json
from pathlib import Path
from .common import ROOT,require,sha,member,write,verify,digest
from project.run_scripts.jlz_realization.observe import reduce_rows

def bind_existing(previous,prior,config,out):
    folder=previous.parent/'shared-W0'
    if not (folder/'binding.json').exists():
        return dict(mode='FRESH_FIRST2000_ONCE',reason='NO_COMPLETED_EXACT_W0',new_W0_allowed=True)
    bound=json.loads((folder/'binding.json').read_text());summary=json.loads((folder/'summary.json').read_text())
    require(bound['completed'] and bound['requests']==2000 and summary['row_count']==26000 and summary['no_mutation'],'W0_COMPLETENESS')
    require(bound['config']==digest(prior) and bound['state']==summary['state'],'W0_ORIGINAL_BINDING')
    require(config['model']==prior['model'] and config['contexts']==prior['contexts'] and config['stream']==prior['stream'],'W0_ASSETS')
    for key in ('torch','transformers','torch_cuda'):
        require(config['runtime'][key]==prior['runtime'][key],'W0_RUNTIME:'+key)
    require([(r['module'],r['sha256']) for r in config['runtime']['source_members']]==
            [(r['module'],r['sha256']) for r in prior['runtime']['source_members']],'W0_RUNTIME_BYTES')
    require(config['observer_identity']['sha256']==prior['observer_identity']['sha256'],'W0_ALL_TOKEN_ROWS')
    original=previous.parent/'source'
    evaluator_paths=['project/run_scripts/jlz_realization/observe.py','project/run_scripts/jlz_realization/inputs.py']
    for relative in evaluator_paths:require(sha(original/relative)==sha(ROOT/relative),'W0_EVALUATOR_BYTES')
    expected={r['identity']:r for r in json.loads(verify(config['observer_identity']).read_text())['rows']}
    rows=[];members=[]
    for p in sorted(folder.glob('chunk-*.json')):
        payload=json.loads(p.read_text());require(payload['state']==bound['state'],'W0_ROW_STATE')
        for row in payload['rows']:
            e=expected[row['identity']]
            for key in ('case_id','kind','prompt_index','new_token_identity','true_token_identity','new_token_count','true_token_count'):
                require(row[key]==e[key],'W0_ROW_IDENTITY:'+key)
        rows.extend(payload['rows']);members.append(member(p))
    require(len(rows)==26000 and reduce_rows(rows)==summary['summary'],'W0_RAW_REDUCTION')
    receipt=dict(mode='EXACT_STATIC_W0_REUSE',prior_configuration=member(previous),binding=member(folder/'binding.json'),
        summary=member(folder/'summary.json'),rows=members,folder=str(folder),state=bound['state'],
        runtime_bytes_equal=True,evaluator_bytes_equal=True,all_26000_token_pairs_bound=True,
        assets_verification='prior_full_SHA_plus_current_stat',model_runtime_cold_state_recheck_required=True,new_W0_forward=0,
        old_experiment_monitoring=False,prior_method_not_inherited=True)
    write(out/'W0-reuse.json',receipt)
    return dict(mode=receipt['mode'],receipt=member(out/'W0-reuse.json'),folder=str(folder),state=bound['state'])

def rows_from(folder,ids=None):
    rows=[]
    for p in sorted(Path(folder).glob('chunk-*.json')):
        rows.extend(r for r in json.loads(p.read_text())['rows'] if ids is None or r['case_id'] in ids)
    return rows
