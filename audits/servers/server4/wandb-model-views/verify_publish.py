"""Read-only remote membership check and compact receipt; no run mutation."""
import json
from pathlib import Path
import wandb
from inspect_views import ROOT,ENTITY,PROJECT
from update_views import IDS,model_filter

def main():
    api=wandb.Api(timeout=30)
    receipt=json.loads((ROOT/'update-receipt.json').read_text())
    membership={}
    for model in IDS:
        predicates=[]
        for leaf in model_filter(model)['filters']:
            key=leaf['key'];path=('config.'+key['name']) if key['section']=='config' else key['name']
            value=leaf['value'] if leaf['op']=='=' else {'$in':leaf['value']}
            predicates.append({path:value})
        membership[model]=sorted(r.id for r in api.runs(ENTITY+'/'+PROJECT,filters={'$or':predicates},per_page=100))
    assert not (set(membership['LLAMA'])&set(membership['QWEN']))
    assert not (set(membership['GPTJ'])&(set(membership['LLAMA'])|set(membership['QWEN'])))
    receipt.update(remote_membership=membership,metadata_tests_passed=4,
        verification_scope='GraphQL saved-spec readback and server run-filter membership; browser rendering not inspected',
        historical_baselines={'LLAMA':'9 imported baselines + W0 with source identity receipts',
                              'QWEN':'NOT_AVAILABLE','GPTJ':'NOT_AVAILABLE'},
        future_run_routing='cap_tracking.model_scoped_arm; existing helper whitelist arm only',
        source_audit='owner review; no independent reviewer',
        existing_production_archives_modified=False)
    dest=Path('audits/servers/server4/wandb-model-views/receipt.json')
    with dest.open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'remote_membership_counts':{m:len(ids) for m,ids in membership.items()},'receipt':str(dest)}))

if __name__=='__main__':main()
