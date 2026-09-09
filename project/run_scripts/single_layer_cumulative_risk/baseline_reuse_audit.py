"""Disclose repeated W0 evaluations; never average or replace recorded rows."""
import argparse
import json
from pathlib import Path
from .analysis import write_csv
from .import_assets import sha
from .records import save,digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--registry',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();groups={};inputs=[];total=0
    for entry,config in json.loads(a.registry.read_text()).items():
        root=Path(config['native']);runtime=json.loads((root/'runtime.json').read_text())
        path=root/'W0-full.json';inputs.append(dict(path=str(path),sha256=sha(path),W0_sha=runtime['W0_sha'],model_revision=runtime['model_revision']))
        for row in json.loads(path.read_text())['rows']:
            identity=(runtime['W0_sha'],runtime['model_revision'],row['identity'],row['metric'])
            groups.setdefault(identity,[]).append(dict(entry=entry,**row));total+=1
    repeated=[]
    for identity,rows in sorted(groups.items()):
        if len(rows)<2:continue
        repeated.append(dict(identity=identity,occurrences=len(rows),entries=[r['entry'] for r in rows],
             panels=[r['panel'] for r in rows],case_id=rows[0]['case_id'],
             new_nll_range=max(r['new_nll'] for r in rows)-min(r['new_nll'] for r in rows),
             true_nll_range=max(r['true_nll'] for r in rows)-min(r['true_nll'] for r in rows),
             success_disagreement=len({r['success'] for r in rows})>1))
    a.output.mkdir(parents=True,exist_ok=False);table=write_csv(a.output/'repeated-W0-rows.csv',repeated)
    save(a.output/'receipt.json',dict(status='OBSERVED_CROSS_ENTRY_REUSE_DEVIATION' if repeated else 'NO_REPEATED_W0_ROWS',
         input_rows=total,unique_state_prompt_pairs=len(groups),redundant_evaluated_pairs=total-len(groups),
         success_disagreement_count=sum(r['success_disagreement'] for r in repeated),
         max_new_nll_range=max((r['new_nll_range'] for r in repeated),default=0.),
         max_true_nll_range=max((r['true_nll_range'] for r in repeated),default=0.),
         inputs=inputs,inputs_root=digest(inputs),output=table,baseline_replacement=0,averaging=0,
         extra_sample_claim=0,editing_rerun=0,model_action=0))

if __name__=='__main__':main()
