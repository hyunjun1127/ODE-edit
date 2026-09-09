"""Training quantities at saved native/scaled states; observation only, no optimizer."""
import json
from pathlib import Path
import torch
from .algebra import action
from .binding import kernel
from .objective import DirectObjective,WEIGHT
from .panels import select
from .records import save,tensor_sha
from .import_assets import sha

def observer(model,tok,records,panel,contexts,we,m,j_native,ledger):
    requests=[dict(records[i]['requested_rewrite'],case_id=records[i]['case_id']) for i in panel['panels']['Current100']]
    for r in requests:
        if not r['target_new']['str'].startswith(' '):r['target_new']=dict(r['target_new'],str=' '+r['target_new']['str'])
    weight=dict(model.named_parameters())[WEIGHT];old=weight.detach().clone();pointer=weight.data_ptr()
    try:
        with torch.no_grad():weight.copy_(we)
        # Zero coefficient dimension is an observation-only interface. No
        # projected reconstruction of the saved physical endpoint is made.
        u=we.new_empty((we.shape[1],0));metric=we.new_empty((0,0))
        obj=DirectObjective(model,tok,requests,contexts,kernel().find_fact_lookup_idx,we,u,metric,j_native,ledger,
                            penalty_cross=we.new_empty((we.shape[0],0)),accumulate_weight_gradient=True)
    finally:
        with torch.no_grad():weight.copy_(old)
        assert weight.data_ptr()==pointer and torch.equal(weight,old)
    a=we.new_empty((we.shape[0],0))
    def observe(state,path):
        obj.forward.entry=state
        obj.penalty_constant=action(state-we,m)
        before=ledger.receipt()
        with ledger.time('native_train_quantity_observation'):terms=obj.evaluate(a,False)
        increment={k:v-before['counts'].get(k,0) for k,v in ledger.counts.items()}
        ledger.add('observation_only_train_kernel_forward',increment.get('training_forward',0))
        ledger.add('observation_only_train_kernel_sequences',increment.get('training_sequences',0))
        save(path,dict(**terms,selected_weight_sha=tensor_sha(state),teacher_entry_sha=tensor_sha(we),
              logical_requests=100,contexts_per_request=len(obj.contexts),backward=0,optimizer_steps=0,
              selection_influence=0,observation_only=True,count_increment=increment,compute=ledger.receipt()))
    return observe

def saved_middle(model,tok,records,w,w0,native,output,ledger):
    output.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((native/'prepared-receipt.json').read_text())
    assert sha(native/'prepared.pt')==receipt['sha256']
    p=torch.load(native/'prepared.pt',map_location='cpu',mmap=True,weights_only=True)
    assert torch.equal(p['W0'],w0.cpu())
    we,wn,m=p['We'].cuda(),p['WN'].cuda(),p['M'][0].cuda()
    observe=observer(model,tok,records,select(records,'Middle'),p['contexts'],we,m,p['J_native'],ledger)
    for name,state in [('W0',w0),('ENTRY',we),('N',wn)]+[(f'native-scale-{s}',we+s*(wn-we)) for s in [.25,.5,.75,1.25]]:
        observe(state,output/f'{name}-train.json')
    assert torch.equal(w,w0)
    save(output/'receipt.json',dict(prepared_sha=receipt['sha256'],states=7,editing_rerun=0,optimizer_steps=0,
          reason='Native/scaled six-context train quantities were absent; canonical rewrite NLL is not substituted.',
          old_results_mutation=0,compute=ledger.receipt()))
