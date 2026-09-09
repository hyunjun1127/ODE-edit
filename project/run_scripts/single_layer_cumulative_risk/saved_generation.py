"""Complete a prescribed saved-state generation observation; editing/optimizer0."""
import argparse
import json
import os
from pathlib import Path
import traceback
import torch
from scripts.fixed_counterfact import load_prefix
from .binding import DATA,MODEL,load_model
from .evaluation import generation,materialized
from .import_assets import ROOT,sha
from .objective import WEIGHT
from .panels import select
from .records import Ledger,save,tensor_sha
from .token_accounting import count_tokens

def validate_selection(snapshot,selection,expected_sha):
    selected=json.loads(selection.read_text())
    if selected['entry']!='Middle' or selected['support']!='C' or selected['PS_NS_influence']!=0:
        raise ValueError('WRONG_TRAIN_ONLY_SELECTION')
    if snapshot.parent!=Path(selected['endpoint_path']).parent or snapshot.name!='snapshot-008.pt':
        raise ValueError('WRONG_PREDECLARED_OBSERVATION')
    if snapshot.is_symlink() or sha(snapshot)!=expected_sha:raise ValueError('SNAPSHOT_IDENTITY_MISMATCH')
    return selected

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--expected-sha',required=True);parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    output=a.output.absolute();output.mkdir(parents=True,exist_ok=False)
    ledger=Ledger();w=w0=None;pointer=None
    try:
        selected=validate_selection(a.snapshot,a.selection,a.expected_sha)
        records=load_prefix(DATA,10000);panel=select(records,'Middle')
        state=torch.load(a.snapshot,map_location='cpu',mmap=True,weights_only=True)
        assert state['step']==8 and state['W'].dtype==torch.float32 and state['W'].shape==(4096,14336)
        assert torch.isfinite(state['W']).all()
        snapshot_weight_sha=tensor_sha(state['W'])
        model,_,tok=load_model(ledger);w=dict(model.named_parameters())[WEIGHT]
        w0=w.detach().clone();pointer=w.data_ptr()
        def counter(module,args,kwargs):
            ledger.add('actual_model_forward_invocations')
            ids=kwargs.get('input_ids')
            if ids is not None:
                ledger.add('actual_model_forward_sequences',ids.shape[0])
                count_tokens(ledger,ids,kwargs.get('attention_mask'))
        model.register_forward_pre_hook(counter,with_kwargs=True)
        save(output/'runtime.json',dict(stage='A_SUPPLEMENT_OBSERVATION_ONLY',entry='Middle',step=8,
             slurm_job=os.environ.get('SLURM_JOB_ID'),source_head=os.environ.get('CUMRISK_SOURCE_HEAD'),
             model_revision=Path(MODEL).name,snapshot_path=str(a.snapshot),snapshot_sha=a.expected_sha,
             snapshot_weight_sha=snapshot_weight_sha,selection_sha=sha(a.selection),alpha=selected['alpha'],
             input_lock_sha=sha(ROOT/'input.lock.json'),W0_sha=tensor_sha(w0),
             parameter_dtype_counts={'torch.float32':len(list(model.parameters()))},
             forward_dtype='torch.float32',autocast=torch.is_autocast_enabled(),
             tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
             optimization_steps=0,native_writer=0,z_recompute=0,scientific_promotion=False))
        with materialized(w,state['W'].cuda(),ledger):
            assert tensor_sha(w)==snapshot_weight_sha
            generation(model,tok,records,panel,ledger,output/'generation.json')
        observed=json.loads((output/'generation.json').read_text())
        assert len(observed['rows'])==60 and len({(r['case_id'],r['prompt_index']) for r in observed['rows']})==60
        assert w.data_ptr()==pointer and torch.equal(w,w0)
        save(output/'terminal.json',dict(status='TERMINAL_VALID',stage='A_SUPPLEMENT_OBSERVATION_ONLY',
             rows=60,W0_restored=True,pointer_restored=True,snapshot_sha=a.expected_sha,
             snapshot_weight_sha=snapshot_weight_sha,generation_sha=sha(output/'generation.json'),
             optimization_steps=0,native_writer=0,z_recompute=0,selection_influence=0,
             compute=ledger.receipt(),peak_gpu_bytes=torch.cuda.max_memory_allocated(),
             peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(),scientific_promotion=False))
    except BaseException as exc:
        restored=None
        if w is not None and w0 is not None:
            with torch.no_grad():w.copy_(w0)
            restored=w.data_ptr()==pointer and torch.equal(w,w0)
        save(output/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),W0_restored=restored,
             optimization_steps=0,compute=ledger.receipt()))
        raise

if __name__=='__main__':main()
