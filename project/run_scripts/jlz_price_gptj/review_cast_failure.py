"""CPU component regression from preserved native failure scalar operands.

No old R payload is reconstructed, no scientific toy/fit/model/GPU is run.
This certifies the declared conversion rule, not replay of the failed tensor.
"""
import argparse,json
from pathlib import Path
import torch
from .common import member,write,require
from project.run_scripts.jlz_interference_l1.cap_projection import first_store

def review(error_path,price_path,out):
    require(not out.exists(),'CREATE_ONCE_CAST_COMPONENT_RECEIPT')
    error=json.loads(error_path.read_text());price=json.loads(price_path.read_text())['payload']
    require(error['error']=='POSTCAST_PRICED_FEASIBILITY' and price['layers']==list(range(3,9)),'ACTUAL_NATIVE_FAILURE')
    caps=torch.tensor(price['local_caps'],dtype=torch.float64)
    stored=torch.tensor(error['operands']['stored_norm'],dtype=torch.float64)
    excess=torch.tensor(error['operands']['local_excess'],dtype=torch.float64)
    require(torch.equal(torch.clamp(stored-caps,min=0),excess),'PRESERVED_FAILURE_CAP_BINDING')
    violations=(excess>1e-6).nonzero().tolist()
    require(violations==[[0,29]],'EXACT_NATIVE_FAILURE_OWNER')
    # Actual cached scalar values exercise conversion, not a made-up model task.
    ideal=caps.reshape(1,-1);mask=torch.ones(ideal.shape[1],dtype=torch.bool)
    directed,evidence=first_store(ideal,mask,'cap_endpoint_toward_zero_v1')
    require(bool((directed.double().abs()<=ideal.abs()).all()),'DIRECTED_COMPONENT_BOUND')
    ordinary,_=first_store(ideal,mask,'nearest')
    require(torch.equal(ordinary,ideal.float()),'ORIGINAL_DEFAULT_CAST_UNCHANGED')
    interior,_=first_store(ideal,torch.zeros_like(mask),'cap_endpoint_toward_zero_v1')
    require(torch.equal(interior,ideal.float()),'POSITIVE_INTERIOR_NEAREST_UNCHANGED')
    exact_zero,_=first_store(torch.zeros_like(ideal),mask,'cap_endpoint_toward_zero_v1')
    require(torch.equal(exact_zero,torch.zeros_like(directed)),'ZERO_STORAGE_UNCHANGED')
    require(not torch.cuda.is_initialized(),'CPU_ONLY')
    result=dict(status='PASS_NATIVE_SCALAR_CAST_COMPONENT_REGRESSION',error=member(error_path),price=member(price_path),
        tested_source=member(Path(__file__)),cast_source=member(Path(__file__).parents[1]/'jlz_interference_l1/cap_projection.py'),
        actual_violation=dict(layer=3,owner=29,cap=float(caps[0,29]),stored_norm=float(stored[0,29]),excess=float(excess[0,29])),
        actual_shared_max_excess=max(error['operands']['shared_excess']),fixed_local_limit=1e-6,
        cached_scalar_component_values=caps.numel(),changed_components=sum(evidence['changed_component_count']),
        default_nearest_unchanged=True,positive_interior_unchanged=True,zero_unchanged=True,
        max_directed_scalar_excess=float(torch.clamp(directed.double()-ideal,min=0).max()),
        old_tensor_replay='NOT_AVAILABLE_NO_CP',target_model_GPU_parity='NOT_OBSERVED',
        scientific_toy_runs=0,model_load=0,new_fit=0,GPU_calls=0,tolerance_change=False,
        endpoint_rounding_explicit=True,postcast_feasibility_rescue=False)
    write(out,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--error',type=Path,required=True)
    p.add_argument('--price',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);print(json.dumps(review(a.error,a.price,a.out)))
