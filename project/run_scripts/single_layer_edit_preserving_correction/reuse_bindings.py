"""Specific retained observer compatibility, no scheduler/model/evaluation."""
import json
from pathlib import Path
from .common import ROOT,member,sha,digest,write

OLD=Path('/data/janghj/ODE-edit/local/local-z-adaptive-allocation/20260916-v1')

def bind():
    new=json.loads((ROOT/'inputs/base-binding.json').read_text())
    old=json.loads((OLD/'execution.lock.json').read_text())
    cold=json.loads(Path(new['cold_capsule']['path']).read_text())
    for key in ('torch','transformers','seed','snapshot','projector','config4','editor_sha256','records_digest','sample_order'):
        if old[key]!=new[key]:raise ValueError('OBSERVER_RUNTIME_MISMATCH:'+key)
    if old['tf32_matmul'] or old['tf32_cudnn']:raise ValueError('PRIOR_TF32_MISMATCH')
    b1=OLD/'arms/N4/attempt-v1/output/B001'
    commit=json.loads((b1/'commit.json').read_text())
    native=json.loads((ROOT/'reuse/b001-native-binding.json').read_text())
    if commit['state']['W']['4']!=native['b1_endpoint_verified'] or commit['entry']['W']!=cold['W0']:
        raise ValueError('N4_OBSERVER_WEIGHT_LINEAGE')
    if commit['state']['W']['8']!=cold['W0']['8'] or commit['selected']!='N4' or not commit['actual_candidate']['L8_zero']:
        raise ValueError('PRIOR_NON_L4_OR_SELECTED_MISMATCH')
    if commit['entry']['contexts']!=digest(cold['contexts']):raise ValueError('PRIOR_CONTEXT_MISMATCH')
    result=dict(status='CPU_IDENTITY_BOUND_NOT_NEW_EVAL',native_binding=member(ROOT/'reuse/b001-native-binding.json'),
        prior_source=old['source_head'],prior_lock=member(OLD/'execution.lock.json'),commit=member(b1/'commit.json'),
        cold_capsule=new['cold_capsule'],N4_B1=member(b1/'selected-current.json'),
        W0_first1000=member(OLD/'technical/attempt-v1/W0-first1000.json'),
        compatibility=['same model revision/W0','same seed/nativeP/context actual tokens/hparams','FP32/eager/TF32off',
            'same exact selected L4 weight bytes and unchanged L8','same canonical evaluator/helper SHA and MB16'],
        W0_subset='old full1000 actual W0 pair rows; source population kept, no B100 padding bit-parity claim',
        raw_token_predictions='NOT_RETAINED_PRIOR; never reconstructed',new_canonical_forwards_for_covered_observations=0)
    for key in ('N4_B1','W0_first1000'):
        value=json.loads(Path(result[key]['path']).read_text())
        if value['evaluator_layout']!='HISTORICAL_MICROBATCH16_MANUAL_LEFT_PADDING_NO_POSITION_OVERRIDE':raise ValueError('LAYOUT_MISMATCH')
        for path,m in value['source_binding'].items():
            if sha(path)!=m['sha256']:raise ValueError('OBSERVER_SOURCE_BYTES')
        expected=100 if key=='N4_B1' else 1000
        if value['requests']!=expected or value['request_order']!=digest(new['sample_order'][:expected]):raise ValueError('OBSERVER_REQUEST_ORDER')
        for metric,n in (('RS',1),('PS',2),('NS',10)):
            if value['metrics'][metric]['denominator']!=expected*n:raise ValueError('OBSERVER_DENOMINATOR')
    output=ROOT/'reuse/observer-binding-r1.json'
    if output.exists():
        if json.loads(output.read_text())!=result:raise ValueError('CREATE_ONCE_BINDING_DRIFT')
        return member(output)
    return write(output,result)

if __name__=='__main__':print(json.dumps(bind()))
