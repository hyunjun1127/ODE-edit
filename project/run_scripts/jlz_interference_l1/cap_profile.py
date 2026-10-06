"""Six explicit cells; independent price and cap/base knobs."""
import copy
from .cap_common import ARMS,MODELS,require

def arm_profile(base,arm,model):
    require(arm in ARMS and model in MODELS,'CELL_IDENTITY')
    p=copy.deepcopy(base)
    p.update(arm=arm,model_profile=model,model_type='llama' if model=='LLAMA' else 'qwen2',
        eligible_layers=[4,5,6,7,8],anchor_layer=8,native_anchor_layer=8,
        nll_layer=31 if model=='LLAMA' else 27,nll_readout_layer=31 if model=='LLAMA' else 27,
        expected_hidden=4096 if model=='LLAMA' else 3584,expected_intermediate=14336 if model=='LLAMA' else 18944,
        n_exp=4,c=.75,beta_max_native_scale=.75,beta_base=.75 if arm=='CAP075' else 1.,
        cap_mode='none' if arm=='FREE100' else 'native',lambda_KL=.0625,lambda_N=.5,lambda_C=15000,
        lr=.1,eps=1e-8,betas=[.9,.999],K_eval=25,max_updates=24,K_grace=12,tau_F=.05,blind=False,
        model_dtype='FP32',geometry_dtype='FP64',TF32=False,autocast=False,
        request_gradient='SUM_OF_CURRENT_ACTIVE_REQUEST_LOSSES',constraint='ABSOLUTE_R_WEIGHTED_GROUP_L1',
        price_scope='OWN_BATCH_C0_ONCE',projector_schema='PRICE_CAP_BASE_PROJECTION_V1')
    return p
