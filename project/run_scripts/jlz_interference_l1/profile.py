"""Three declared policies; same native optimizer, fresh own-state prices."""
import copy
from . import require

ARMS=('PRICE','FLAT','REVERSE')

def arm_profile(base,arm):
    require(arm in ARMS,'UNKNOWN_ARM')
    p=copy.deepcopy(base)
    p.update(arm=arm,eligible_layers=[4,5,6,7,8],anchor_layer=8,native_anchor_layer=8,
        n_exp=4,c=.75,lambda_KL=.0625,lambda_N=.5,lambda_C=15000,lr=.1,eps=1e-8,
        betas=[.9,.999],K_eval=25,max_updates=24,K_grace=12,tau_F=.05,blind=False,
        nll_readout_layer=31,request_gradient='SUM_OF_CURRENT_ACTIVE_REQUEST_LOSSES',
        model_dtype='FP32',geometry_dtype='FP64',TF32=False,autocast=False,
        price_relative_floor=1e-6,price_absolute_floor=1e-12,price_denominator_min=1e-8,
        constraint='ABSOLUTE_R_WEIGHTED_CAPPED_GROUP_L1',price_scope='OWN_BATCH_C0_ONCE')
    return p

def history_expected(arm):
    require(arm in ARMS,'UNKNOWN_ARM')
    return 100

def resource_order(parallel=2):
    require(parallel in (1,2),'LEGAL_LANES')
    return {'PRICE':[], 'FLAT':['PRICE'],
        'REVERSE':['PRICE'] if parallel==2 else ['FLAT'], 'collector':list(ARMS)}
