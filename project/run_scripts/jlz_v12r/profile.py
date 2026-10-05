"""Five immutable arm profiles; no method or performance-dependent choice."""
import copy
import math
from . import require

ARMS=('MAIN','BLIND','L4-ONLY','BASE-2X','NO-EXPAND')

def arm_profile(base,arm):
    require(arm in ARMS,'UNKNOWN_ARM')
    p=copy.deepcopy(base)
    p.update(arm=arm,eligible_layers=[4] if arm=='L4-ONLY' else [4,5,6,7,8],anchor_layer=8,
        native_anchor_layer=8,n_exp=0 if arm in ('L4-ONLY','NO-EXPAND') else 4,
        base_multiplier=math.sqrt(2) if arm=='BASE-2X' else 1.,blind=arm=='BLIND',
        c=.75,lambda_KL=.0625,lambda_N=.5,lambda_C=15000,lr=.1,eps=1e-8,
        betas=[.9,.999],K_eval=25,max_updates=24,K_grace=12,tau_F=.05,
        nll_readout_layer=31,request_gradient='SUM_OF_CURRENT_ACTIVE_REQUEST_LOSSES',
        model_dtype='FP32',geometry_dtype='FP64',TF32=False,autocast=False)
    return p

def history_expected(arm):return 20 if arm=='L4-ONLY' else 100

def resource_order(parallel=2):
    require(parallel in (1,2),'LEGAL_LANES')
    result={'qualification':[], 'MAIN':['qualification']}
    if parallel==2:
        result.update({'BLIND':['MAIN'],'L4-ONLY':['MAIN'],'BASE-2X':['BLIND']})
    else:
        result.update({'BLIND':['MAIN'],'L4-ONLY':['BLIND'],'BASE-2X':['L4-ONLY']})
    result['NO-EXPAND']=['BLIND','L4-ONLY','BASE-2X']
    result['collector']=['qualification',*ARMS]
    return result
