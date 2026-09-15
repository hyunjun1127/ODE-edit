"""Explicit deployment contract; CPU demo defaults never become runtime defaults."""
from copy import deepcopy
import math


def validate_config(contract):
    c=deepcopy(contract)
    s=c['integrator']; mode=s['barrier_mode']; budget=s['budget']
    if mode not in ('off','fixed','exponential'):
        raise ValueError('INVALID_BARRIER_MODE')
    if mode=='off' and budget is not None:
        raise ValueError('OFF_REQUIRES_EXPLICIT_NULL_BUDGET')
    if mode!='off' and (not isinstance(budget,(int,float)) or isinstance(budget,bool)
                       or not math.isfinite(budget) or budget<=0):
        raise ValueError('ON_REQUIRES_POSITIVE_BUDGET')
    if mode=='exponential' and s['kappa']<=0:
        raise ValueError('EXPONENTIAL_REQUIRES_POSITIVE_KAPPA')
    if c['runtime']['native_z_warm_start'] or c['objective']['native_endpoint_normalization']:
        raise ValueError('NO_NATIVE_ENDPOINT_PREPARATION')
    return c


def validate_main(contract):
    c=validate_config(contract);s=c['integrator']
    fixed={'barrier_mode':'off','budget':None,'initial_step':1.,'maximum_step':1.,
           'step_growth':1.5,'growth_after_accepts':2,'max_oracle_calls':25,
           'relative_stationarity':1e-5,'absolute_stationarity':1e-9,
           'active_tolerance':1e-8,'feasibility_tolerance':1e-10,
           'complementarity_tolerance':1e-7,'armijo':1e-4,'minimum_step':1e-12}
    if any(s[k]!=v for k,v in fixed.items()):
        raise ValueError('LOCKED_MAIN_INTEGRATOR')
    if (c['native_writer']['ridge']!=1 or c['objective']['price']!=1
        or c['objective']['beta_essence']!=.0625
        or c['geometry']['metric_ridge_relative']!=.001
        or c['runtime']['physical_layer']!=4):
        raise ValueError('LOCKED_MAIN_GEOMETRY_OBJECTIVE')
    return c
