"""Writer-specific unchanged fit diagnostics and price operators."""
from project.run_scripts.jlz_interference_l1 import cap_price
from project.run_scripts.jlz_price_alpha_writer import price as alpha_price
from . import memit_telemetry as cap_telemetry,alpha_telemetry

def initialize(a,*args,**kwargs):
    return (alpha_price if a.profile['writer']=='alphaedit' else cap_price).initialize(a,*args,**kwargs)

def candidate(a,*args,**kwargs):
    return (alpha_telemetry if a.profile['writer']=='alphaedit' else cap_telemetry).candidate(a,*args,**kwargs)

def terminal(a,*args,**kwargs):
    return (alpha_telemetry if a.profile['writer']=='alphaedit' else cap_telemetry).terminal(a,*args,**kwargs)
