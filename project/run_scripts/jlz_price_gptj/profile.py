"""GPT-J architecture only; inherited repaired PRICE arm coefficients unchanged."""
from project.run_scripts.jlz_interference_l1.cap_profile import arm_profile as parent

def arm_profile(base,arm,writer):
    p=parent(base,arm,'LLAMA')
    p.update(model_profile='GPTJ',model_type='gptj',writer=writer,
        expected_hidden=4096,expected_intermediate=16384,nll_layer=27,nll_readout_layer=27)
    return p
