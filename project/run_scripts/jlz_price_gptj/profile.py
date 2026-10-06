"""GPT-J EasyEdit hparams; PRICE arm cap/base and method remain unchanged."""
from project.run_scripts.jlz_interference_l1.cap_profile import arm_profile as parent

def arm_profile(base,arm,writer):
    p=parent(base,arm,'LLAMA')
    p.update(model_profile='GPTJ',model_type='gptj',writer=writer,
        expected_hidden=4096,expected_intermediate=16384,nll_layer=27,nll_readout_layer=27,
        eligible_layers=[3,4,5,6,7,8],lr=.5,lambda_alpha=10.,adapter='gptj_parallel_fc_out')
    return p
