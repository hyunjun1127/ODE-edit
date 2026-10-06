"""Exact native scalar consumption; no baseline optimizer is substituted."""
from .common import require
def arm_profile(hp,arm,writer):
    require(arm in ('CAP075','CAP100','FREE100') and hp.layers==[13,14,15,16,17],
            'GPT2_PROFILE_IDENTITY')
    require((hp.v_lr,hp.v_num_grad_steps,hp.v_loss_layer,hp.v_weight_decay,hp.kl_factor)==
            (.5,20,47,.5,.0625),'GPT2_EFFECTIVE_NATIVE_HPARAMS')
    return dict(arm=arm,model_profile='GPT2XL',model_type='gpt2',writer=writer,
        eligible_layers=list(hp.layers),anchor_layer=max(hp.layers),nll_layer=hp.v_loss_layer,
        nll_readout_layer=hp.v_loss_layer,expected_hidden=1600,expected_intermediate=6400,
        lr=hp.v_lr,K_eval=hp.v_num_grad_steps,max_updates=hp.v_num_grad_steps-1,
        terminal_candidate=hp.v_num_grad_steps-1,lambda_C=hp.mom2_update_weight if writer=='memit' else 0.,
        lambda_alpha=getattr(hp,'L2',None),lambda_KL=hp.kl_factor,lambda_N=hp.v_weight_decay,
        c=hp.clamp_norm_factor,beta_base=.75 if arm=='CAP075' else 1.,
        cap_mode='none' if arm=='FREE100' else 'native',beta_max_native_scale=.75,
        n_exp=4,K_grace=12,tau_F=.05,save_checkpoints=False)
