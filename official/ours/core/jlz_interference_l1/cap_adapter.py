"""Task-local model binding; unchanged parent Llama path and Qwen2 layout."""
import torch
from official.ours.config import require_config
from official.ours.core.jlz_native_writer_aware.physical import Adapter as Parent
from official.ours.common import require

class Adapter(Parent):
    def __init__(self,model,profile):
        profile=require_config(profile)
        require(model.config.model_type==profile['model_type'],'MODEL_PROFILE_TYPE')
        require((model.config.hidden_size,model.config.intermediate_size)==
            (profile['expected_hidden'],profile['expected_intermediate']),'MODEL_PROFILE_DIMS')
        if profile['model_type']=='llama':
            super().__init__(model,profile)
            return
        # Exact common validation from jlz_writer_coupled.physical constructor;
        # only architecture predicate and model-specific profile are generalized.
        require(model.config.model_type=='qwen2' and not model.config.use_sliding_window,'QWEN_LAYOUT')
        self.model,self.profile=model,profile;self.blocks=model.model.layers;self.sites=tuple(profile['eligible_layers'])
        require(self.sites==tuple(sorted(set(self.sites))) and 0<=min(self.sites)<=max(self.sites)<len(self.blocks),'ELIGIBLE_ORDER')
        self.first=min(self.sites)
        require(all(self.blocks[l].mlp.down_proj.bias is None for l in self.sites),'DOWN_PROJ_BIAS')
        self.weights={l:self.blocks[l].mlp.down_proj.weight for l in self.sites};ptrs={w.data_ptr() for w in self.weights.values()}
        require(len(ptrs)==len(self.sites) and sum(p.data_ptr() in ptrs for _,p in model.named_parameters(remove_duplicate=False))==len(self.sites),'PARAMETER_ALIAS')
        require(all(w.dtype==torch.float32 for w in self.weights.values()),'FP32_REQUIRED')
        self.device=next(model.parameters()).device;self.nll_layer=profile['nll_layer'];self.final_layer=len(self.blocks)-1
        require(max(self.sites)<=self.nll_layer<=self.final_layer,'QWEN_READOUT')
        self.dims={l:tuple(w.shape) for l,w in self.weights.items()};model.eval();model.requires_grad_(False)
        self.checkpoint_enabled=True
