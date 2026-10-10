import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from official.baselines.blue.rome.layer_stats import layer_stats
from official.baselines.blue.util import nethook
from official.baselines.blue.util.generate import generate_fast
from official.baselines.blue.util.globals import *

from official.baselines.blue.AlphaEdit.compute_ks import compute_ks
from official.baselines.blue.AlphaEdit.compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .Cake_hparams import CakeHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

def apply_Cake_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: CakeHyperParams,
    cache_template: Optional[str] = None,
    cache_c = None,
    P = None,
) -> Dict[str, Tuple[torch.Tensor]]:

    if hparams.temperature == "inf":
        temperature = float("inf")
    else:
        temperature = float(hparams.temperature)


    layer_weights_dict = compute_layer_weights(hparams.causal_scores, temperature)

    layer_weights = torch.tensor(
        [layer_weights_dict[i] for i in range(len(hparams.layers))],
        dtype=torch.float,
        device='cuda'
    )

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    
    for request in requests[:10]:
        print(
            f"MEMIT request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    
    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []

    for request in requests:
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if (
            cache_fname is not None
            and cache_fname.exists()
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to("cuda"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        if not data_loaded:
            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
            )
            z_list.append(cur_z)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{"v_star": cur_z.detach().cpu().numpy()},
                )
                print(f"Cached k/v pair at {cache_fname}")
    
    zs = torch.stack(z_list, dim=1)

    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T
        
        targets = zs - cur_zs  # R = z_target - z_current
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)

        current_weight = layer_weights[i]

        remaining_weight_sum = layer_weights[i:].sum()

        if remaining_weight_sum < 1e-6:
            effective_ratio = 1.0
        else:
            effective_ratio = current_weight / remaining_weight_sum
            
        resid = effective_ratio * targets
        
        print(f"Target delta norm for this layer: {torch.linalg.norm(resid):.4f}")


        upd_matrix = torch.linalg.solve(
            P[i,:,:].cuda() @ ((layer_ks @ layer_ks.T)
            + cache_c[i,:,:].cuda()) + 
            hparams.L2 * torch.eye(layer_ks.shape[0], dtype=torch.float, device="cuda"),
            P[i,:,:].cuda() @ layer_ks @ resid.T 
        )
        
        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))
        
        with torch.no_grad():
            weights[weight_name][...] = weights[weight_name] + upd_matrix
        
        # Clear GPU memory
        #del U,S,cov
        for x in [layer_ks, cur_zs, targets, upd_matrix]:
            x.cpu()
            del x
        torch.cuda.empty_cache()
    
    for i, layer in enumerate(hparams.layers):
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        cache_c[i,:,:] += layer_ks.cpu() @ layer_ks.cpu().T

    print(f"Deltas successfully computed for {list(weights.keys())}")
    return model, cache_c


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """
    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    return (
        torch.inverse(COV_CACHE[key].to("cuda")) if inv else COV_CACHE[key].to("cuda")
    )


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """
    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE


def compute_layer_weights(causal_scores: Dict[Any, float], temperature: float = 1.0) -> Dict[int, float]:

    causal_scores_int = {int(k): v for k, v in causal_scores.items()}
    
    layers = sorted(causal_scores_int.keys())
    scores = np.array([causal_scores_int[l] for l in layers])

    if temperature == float("inf"):
        return {l: 1.0 / len(layers) for l in layers}

    elif temperature == 0:
        max_idx = np.argmax(scores)
        return {l: 1.0 if i == max_idx else 0.0 for i, l in enumerate(layers)}

    else:
        exp_scores = np.exp((scores - np.max(scores)) / float(temperature))
        weights = exp_scores / np.sum(exp_scores)
        
        layer_weights = {layers[i]: float(weights[i]) for i in range(len(layers))}
        
        print("Layer weights computed:")
        for layer, weight in layer_weights.items():
            print(f"  Layer {layer}: {weight:.6f}")
    
        return layer_weights

def compute_optimal_deltas(
    layer_weights: torch.Tensor,
    targets: torch.Tensor,
    M_list: List[torch.Tensor],
    alpha: float = 1.0
) -> List[torch.Tensor]:

    num_layers = len(layer_weights)
    deltas = []
    
    for i in range(num_layers):
        delta_l = layer_weights[i] * targets
        deltas.append(delta_l)
        print(f"Layer {i} delta norm: {torch.linalg.norm(delta_l):.4f}, weight: {layer_weights[i]:.4f}")
    
    sum_deltas = sum(deltas)
    constraint_error = torch.linalg.norm(sum_deltas - targets)
    print(f"Constraint error ||Σ δ_l - R||: {constraint_error:.6f}")
    
    return deltas