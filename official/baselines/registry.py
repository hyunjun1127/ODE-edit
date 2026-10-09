"""One implementation/hparams registry used by all server runners."""
from dataclasses import fields
from importlib import import_module
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "MEMIT_FE_HISTORY": ("memit_fe_history", "MEMITFEHyperParams", "apply_memit_fe_history_to_model"),
    "FT": ("easyedit.models.ft.ft_main", "FTHyperParams", "apply_ft_to_model"),
    "MEMIT": ("easyedit.models.memit.memit_main", "MEMITHyperParams", "apply_memit_to_model"),
    "MEMIT_LLAMA": ("sphere.memit.memit_main", "MEMITHyperParams", "apply_memit_to_model"),
    "ALPHAEDIT": ("easyedit.models.alphaedit.AlphaEdit_main", "AlphaEditHyperParams", "apply_AlphaEdit_to_model"),
    "ALPHAEDIT_BLUE": ("blue.AlphaEdit.AlphaEdit_main", "AlphaEditHyperParams", "apply_AlphaEdit_to_model"),
    "MEMIT_FE": ("easyedit.models.memit_FE.memit_FE_main", "MEMITFEHyperParams", "apply_memit_FE_to_model"),
    "SPHERE": ("easyedit.models.SPHERE.SPHERE_main", "SPHEREHyperParams", "apply_SPHERE_to_model"),
}


def implementation(method, model):
    """Import exclusively from this distribution, never a server's EasyEdit code."""
    key = "MEMIT_LLAMA" if (method, model) == ("MEMIT", "llama3") else method
    module_name, parser_name, apply_name = SPECS[key]
    module = import_module("official.baselines." + module_name)
    return module, getattr(module, parser_name), getattr(module, apply_name)


def hparams(method, model, *, overrides=None):
    profile = "MEMIT_FE" if method == "MEMIT_FE_HISTORY" else method
    values = json.loads((ROOT / "hparams" / profile / (model + ".json")).read_text())
    values.update(overrides or {})
    if values.get("L2", 1) is None:
        raise ValueError("QWEN_BLUE_L2_SELECTION_REQUIRED")
    _, cls, _ = implementation(method, model)
    allowed = {field.name for field in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError("UNSUPPORTED_HPARAMS: " + ",".join(sorted(unknown)))
    return cls(**values)


def requests(records, method, model):
    """Preserve the native input schema for each pinned implementation."""
    import copy
    easyedit = method != "ALPHAEDIT_BLUE" and (method, model) != ("MEMIT", "llama3")
    result = []
    for record in records:
        row = copy.deepcopy(record["requested_rewrite"])
        row["case_id"] = record["case_id"]
        if easyedit:
            row["target_new"] = row["target_new"]["str"]
            row["target_true"] = row["target_true"]["str"]
        if method == "FT":
            row["prompt"] = row["prompt"].format(row["subject"])
        result.append(row)
    return result


def call_options(method, model):
    """Required native call options; persistence belongs to the common checkpoint."""
    if (method, model) == ("MEMIT", "llama3"):
        return dict(copy=False, return_orig_weights=False, cache_template=None,
                    beta_hse=0, alpha=0, save_weights=False)
    if method == "ALPHAEDIT_BLUE":
        return dict(cache_template=None)  # caller supplies native cache_c and P
    return dict(copy=False, return_orig_weights=False)
