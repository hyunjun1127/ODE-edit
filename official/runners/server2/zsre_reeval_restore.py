"""Eval-only checkpoint restore. No native engine, editing, C0/P or generation."""
from pathlib import Path
import torch
from official.experiments.prepare import file_sha, read

def require(value, code):
    if not value:
        raise ValueError(code)

def verify_member(row):
    path=Path(row["path"])
    stat=path.stat()
    require(stat.st_size==row["bytes"], "INPUT_SIZE_MISMATCH")
    require(file_sha(path)==row["sha256"], "INPUT_SHA_MISMATCH")
    return path

def expected_names(model, method, hparams):
    parameters=dict(model.named_parameters())
    modules=[hparams["rewrite_module_tmp"].format(layer) for layer in hparams["layers"]]
    if method=="FT":
        names={name for name in parameters if any(module in name for module in modules)}
    else:
        names={module+".weight" for module in modules}
    require(bool(names) and names<=set(parameters), "RESTORE_PARAMETER_MAPPING")
    return names

def restore_weights(model, payload, row, hparams):
    """Restore selected FP32 W only; native history is not needed for evaluation."""
    require(payload.get("schema")=="official-baseline-checkpoint-v1", "CHECKPOINT_SCHEMA")
    require(payload.get("identity")==row["identity"] and payload.get("batch")==20
            and payload.get("method")==row["method"], "CHECKPOINT_IDENTITY")
    parameters=dict(model.named_parameters())
    names=expected_names(model,row["method"],hparams)
    require(set(payload["weights"])==names, "CHECKPOINT_WEIGHT_COVERAGE")
    before={k:(v.data_ptr(),v._version) for k,v in parameters.items() if k not in names}
    for name,value in payload["weights"].items():
        require(value.dtype==torch.float32 and parameters[name].dtype==torch.float32
                and value.shape==parameters[name].shape and bool(torch.isfinite(value).all()),
                "CHECKPOINT_WEIGHT_SHAPE_DTYPE_FINITE")
    with torch.no_grad():
        for name,value in payload["weights"].items():
            parameters[name].copy_(value)
            require(torch.equal(parameters[name],value.to(parameters[name].device)),
                    "CHECKPOINT_RESTORE_EXACT")
    require(before=={k:(v.data_ptr(),v._version) for k,v in parameters.items() if k not in names},
            "NONEDITED_PARAMETER_MUTATION")
    return dict(restored_parameters=sorted(names),batch=20,actual_edit_calls=0,
                history_restored=False,reason="forward-only; native H/context unused")

def load_and_restore(model, row, hparams):
    path=verify_member(row["checkpoint"])
    # The prior local full-SHA/identity allowlist is mandatory before deserialization.
    # mmap avoids materializing multi-GiB native history which evaluation never uses.
    payload=torch.load(path,map_location="cpu",mmap=True,weights_only=False)
    return restore_weights(model,payload,row,hparams)
