"""Fail-closed scientific/scope binding; no Slurm or model side effects."""
from dataclasses import asdict, dataclass
import math

ARMS = ("LEGACY_SCHEDULE_R512_G256", "REUSE_SCHEDULE_R512_G256")
INPUT_SHA = "507b202108acc90e10810455f57da60df14e9e9e31ba567c8c75f51cf76afaeb"
DATA_ID = "EN-R512-G256-v1"


class ScopeError(ValueError):
    pass


@dataclass(frozen=True)
class Scope:
    max_batches: int = 1
    batch_size: int = 100
    sequential_authorized: bool = False
    auto_continue: bool = False
    task_gpu_cap: int = 1
    project_gpu_cap: int = 2
    mem_mib: int = 60416
    cpus: int = 8
    arms: tuple = ARMS
    method_gradient_shared: bool = False
    reference_documents: int = 512
    dev_documents: int = 128
    max_new_tokens: int = 256
    trial_cap: int = 8
    gradient_sweeps_per_arm: int = 1
    inputs_sha256: str = INPUT_SHA
    data_id: str = DATA_ID

    def validate(self):
        expected = asdict(Scope())
        actual = asdict(self)
        for key, want in expected.items():
            got = actual[key]
            if type(got) is not type(want) or got != want:
                raise ScopeError("UNAUTHORIZED_SCOPE_" + key.upper())
        return self

    def require_batch(self, index):
        self.validate()
        if type(index) is not int or index != 0:
            raise ScopeError("B2_PLUS_REQUIRES_NEW_USER_AUTHORITY")

    def require_next(self):
        raise ScopeError("NO_B2_JOB_CALLBACK_DEPENDENCY_OR_AUTO_CONTINUATION")


def validate_runtime_policy(policy):
    """No technical waiver or method change inherited from older EN tasks."""
    exact = dict(model_revision="8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
                 physical_layer=4, dtype="float32", attention="eager",
                 matmul_tf32=False, cudnn_tf32=False, seed=20260916,
                 transformers="4.44.2", backtrack=.5, armijo=1e-4,
                 loss_floor=1e-6, current_individual_nll_allowance=1e-4,
                 gradient_accumulation="CPU_FP64_DOCUMENT_ORDER",
                 full_vocabulary=True, prior_T_skip_inherited=False,
                 storage_waiver_inherited=False, head_chunking_optimized=False,
                 generation_KV_optimized=False)
    for key, want in exact.items():
        got = policy.get(key)
        if type(got) is not type(want) or got != want:
            raise ScopeError("RUNTIME_POLICY_" + key.upper())
        if isinstance(got, float) and not math.isfinite(got):
            raise ScopeError("NONFINITE_POLICY")
    return policy


def runtime_policy():
    return dict(model_revision="8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
                physical_layer=4, dtype="float32", attention="eager",
                matmul_tf32=False, cudnn_tf32=False, seed=20260916,
                transformers="4.44.2", backtrack=.5, armijo=1e-4,
                loss_floor=1e-6, current_individual_nll_allowance=1e-4,
                gradient_accumulation="CPU_FP64_DOCUMENT_ORDER",
                full_vocabulary=True, prior_T_skip_inherited=False,
                storage_waiver_inherited=False, head_chunking_optimized=False,
                generation_KV_optimized=False)
