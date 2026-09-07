"""One explicit, immutable configuration for solve, clocks and observations."""
from dataclasses import asdict, dataclass
import math

INSTRUCTION_ID = 'ODEEDIT-S06-ALPHA-JV-LLAMA-DIAGNOSIS-SWEEP-SH1-V1'
RUNTIME_HEAD = '77358b1546d1baf83b3e251afcce663b08d7bfd7'
PUBLICATION_HEAD = '0d0a0131e4a6a2a645dfa6530377d420a084d136'
CONTRACT_SHA256 = '98ee832e597b1f79c79ca48c62838af8f49f5306c08b3480b990d494f42f15f5'
LAYERS = (4, 5, 6, 7, 8)
ALIASES = ('llama3-8b-inst', 'qwen2.5-7b-inst')


class BindingBoundary(RuntimeError):
    """An input/lifecycle mismatch, never a low-performance filter."""


@dataclass(frozen=True)
class TrajectoryConfig:
    lambda_response: float
    T: float
    N: int
    normalization_id: str = 'N0_SOURCE'

    def __post_init__(self):
        if (isinstance(self.N, bool) or not isinstance(self.N, int) or self.N <= 0
                or not all(math.isfinite(v) and v > 0 for v in (self.T, self.lambda_response))
                or self.normalization_id not in ('N0_SOURCE', 'NRMS_ENTRY')):
            raise BindingBoundary('IMMUTABLE_CONFIG_BOUNDARY')

    @property
    def h(self):
        return self.T / self.N

    def receipt(self):
        return dict(asdict(self), h=self.h, actual_clock=math.fsum(self.h for _ in range(self.N)),
                    field_depends_on_total_horizon=False, first_hit_controls_dynamics=False,
                    adaptive_step_count=0, solve_gain_count=0)

    def endpoint_clock(self, completed_nodes):
        if not 0 < completed_nodes <= self.N:
            raise BindingBoundary('PREFIX_NODE_BOUNDARY')
        return dict(effective_T=completed_nodes*self.h, effective_N=completed_nodes,
                    h=self.h, parent_T=self.T, parent_N=self.N,
                    lambda_response=self.lambda_response, normalization_id=self.normalization_id)


def primary_residual(target, terminal):
    import torch
    if (target.dtype != torch.float32 or terminal.dtype != torch.float32
            or target.ndim != 2 or target.shape != terminal.shape
            or not torch.isfinite(target).all() or not torch.isfinite(terminal).all()):
        raise BindingBoundary('PRIMARY_FP32_RESIDUAL_BOUNDARY')
    return target-terminal
