"""Request-local irreversible activation and update-count radius expansion."""
import math
import torch


class RequestController:
    def __init__(self, anchors, anchor_star, layers, n_exp=4, base_multiplier=1,
                 grace=12, threshold=.05, c=.75):
        self.layers = tuple(layers)
        self.anchors = torch.as_tensor(anchors).double()
        self.anchor_star = torch.as_tensor(anchor_star, device=self.anchors.device).double()
        B = self.anchor_star.numel()
        if self.anchors.shape != (len(self.layers), B) or not self.layers:
            raise RuntimeError('CONTROLLER_ANCHOR_SHAPE')
        if not bool(torch.isfinite(self.anchors).all() and torch.isfinite(self.anchor_star).all()
                    and (self.anchors > 0).all() and (self.anchor_star > 0).all()):
            raise RuntimeError('CONTROLLER_NATIVE_ANCHOR')
        self.caps = c * self.anchors
        self.main_base = torch.minimum(c * self.anchors[0], c * self.anchor_star)
        self.maximum = c * self.anchor_star
        multiplier = math.sqrt(2) if base_multiplier in ('sqrt2', 'sqrt(2)') else float(base_multiplier)
        self.requested_base = multiplier * self.main_base
        self.base = torch.minimum(self.requested_base, self.maximum)
        self.clipped = self.requested_base > self.maximum
        self.n_exp, self.grace, self.threshold = int(n_exp), int(grace), float(threshold)
        if self.n_exp < 0 or self.grace < 0 or multiplier < 1:
            raise RuntimeError('CONTROLLER_PROFILE')
        self.expansion = torch.zeros(B, dtype=torch.int64, device=self.anchors.device)
        self.t = torch.zeros_like(self.expansion)
        self.active = torch.zeros(B, dtype=torch.bool, device=self.anchors.device)
        self.radii = self.base.clone()
        self.last_candidate = -1

    def observe(self, F, candidate):
        F = torch.as_tensor(F, device=self.anchors.device)
        if F.shape != self.active.shape or not bool(torch.isfinite(F).all()):
            raise RuntimeError('CONTROLLER_LOSS')
        if candidate != self.last_candidate + 1 or not 0 <= candidate <= 24:
            raise RuntimeError('CONTROLLER_CANDIDATE_SEQUENCE')
        self.active |= F >= self.threshold
        self.last_candidate = candidate
        terminal = candidate == 24 or not bool(self.active.any())
        return self.active.clone(), terminal

    def before_update(self, F):
        if not 0 <= self.last_candidate < 24 or not bool(self.active.any()):
            raise RuntimeError('TERMINAL_EXPANSION_FORBIDDEN')
        F = torch.as_tensor(F, device=self.anchors.device)
        if F.shape != self.active.shape or not bool(torch.isfinite(F).all()):
            raise RuntimeError('CONTROLLER_LOSS')
        if self.n_exp:
            expand = self.active & (self.t >= self.grace) & (F >= self.threshold) & (self.expansion < self.n_exp) & (self.base < self.maximum)
            self.expansion[expand] += 1
            self.radii = self.base * torch.pow(self.maximum / self.base, self.expansion.double() / self.n_exp)
            # The final stage is exactly the originally prescribed maximum;
            # avoid one-ulp formula drift changing the terminal state label.
            self.radii = torch.where(self.expansion == self.n_exp, self.maximum, self.radii)
        else:
            self.radii = self.base.clone()
        return self.radii.clone()

    def record_update(self, active):
        mask = torch.as_tensor(active, device=self.active.device, dtype=torch.bool)
        if not torch.equal(mask, self.active) or self.last_candidate == 24:
            raise RuntimeError('CONTROLLER_UPDATE_MASK')
        self.t[mask] += 1
        if bool((self.t > 24).any()):
            raise RuntimeError('CONTROLLER_UPDATE_BUDGET')

    def terminal_states(self, F):
        values = torch.as_tensor(F).detach().cpu().tolist()
        active, expansion, radii, maximum = [x.detach().cpu().tolist() for x in (self.active, self.expansion, self.radii, self.maximum)]
        states = []
        for r, loss in enumerate(values):
            if not active[r]:
                states.append('ZERO_STEP')
            elif loss < self.threshold:
                states.append('SATISFIED_EXPANDED' if expansion[r] else 'SATISFIED_BASE')
            else:
                states.append('UNSATISFIED_MAX' if radii[r] >= maximum[r] else 'UNSATISFIED')
        return states

    def receipt(self):
        return {
            'update_counts': self.t.cpu().tolist(), 'expansion': self.expansion.cpu().tolist(),
            'active': self.active.cpu().tolist(), 'radius': self.radii.cpu().tolist(),
            'anchor_values': self.anchors.cpu().tolist(), 'anchor_star': self.anchor_star.cpu().tolist(),
            'local_caps': self.caps.cpu().tolist(), 'eligible_layers': list(self.layers),
            'base_requested': self.requested_base.cpu().tolist(), 'base_effective': self.base.cpu().tolist(),
            'base2x_clipped': self.clipped.cpu().tolist(), 'base2x_clipped_count': int(self.clipped.sum()),
            'actual_base_energy_ratio': (self.base.square() / self.main_base.square()).cpu().tolist(),
            'radius_max': self.maximum.cpu().tolist(), 'n_exp': self.n_exp,
        }
