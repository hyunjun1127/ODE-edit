"""Task-local native subject-row actions; frozen V13 readout/cache preserved."""
import torch

from project.run_scripts.jlz_realized_writer.capture import Adapter as V13Adapter
from project.run_scripts.jlz_realization.profile import move
from project.run_scripts.jlz_realization.common import require


class Adapter(V13Adapter):
    """Accept explicit global native rows, never infer an owner from row modulo."""

    def _native_action(self, group, actions, capture=False):
        rows = group['rows']
        ix = torch.arange(len(rows), device=self.device)
        lookup = torch.tensor([r['lookup'] for r in rows], device=self.device)
        require(set(actions) == set(self.sites), 'ALL_ELIGIBLE_ACTIONS')
        require(all(v.shape == (self.dims[l][0], len(rows)) and v.dtype == torch.float32
                    for l, v in actions.items()), 'PROJECTED_ACTION_SHAPE')
        self.calls['native_cached_forward' if self.native_route == 'cached'
                   else 'native_full_forward'] += 1

        def inject(x, action):
            y = x.clone()
            y[ix, lookup] = y[ix, lookup] + action.T
            return y

        if self.native_route != 'cached':
            subjects, handles = {}, []
            for l in self.sites:
                def hook(module, args, output, l=l):
                    y = inject(output, actions[l])
                    if capture:
                        subjects[l] = y[ix, lookup]
                    return y
                handles.append(self.blocks[l].register_forward_hook(hook))
            try:
                nh, fh = self.full(move(group['tokens'], self.device))
            finally:
                for handle in handles:
                    handle.remove()
        else:
            cache = move(group['cache'], self.device)
            kw = cache['kwargs']
            x = cache['residual'] + self.blocks[self.first].mlp.down_proj(cache['key'])
            x = inject(x, actions[self.first])
            subjects = {self.first: x[ix, lookup]} if capture else {}
            nh = x if self.nll_layer == self.first else None
            for l in range(self.first + 1, len(self.blocks)):
                if l in actions:
                    def step(h, action, l=l):
                        return inject(self.blocks[l](h, **kw), action)
                    x = self.recompute(step, x, actions[l])
                    if capture:
                        subjects[l] = x[ix, lookup]
                else:
                    def step(h, l=l):
                        return self.blocks[l](h, **kw)
                    x = self.recompute(step, x)
                if l == self.nll_layer:
                    nh = x
            fh = x
        if self.capture_virtual and capture:
            for j, row in enumerate(rows):
                self.last_virtual[row['global_row']] = {
                    l: h[j].detach().cpu().clone() for l, h in subjects.items()}
        return nh, fh, subjects

    def native(self, group, D, capture=False):
        """Unchanged owner-D path, retained solely as qualification reference."""
        owners = torch.tensor([r['request'] for r in group['rows']], device=self.device)
        return self._native_action(group, {l: d[:, owners] for l, d in D.items()}, capture)

    def native_projected(self, group, Y, capture=False, *, local_rows=False):
        """Y is global-row ordered unless the caller explicitly supplies local rows."""
        if local_rows:
            actions = Y
        else:
            indices = torch.tensor([r['global_row'] for r in group['rows']], device=self.device)
            actions = {l: value[:, indices] for l, value in Y.items()}
        return self._native_action(group, actions, capture)
