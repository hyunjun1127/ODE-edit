"""Full native geometry to same-dtype weight-space operators, no science change."""
from ..linear_solve import finite


def operators(geometries):
    """Preserve PCG-vector dtype; native factor/action internals remain FP64.

    This is explicit numerical storage adaptation, never a BF16/model cast.
    All full-P* axes stay available. Geometry state is fixed at We.
    """
    def mapped(method,kwargs):
        def action(values):
            if len(values)!=len(geometries):raise ValueError('NATIVE_SUPPORT_MAPPING')
            return finite(tuple(getattr(g,method)(x,**kwargs).to(x) for g,x in zip(geometries,values)))
        return action
    return dict(project=mapped('project',{}),
                native_metric=mapped('s_action',dict(normalized=True,projected=True)),
                native_metric_inverse=mapped('inverse',dict(normalized=True)))
