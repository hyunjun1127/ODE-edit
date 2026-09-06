"""Process-local count-only instrumentation; scientific values are untouched."""
import ast,inspect,textwrap,time
import torch


class Accounting:
    def __init__(self, model):
        self.counts=dict(forward=0,forward_input_tokens=0,backward=0,linalg_solve=0,
            native_keys=0,native_keys_seconds=0.,linalg_solve_seconds=0.,history_key_captures=0)
        self.module=None;self.history_start=None
        def forward(module,args,kwargs):
            self.counts['forward']+=1
            x=kwargs.get('input_ids',args[0] if args else None)
            if x is not None:self.counts['forward_input_tokens']+=x.numel()
        self.hook=model.register_forward_pre_hook(forward,with_kwargs=True)
        self.backward=torch.autograd.backward;self.solve=torch.linalg.solve
        def backward(*a,**kw):
            self.counts['backward']+=1
            return self.backward(*a,**kw)
        def solve(*a,**kw):
            self.counts['linalg_solve']+=1
            start=time.perf_counter()
            try:return self.solve(*a,**kw)
            finally:self.counts['linalg_solve_seconds']+=time.perf_counter()-start
        torch.autograd.backward=backward;torch.linalg.solve=solve
    def bind_native(self,module):
        self.module=module;self.keys=module.compute_ks
        lines,start=inspect.getsourcelines(module.execute_AlphaEdit)
        tree=ast.parse(textwrap.dedent(''.join(lines)))
        loops=[n for n in ast.walk(tree) if isinstance(n,ast.For) and any(
            isinstance(a,ast.AugAssign) and 'cache_c' in ast.unparse(a.target) for a in ast.walk(n))]
        if len(loops)!=1:raise RuntimeError('PINNED_HISTORY_PROFILING_SITE')
        low,high=start+loops[0].lineno-1,start+loops[0].end_lineno-1
        def keys(*a,**kw):
            caller=inspect.currentframe().f_back
            history=(caller.f_code.co_name=='finalize' or
                (caller.f_code is module.execute_AlphaEdit.__code__ and low<=caller.f_lineno<=high))
            del caller
            if history:
                self.counts['history_key_captures']+=1
                if self.history_start is None:self.history_start=time.perf_counter()
            self.counts['native_keys']+=1;begin=time.perf_counter()
            try:
                value=self.keys(*a,**kw)
                if torch.cuda.is_available():torch.cuda.synchronize()
                return value
            finally:self.counts['native_keys_seconds']+=time.perf_counter()-begin
        module.compute_ks=keys
    def finish_history(self,excluded_endpoint_seconds=0.):
        if self.history_start is None:raise RuntimeError('HISTORY_CAPTURE_NOT_OBSERVED')
        value=time.perf_counter()-self.history_start;self.history_start=None
        return dict(seconds=value-excluded_endpoint_seconds,gross_seconds=value,
            excluded_endpoint_seconds=excluded_endpoint_seconds,
            scope='first actual post-key capture through transaction return; includes native tail/snapshot/restore, measured endpoint evaluator excluded for O',
            strict_append_only_seconds='NOT_SEPARATELY_ISOLATED',controller_influence_count=0)
    def snapshot(self):return dict(self.counts,wall=time.perf_counter())
    def difference(self,before):return {k:v-before[k] for k,v in self.snapshot().items()}
    def close(self):
        self.hook.remove();torch.autograd.backward=self.backward;torch.linalg.solve=self.solve
        if self.module is not None:self.module.compute_ks=self.keys
