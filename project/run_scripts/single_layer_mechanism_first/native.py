"""Capture the actual solve result without another native fit or solve."""
import types
import time
from .z_hook import HookedNativeSingletonFitter


class CapturedHookedFitter(HookedNativeSingletonFitter):
    def _functions(self,counts,capture):
        fit,final=super()._functions(counts,capture)
        if capture is None:return fit,final
        ns=dict(fit.__globals__);original=ns['torch']
        class Proxy:
            def __init__(self,base,**values):self.base,self.values=base,values
            def __getattr__(self,key):return self.values[key] if key in self.values else getattr(self.base,key)
        def solve(*args,**kwargs):
            result=original.linalg.solve(*args,**kwargs)
            start=time.monotonic()
            capture.setdefault('solve_update',[]).append(result.T.detach().cpu().clone())
            counts['solve_output_capture_seconds']=counts.get('solve_output_capture_seconds',0.)+time.monotonic()-start
            return result
        ns['torch']=Proxy(original,linalg=Proxy(original.linalg,solve=solve))
        return types.FunctionType(fit.__code__,ns,fit.__name__,fit.__defaults__,fit.__closure__),final
