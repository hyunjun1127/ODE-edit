import ast
import importlib.util
import inspect
import math
import sys
from pathlib import Path
from project.run_scripts.joint_multilayer_bs10.server2_entry import bind
bind()
from project.run_scripts.joint_multilayer_bs10 import runtime as original
from project.run_scripts.joint_multilayer_bs10.common import require, sha

OLD = 'if loss < 5e-2:'
NEW = 'if nll_loss <= 1.0:'

def patch_source(text):
    require(text.count(OLD)==1 and NEW not in text,'EXACT_STOP_SOURCE')
    result=text.replace(OLD,NEW)
    a=ast.parse(text);b=ast.parse(result)
    old=[n for n in ast.walk(a) if isinstance(n,ast.If) and ast.unparse(n.test)=='loss < 0.05']
    new=[n for n in ast.walk(b) if isinstance(n,ast.If) and ast.unparse(n.test)=='nll_loss <= 1.0']
    require(len(old)==len(new)==1,'STOP_AST_COUNT')
    new[0].test=old[0].test
    require(ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),'ONLY_STOP_CHANGED')
    return result

def loss_line(fn):
    lines,start=inspect.getsourcelines(fn)
    found=[start+n.lineno-1 for n in ast.walk(ast.parse(''.join(lines)))
           if isinstance(n,ast.If) and ast.unparse(n.test)=='nll_loss <= 1.0']
    require(len(found)==1,'WEAK_TRACE_STOP_LINE')
    return found[0]

def finite_scalars(values):
    require(all(math.isfinite(float(x)) for x in values),'FIT_NONFINITE')

def stop_receipt(losses, adam):
    require(1<=len(losses)<=25 and adam==len(losses)-1,'FIT_WORK_COUNTS')
    finite_scalars(r['nll'] for r in losses)
    crosses=[i for i,r in enumerate(losses) if r['nll']<=1.0]
    if crosses:
        require(crosses==[len(losses)-1],'FIRST_CROSSING_ONLY')
        reason='INITIAL_BELOW_THRESHOLD' if len(losses)==1 else 'NLL_THRESHOLD_REACHED'
    else:
        require(len(losses)==25,'BUDGET_END')
        reason='BUDGET_EXHAUSTED_ABOVE_TARGET'
    return dict(reason=reason,first_crossing_iteration=crosses[0] if crosses else None,
                initial_nll=losses[0]['nll'],final_nll=losses[-1]['nll'],threshold=1.0,
                undershoot=1.0-losses[-1]['nll'] if crosses else None)

class WeakTrace(original.Trace):
    def __init__(self,native):
        self.native=native;self.rows=[];self.losses=[];self.history_calls=0;self.target_calls=0
        self.key_calls=0;self.key_rows=[];self.post_key=None;self.native_receipt={}
        self.loss_line=loss_line(native.compute_z);self.solve_calls=0;self.solve_layers=[]
        lines,start=inspect.getsourcelines(native.apply_AlphaEdit_to_model)
        self.solve_line=next(start+n.lineno-1 for n in ast.walk(ast.parse(''.join(lines)))
            if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='torch.linalg.solve')

    def __call__(self,frame,event,arg):
        d=frame.f_locals;fit=frame.f_code is self.native.compute_z.__code__
        if fit and event=='line' and frame.f_lineno==self.loss_line:
            finite_scalars(float(d[k].detach()) for k in ('nll_loss','kl_loss','weight_decay','loss'))
        if frame.f_code is self.native.apply_AlphaEdit_to_model.__code__ and event=='line' and frame.f_lineno==self.solve_line:
            # A multi-line call may emit repeated line events on its assignment.
            if d['layer'] not in self.solve_layers:self.solve_layers.append(d['layer'])
            self.solve_calls=len(self.solve_layers)
        result=super().__call__(frame,event,arg)
        if fit and event=='line' and frame.f_lineno==self.loss_line:
            self.losses[-1].update(context_nll=d['nll_loss_each'].detach().cpu().tolist(),
                delta_norm=float(d['delta'].detach().norm()))
        if fit and event=='return' and arg is not None:
            self.rows[-1]['stop']=stop_receipt(self.losses,self.rows[-1]['adam_updates'])
            finite_scalars([self.rows[-1]['target_norm'],self.rows[-1]['value_delta_norm']])
        if frame.f_code is self.native.apply_AlphaEdit_to_model.__code__ and event=='return' and arg is not None:
            require(self.solve_calls==5 and self.solve_layers==[4,5,6,7,8],'EXACT_FIVE_NATIVE_SOLVES')
        return result

class WeakRuntime(original.Runtime):
    def __init__(self,config,checkpoint,repo,patch):
        super().__init__(config,checkpoint,repo)
        path=Path(patch['path']);require(sha(path)==patch['sha256'],'WEAK_PATCH_SHA')
        old=Path(inspect.getsourcefile(self.native.compute_z))
        require(path.read_text()==patch_source(old.read_text()),'COMPUTE_Z_EXACT_DIFF')
        name='AlphaEdit.native_weak_compute_z';spec=importlib.util.spec_from_file_location(name,path)
        mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
        self.native.compute_z=mod.compute_z
        # Only this fresh process sees the adapter; original module bytes are untouched.
        original.Trace=WeakTrace
        self.total_solves=0

    def fit_batch(self,requests):
        result=super().fit_batch(requests)
        # Original fit_batch already checks exactly 1 target, 10 key captures,
        # 5 post-all-layer Gram appends and each exact updated history tensor.
        require(len(result['history'])==5 and len(result['targets'])==1,'WEAK_COUNTS')
        self.total_solves+=5
        result.update(stop_policy='native nll_loss <= 1.0 before backward',native_solves=5)
        return result
