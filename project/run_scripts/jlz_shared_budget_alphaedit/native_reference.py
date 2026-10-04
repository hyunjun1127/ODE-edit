"""Pinned native expression oracle; only CUDA placement becomes input.device for CPU fixtures."""
import ast
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import torch
from .common import require

@lru_cache(maxsize=4)
def expression(path):
    tree=ast.parse(Path(path).read_text())
    nodes=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='upd_matrix' for t in n.targets)
           and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='torch.linalg.solve']
    require(len(nodes)==2 and ast.dump(nodes[0].value)==ast.dump(nodes[1].value),'PINNED_NATIVE_OPERATOR')
    class Placement(ast.NodeTransformer):
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if isinstance(node.func,ast.Attribute) and node.func.attr=='cuda':
                node.func.attr='to';node.args=[ast.Attribute(value=ast.Name(id='layer_ks',ctx=ast.Load()),attr='device',ctx=ast.Load())]
            for kw in node.keywords:
                if kw.arg=='device' and isinstance(kw.value,ast.Constant) and kw.value.value=='cuda':
                    kw.value=ast.Attribute(value=ast.Name(id='layer_ks',ctx=ast.Load()),attr='device',ctx=ast.Load())
            return node
    node=Placement().visit(nodes[1].value)
    return compile(ast.fix_missing_locations(ast.Expression(body=node)),str(path),'eval')

@torch.no_grad()
def native_update(Pi,H,K,R,path):
    X=eval(expression(str(path)),{'torch':torch,'P':Pi[None], 'cache_c':H[None], 'i':0,
        'layer_ks':K,'resid':R,'hparams':SimpleNamespace(L2=10.)})
    return X.T
