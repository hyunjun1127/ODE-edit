"""FP64 oracle reference with exact promotion of FP32 model/cache values.

Llama RMSNorm and eager Attention contain explicit float32 operations even
after model.double(). Instance-local function clones promote those two
scratch operations; the original class and FP32 production path stay intact.
"""
import ast,inspect,textwrap,types
from contextlib import contextmanager
import torch

def promote_function(fn):
 tree=ast.parse(textwrap.dedent(inspect.getsource(fn)));count=[0]
 class Precision(ast.NodeTransformer):
  def visit_Attribute(self,n):
   if isinstance(n.value,ast.Name) and n.value.id=='torch' and n.attr=='float32':
    count[0]+=1;return ast.copy_location(ast.Attribute(value=ast.Name(id='torch',ctx=ast.Load()),attr='float64',ctx=ast.Load()),n)
   return self.generic_visit(n)
 tree=Precision().visit(tree);ns=dict(fn.__globals__)
 exec(compile(ast.fix_missing_locations(tree),inspect.getsourcefile(fn)+':HJ_FP64_REFERENCE','exec'),ns)
 return ns[fn.__name__],count[0]

@contextmanager
def fp64_reference(model,oracle):
 originals=[];patches=[];parameter_ids={k:id(p) for k,p in model.named_parameters()}
 prefix,initial,kl=oracle.prefix,oracle.initial,oracle.kl
 try:
  torch.cuda.empty_cache();model.double();oracle.dtype(torch.float64)
  for name,module in model.named_modules():
   if type(module).__name__ in ['LlamaRMSNorm','LlamaAttention']:
    fn,count=promote_function(type(module).forward)
    assert count>0,'EXPECTED_LLAMA_FP32_SCRATCH'
    originals.append((module,module.__dict__.get('forward')));module.forward=types.MethodType(fn,module)
    patches.append(dict(module=name,kind=type(module).__name__,FP32_constants_promoted=count))
  assert patches and all(p.dtype==torch.float64 for p in model.parameters())
  yield patches
 finally:
  for module,method in reversed(originals):
   if method is None:del module.__dict__['forward']
   else:module.forward=method
  model.float();assert {k:id(p) for k,p in model.named_parameters()}==parameter_ids,'MODEL_PARAMETER_ALIAS_CHANGED'
  oracle.prefix,oracle.initial,oracle.kl=prefix,initial,kl;oracle.precision='float32'
  torch.cuda.empty_cache()
