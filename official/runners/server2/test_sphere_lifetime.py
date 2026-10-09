"""Small CPU projection parity and actual native apply-loop lifetime controls."""
import ast
import contextlib
import importlib
import inspect
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref
import torch

m=importlib.import_module('official.baselines.easyedit.models.SPHERE.SPHERE_main')

class LifetimeTests(unittest.TestCase):
    def exercise(self, legacy=False, projected=True):
        torch.manual_seed(20261010)
        model=torch.nn.Module();model.layers=torch.nn.ModuleList([torch.nn.Linear(7,4,bias=False) for _ in range(3)])
        original={n:p.detach().clone() for n,p in model.named_parameters()}
        deltas={n:torch.randn_like(p)*.03 for n,p in model.named_parameters()}
        hp=SimpleNamespace(P_loc='/fixture/not-loaded.pt',cumulative_ratio=.5 if projected else 0.,suppression_strength=.5)
        namespace=dict(m.__dict__)
        tree=ast.parse(inspect.getsource(m.apply_SPHERE_to_model))
        if legacy:
            class RemoveRelease(ast.NodeTransformer):
                def visit_Delete(self,node):
                    if [t.id for t in node.targets if isinstance(t,ast.Name)]==['upd_matrix_proj','P_soft','U']:return None
                    return node
            tree=RemoveRelease().visit(tree)
        exec(compile(ast.fix_missing_locations(tree),'<actual-native-loop>', 'exec'),namespace)
        namespace.update(P_loaded=True,cache_c_new=True)
        references=[];alive=[];fit_grad=[]
        def fit(*args,**kwargs):
            fit_grad.append(torch.is_grad_enabled());return deltas
        def projection(*args,**kwargs):
            if references:alive.append([r() is not None for r in references[-1]])
            result=m.sparse_projection(*args,**kwargs)
            references.append([weakref.ref(result[0]),weakref.ref(result[1]),weakref.ref(result[2]._base)])
            return result
        namespace.update(execute_AlphaEdit=fit,sparse_projection=projection)
        with patch.object(m.os.path,'exists',return_value=True),contextlib.redirect_stdout(io.StringIO()):
            result,copies=namespace['apply_SPHERE_to_model'](model,None,[],hp,return_orig_weights=True)
        self.assertEqual(fit_grad,[True]);self.assertIs(result,model)
        for name,value in copies.items():self.assertTrue(torch.equal(value,original[name]))
        return {n:p.detach().clone() for n,p in model.named_parameters()},alive

    def test_actual_loop_releases_before_next_eigh(self):
        actual,alive=self.exercise()
        self.assertEqual(alive,[[False,False,False],[False,False,False]])
        old,old_alive=self.exercise(legacy=True)
        self.assertEqual(old_alive,[[True,True,True],[True,True,True]])
        for key in actual:self.assertTrue(torch.equal(actual[key],old[key]))

    def test_disabled_projection_preserves_native_branch(self):
        a,alive=self.exercise(projected=False);b,_=self.exercise(legacy=True,projected=False)
        self.assertEqual(alive,[])
        for key in a:self.assertTrue(torch.equal(a[key],b[key]))

    def test_projection_formula_unchanged(self):
        torch.manual_seed(9);A=torch.randn(4,7);B=torch.randn(4,7)
        with torch.no_grad(),contextlib.redirect_stdout(io.StringIO()):
            actual,P,U=m.sparse_projection(A,B,eta=.5,alpha=.5)
            Ahat=A/(A.norm(dim=1,keepdim=True)+1e-8);C=Ahat.T@Ahat/Ahat.size(0)
            eigvals,eigvecs=torch.linalg.eigh(C)
            r=(torch.cumsum(eigvals.flip(0),0)/eigvals.sum()<=.5).sum().item()+1
            expectedP=torch.eye(A.size(1))-.5*(eigvecs[:,-r:]@eigvecs[:,-r:].T)
        self.assertTrue(torch.equal(P,expectedP));self.assertTrue(torch.equal(actual,B@expectedP.T))

if __name__=='__main__':unittest.main()
