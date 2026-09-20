"""Independent CPU algebra for full-bank KL weighting and own-arm history.

Synthetic three-word full vocabulary, unequal document lengths, nonuniform
teachers. Expected values/gradients use NumPy closed form, not runner helpers.
"""
import ast
import contextlib
import inspect
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from .objective import Objective


def features(index, length):
    return np.array([[.15*(1+index%5)+.07*j, (-1.)**(index+j)*(.2+.03*j)] for j in range(length)],dtype=np.float64)


def logp(array):
    shifted=array-array.max(axis=-1,keepdims=True)
    return shifted-np.log(np.exp(shifted).sum(axis=-1,keepdims=True))


class FakeOracle:
    def __init__(self, docs, source_weight, shared):
        self.device=torch.device('cpu');self.docs=docs;self.shared=shared;self.calls=[]
        self.teachers=[torch.tensor(logp(x @ source_weight.T),dtype=torch.float32) for x in docs]
        self._capsules=[dict(score_positions=list(range(len(x))),y0=[i%3]*len(x),ordinal=i,role='R512') for i,x in enumerate(docs)]
    def suffix_hidden(self,i,weight):
        self.calls.append(i)
        return torch.tensor(self.docs[i],dtype=torch.float32) @ weight.T
    def _physical_hidden(self,i):return self.suffix_hidden(i,self.shared.leaf)
    def _head(self,hidden,positions):return hidden[positions]
    def _teacher(self,i):return self.teachers[i]
    @contextlib.contextmanager
    def physical_weight(self,weight,gradient=False):
        self.shared.leaf=weight.detach().requires_grad_(gradient)
        try:yield self.shared.leaf
        finally:self.shared.leaf=None


def expected_block(oracle, indices, weight):
    """Document mean of full-vocab token KL and its analytical derivative."""
    losses=[];gradients=[]
    for i in indices:
        x=oracle.docs[i]
        teacher_logp=oracle.teachers[i].numpy().astype(np.float64)
        p=np.exp(teacher_logp)
        current=logp(x @ weight.T)
        q=np.exp(current)
        losses.append(np.mean(np.sum(p*(teacher_logp-current),axis=1)))
        # Retain the exact sealed teacher mass; no renormalization shortcut.
        gradients.append(((q*p.sum(axis=1,keepdims=True)-p).T @ x)/len(x))
    return float(np.mean(losses)),np.mean(gradients,axis=0)


def fixture(n=512):
    shared=SimpleNamespace(leaf=None)
    source=np.array([[.8,-.6],[-.3,.9],[.2,.4]])
    reference=FakeOracle([features(i,[1,2,4][i%3]) for i in range(n)],source,shared)
    obj=Objective.__new__(Objective)
    obj.ref=reference;obj.history={};obj.sweeps=[]
    obj.store=SimpleNamespace(indices=lambda role:tuple(range(n)))
    obj.counts=dict(reference_gradient=0,reference_candidate=0,reference_observer=0,history_gradient=0,history_candidate=0,technical_sweeps=0)
    weight=torch.tensor([[-.2,.7],[.4,-.5],[.1,.3]],dtype=torch.float32)
    def add_history(arm,ids,source_weight):
        oracle=FakeOracle([features(40+i,[1,3,5][i%3]) for i in range(len(ids))],source_weight,shared)
        rows=[dict(case_id=case,positions=list(range(len(oracle.docs[i]))),labels=[i%3]*len(oracle.docs[i])) for i,case in enumerate(ids)]
        obj.history[(arm,len(obj.history))]=dict(arm=arm,oracle=oracle,rows=rows,teachers=oracle.teachers)
        return oracle
    return obj,weight,add_history


class ObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.old_threads)

    def test_all512_equal_document_mean_not_token_pooling(self):
        obj,w,_=fixture()
        j,g,receipt=obj.evaluate(w,gradient=True)
        ej,eg=expected_block(obj.ref,range(512),w.double().numpy())
        self.assertAlmostEqual(j,ej,places=6)
        np.testing.assert_allclose(g.numpy(),eg,atol=2e-7,rtol=3e-6)
        self.assertEqual(obj.ref.calls,list(range(512)))
        self.assertEqual(receipt['reference_documents'],512)
        self.assertEqual(receipt['reference_positions'],sum(len(x) for x in obj.ref.docs))
        self.assertEqual(receipt['L_H'],0.)
        # Explicit unequal-length witness: pooled token KL would differ.
        values=[expected_block(obj.ref,[i],w.double().numpy())[0] for i in range(512)]
        pooled=np.average(values,weights=[len(x) for x in obj.ref.docs])
        self.assertGreater(abs(pooled-j),1e-4)
        self.assertEqual(receipt['dense_gradient_D2H'],1)
        self.assertEqual(obj.counts['reference_gradient'],1)

    def test_reference_plus_own_history_equal_blocks_and_analytic_gradient(self):
        obj,w,add=fixture()
        own=add('A',[7,8],np.array([[.1,.5],[1.,-.3],[-.6,.7]]))
        foreign=add('B',[7,9],np.array([[5.,5.],[-5.,2.],[3.,-4.]]))
        j,g,r=obj.evaluate(w,arm='A',active_ids=[7,8],gradient=True)
        jr,gr=expected_block(obj.ref,range(512),w.double().numpy())
        jh,gh=expected_block(own,[0,1],w.double().numpy())
        self.assertAlmostEqual(r['L_R'],jr,places=6)
        self.assertAlmostEqual(r['L_H'],jh,places=6)
        self.assertAlmostEqual(j,jr+jh,places=6)
        np.testing.assert_allclose(g.numpy(),gr+gh,atol=2e-7,rtol=3e-6)
        self.assertEqual(foreign.calls,[])
        self.assertEqual([x['case_id'] for x in r['rows']['history']],[7,8])
        self.assertEqual(r['history_requests'],2)
        self.assertGreater(abs(j-(jr*512+jh*2)/514),1e-3)

    def test_active_subset_and_missing_membership(self):
        obj,w,add=fixture(3)
        own=add('A',[7,8],np.array([[.1,.5],[1.,-.3],[-.6,.7]]))
        _,_,r=obj.evaluate(w,arm='A',active_ids=[8])
        self.assertEqual(own.calls,[1]);self.assertEqual(r['history_requests'],1)
        obj.ref.calls.clear()
        with self.assertRaisesRegex(ValueError,'HISTORY_MEMBERSHIP'):obj.evaluate(w,arm='B',active_ids=[8])
        self.assertEqual(obj.ref.calls,[])

    def test_physical_and_cached_use_same_full_objective(self):
        obj,w,add=fixture(4)
        add('A',[7],np.array([[.1,.5],[1.,-.3],[-.6,.7]]))
        a,ga,_=obj.evaluate(w,arm='A',active_ids=[7],gradient=True,indices=[0,2],route='cached')
        b,gb,_=obj.evaluate(w,arm='A',active_ids=[7],gradient=True,indices=[0,2],route='physical')
        self.assertEqual(a,b);self.assertTrue(torch.equal(ga,gb))
        self.assertEqual(obj.counts['technical_sweeps'],2)
        self.assertEqual(obj.counts['reference_gradient'],0)

    def test_full_vocabulary_loss_and_gradient_not_argmax_only(self):
        obj,w,_=fixture(2)
        j,g,_=obj.evaluate(w,gradient=True)
        # Every logit participates: a third vocabulary row has nonzero gradient,
        # despite supplied observer labels being only zero/one for this fixture.
        self.assertGreater(float(g[2].abs().sum()),1e-4)
        direction=np.array([[.2,-.3],[-.7,.4],[.6,.1]])
        h=1e-5
        jp,_=expected_block(obj.ref,[0,1],w.double().numpy()+h*direction)
        jm,_=expected_block(obj.ref,[0,1],w.double().numpy()-h*direction)
        self.assertAlmostEqual(float(np.sum(g.numpy()*direction)),(jp-jm)/(2*h),places=6)

    def test_dev_gradient_and_broad_technical_subset_rejected(self):
        obj,w,_=fixture(8)
        with self.assertRaisesRegex(ValueError,'DEV_OBSERVER_ONLY'):obj.evaluate(w,role='Dev128',gradient=True)
        with self.assertRaisesRegex(ValueError,'T0_REFERENCE_FOUR_MAX'):obj.evaluate(w,indices=range(5))
        self.assertEqual(obj.ref.calls,[])

    def test_dense_gradient_cpu_transfer_occurs_only_after_document_loops(self):
        # Structural check complements the numerical fixture: CPU execution
        # cannot establish CUDA transfer volume, so verify the actual operation
        # sits outside every loop and operates on the accumulated gradient.
        import textwrap
        tree=ast.parse(textwrap.dedent(inspect.getsource(Objective.evaluate)))
        calls=[]
        def visit(node,in_loop=False):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='cpu':
                calls.append((node,in_loop))
            for child in ast.iter_child_nodes(node):visit(child,in_loop or isinstance(node,(ast.For,ast.While)))
        visit(tree)
        self.assertEqual(len(calls),1)
        self.assertFalse(calls[0][1])
        self.assertEqual(ast.unparse(calls[0][0]),'accumulation.cpu()')
        source=inspect.getsource(Objective.evaluate)
        self.assertIn('device=self.ref.device',source)
        self.assertIn('accumulation.add_(g.double()',source)


if __name__=='__main__':unittest.main()
