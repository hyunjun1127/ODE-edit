import copy
import unittest
from .w0_ready_gate import validate


class GateTests(unittest.TestCase):
    def test_ready_and_rejection(self):
        state={'W':'cold','H':'zero'};config={'observer_identity':{'sha':'fixed'}}
        binding=dict(completed=True,requests=2000,source='source',config='config',state=state,observer_identity=config['observer_identity'])
        summary=dict(state=state,endpoint=0,requests=2000,row_count=26000,no_mutation=True,
                     summary={k:{'denominator':v} for k,v in {'R':2000,'P':4000,'N':20000}.items()})
        validate(binding,summary,state,config,'source','config')
        for field,value in [('completed',False),('source','wrong'),('state',{}),('requests',100)]:
            bad=copy.deepcopy(binding);bad[field]=value
            with self.assertRaises(RuntimeError):validate(bad,summary,state,config,'source','config')
        for field,value in [('no_mutation',False),('row_count',1300),('endpoint',1),('state',{})]:
            bad=copy.deepcopy(summary);bad[field]=value
            with self.assertRaises(RuntimeError):validate(binding,bad,state,config,'source','config')


if __name__=='__main__':
    unittest.main()
