import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import torch
import transformers
from ..contracts import Ledger,tensor_sha,digest,save,tensor_save
from ..observations import JointView
from ..track_b.runtime import SelectedView
from ..track_b.followup import initial_reuse
from ..track_b.supplement import attribution
from .test_observations import Toy,rows

class Followup(unittest.TestCase):
    def test_reuse_and_actual_nonzero_probe_preserve_state(self):
        torch.manual_seed(36);model=Toy().eval().requires_grad_(False);ledger=Ledger()
        full=JointView(model,('first.weight','second.weight'),ledger)
        wn=(full.entry[0]+.02,full.entry[1]);view=SelectedView(full,wn,(8,))
        rr=rows()
        for i,r in enumerate(rr):r.update(ordinal=1000+(i==2),case_id=71+(i==2))
        chosen=[r for r in rr if r['ordinal']==1000]
        ref={'mean_nll':2.,'value':0.};candidate=(wn[0],wn[1]+.04)
        signatures={n:tensor_sha(w) for n,w in zip(full.names,wn)}
        problem=SimpleNamespace(support=(8,),wn=(wn[1],),fixture_identity={'common_ready_sha':'fixture','WN':signatures})
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'output';out.mkdir()
            save(out/'run.lock.json',dict(torch=torch.__version__,transformers=transformers.__version__,TF32=False))
            save(out/'INITIAL_VALID.json',dict(common_ready_sha='fixture',evidence=dict(FD_identity={'WN':[tensor_sha(wn[1])]},row_identity=digest([r['identity'] for r in chosen]))))
            save(out/'fd-refinement.json',{'status':'prior'})
            save(out/'nodes/node-00.json',{'before':[{}, {},ref]})
            tensor_save(out/'endpoint.pt',{'weights':candidate})
            save(out/'terminal.json',{'endpoint_sha':{n:tensor_sha(w) for n,w in zip(full.names,candidate)}})
            result=initial_reuse({'reference_run':td},problem,view,{'Current':rr},SimpleNamespace(pad_token_id=0),model,ledger,ref)
            self.assertEqual(result['functional_physical_max_abs'],0)
            self.assertEqual(result['FD_GGN_repeated'],0)
            full.assert_live(bytes_check=True)
            with self.assertRaisesRegex(ValueError,'REFERENCE_ACTUAL_FORWARD'):
                initial_reuse({'reference_run':td},problem,view,{'Current':rr},SimpleNamespace(pad_token_id=0),model,ledger,{'mean_nll':2.01,'value':0.})

    def test_signed_attribution_identity_and_sum(self):
        def row(nll):return [dict(panel='Current100',metric='RS',case_id=1,prompt_index=0,identity='same',new_nll=nll)]
        values=attribution(row(10),row(4),row(8),row(1))[0]
        self.assertEqual(values['signed_e4']+values['signed_e8'],9)
        self.assertEqual(values['interaction'],1)
        self.assertIsNone(attribution(row(1),row(2),row(3),row(4))[0]['positive_total_share_only'])
        changed=row(1);changed[0]['identity']='different'
        with self.assertRaises(ValueError):attribution(row(10),row(4),row(8),changed)

if __name__=='__main__':unittest.main()
