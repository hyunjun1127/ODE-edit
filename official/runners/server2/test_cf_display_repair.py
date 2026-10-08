"""Own actual caller -> shared schema and four-lane CPU regressions."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
from official.runners.server2 import run, cf_display_repair as repair
from official.runners.server2 import no_gpu_qualification as nq
from official.tracking import schema
from official.tracking.method import validate
from official.runners.server1.test_tracking_binding import cf_endpoint


class DisplayRepair(unittest.TestCase):
    def test_caller_companions_and_raw_unchanged(self):
        value=cf_endpoint(2000);before=copy.deepcopy(value)
        # This fixture isolates display; full prompt/token diagnostics are
        # covered separately by the actual saved W0 caller/schema replay.
        with patch.object(run,'cf_diagnostics',return_value={}):
            payload=run.evaluate_payload(value,'W0',0)
        validate(payload,official=True)
        self.assertEqual(value,before)
        for field in ('Efficacy','Generalization','Specificity'):
            self.assertIn('official/W0_first2000/'+field+'_AlphaEdit_display',payload)
        wrong=copy.deepcopy(payload)
        wrong['official/W0_first2000/Specificity_AlphaEdit_display']=80.003
        with self.assertRaises(ValueError): validate(wrong,official=True)

    def test_no_new_input_or_healthy_replacement(self):
        self.assertNotIn('CF_MEMIT',repair.ROLES)
        self.assertFalse(any(r.startswith('W0_') for r in repair.ROLES))
        self.assertTrue(all(nq.cell(r)[2]=='chain' for r in repair.ROLES))

    def test_complete_mixed_dag_width_four(self):
        jobs={};parents={}
        for role in repair.ROLES:
            parents[role]=repair.dependencies(role,jobs)
            jobs[role]=str(80000+len(jobs))
        alljobs=dict(repair.KEPT,**jobs)
        edges={alljobs[r]:set(parents[r]) for r in repair.ROLES}
        edges.update({'61725':set(),'61726':set(),'61728':set(),
            '61730':{'61726'},'61732':{'61728'},'61734':{'61730'},
            '61735':{jobs['CF_MEMIT_FE']}})
        todo=[frozenset()];seen=set(todo)
        while todo:
            done=todo.pop();ready=[j for j,p in edges.items() if j not in done and p<=done]
            self.assertLessEqual(len(ready),4)
            for job in ready:
                nxt=done|{job}
                if nxt not in seen: seen.add(nxt);todo.append(nxt)
        self.assertIn(frozenset(edges),seen)
        self.assertEqual(repair.dependencies('CF_ALPHAEDIT',{}),[])

    def test_zsre_science_argv_unchanged(self):
        self.assertEqual(repair.KEPT['ZSRE_SPHERE'],'61735')
        self.assertEqual(len(repair.KEPT),7)
        self.assertEqual(set(repair.ROLES),{'CF_ALPHAEDIT','CF_ALPHAEDIT_BLUE','CF_MEMIT_FE','CF_SPHERE'})


if __name__=='__main__':unittest.main()
