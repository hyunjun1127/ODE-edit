"""CPU handoff checks: missing evidence, failure, alias and cost semantics."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from . import metrics
from .report import report


def observation():
    rows=[]
    for family in ('RS','PS','NS'):
        rows.append(dict(case_id=1,family=family,prompt_index=0,identity=family,token_identity='t',success=True,
            new_nll=1.,true_nll=2.,desired_nll=1. if family!='NS' else 2.,desired_margin=1.,
            desired_token_correct=[True],desired_token_count=1,desired_strict=True))
    return metrics.summarize(rows,[1])


def write(root,name,value):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


class ReportTests(unittest.TestCase):
    def test_missing_evidence_never_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'output';out.mkdir()
            write(out,'complete.json',{'status':'B300_COMPLETE','seconds':30})
            manifest=report(out,root/'report')
            self.assertEqual(manifest['status'],'INCOMPLETE')
            self.assertTrue(manifest['missing_required_evidence'])
            text=(root/'report/report.md').read_text()
            self.assertIn('NOT_ESTABLISHED',text)
            self.assertNotIn('**B300_COMPLETE**',text)

    def test_failure_overrides_complete_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'output';out.mkdir()
            write(out,'complete.json',{'status':'B300_COMPLETE'})
            write(out,'technical-failure.json',{'status':'TECHNICAL_FAILURE','traceback':'raw private prompt should not be copied'})
            manifest=report(out,root/'report')
            self.assertEqual(manifest['status'],'TECHNICAL_FAILURE')
            self.assertNotIn('raw private prompt',(root/'report/report.md').read_text())

    def test_fallback_alias_and_full_standalone_shared_cost(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'output';out.mkdir();obs=observation()
            write(out,'B1/SHARED-native.json',{'seconds':10})
            write(out,'B1/SHARED-spectrum.json',{'timing':{'current_capture':1,'weighted_TSQR_SVD':2,'spectrum_projection':3,'objective_gradient':4},'selection':{'frontiers':{}}})
            write(out,'B1/W0-current.json',obs)
            for arm in ['N4','EN_ADAPT']:
                write(out,f'B1/{arm}-metrics.json',obs)
                write(out,f'B1/{arm}-paired-analysis.json',{})
                write(out,f'B1/{arm}-summary.json',{'selection_status':'NATIVE_BASELINE' if arm=='N4' else 'SEARCH_LIMIT_NO_ACCEPTED_CANDIDATE','observer_alias':None if arm=='N4' else 'N4','observer_seconds':2})
            write(out,'B1/EN_ADAPT-controller.json',{'status':'SEARCH_LIMIT_NO_ACCEPTED_CANDIDATE','alias':'native','evaluations':0,'ledger':[{'trial':'candidate1','alias':'EN_EXACT:candidate1','objective':{'J':1.,'seconds':2}}]})
            report(out,root/'report')
            with (root/'report/costs.csv').open() as handle:costs=list(csv.DictReader(handle))
            actual=[r for r in costs if r['component']=='native_fit']
            self.assertEqual(len(actual),1)
            standalone=[r for r in costs if r['scope']=='STANDALONE_RECONSTRUCTED_CORE' and r['arm']=='EN_ADAPT'][0]
            self.assertEqual(float(standalone['native_seconds']),10.)
            self.assertEqual(float(standalone['seconds']),22.)
            text=(root/'report/report.md').read_text()
            self.assertIn('SEARCH_LIMIT_NO_ACCEPTED_CANDIDATE',text)
            self.assertIn('EN_ADAPT | SEARCH_LIMIT_NO_ACCEPTED_CANDIDATE | native | N4',text)

    def test_explicit_allocation_receipt_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'output';out.mkdir()
            write(root,'accounting.json',{'job_id':'123','allocation_gpu_seconds':1800,'state':'COMPLETED'})
            report(out,root/'report')
            with (root/'report/costs.csv').open() as handle:costs=list(csv.DictReader(handle))
            allocation=[r for r in costs if r['scope']=='ACTUAL_ALLOCATION'][0]
            self.assertEqual(float(allocation['allocation_gpu_hours']),.5)
            self.assertEqual(allocation['job_id'],'123')


if __name__=='__main__':unittest.main()
