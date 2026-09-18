"""CPU-only accounting/publication fixtures; no Slurm calls or scientific raw."""
import copy
import csv
import io
import unittest
from .accounting import parse
from .report import table


def accounting_fixture():
    rows=[dict(JobIDRaw='1',JobName='odeedit_en_reuse_g256_prep_s4',User='janghj',State='COMPLETED',ExitCode='0:0',
               ElapsedRaw='3600',AllocTRES='cpu=8,gres/gpu=1,mem=59G',Start='2026-09-19T01:00:00',
               End='2026-09-19T02:00:00',NodeList='server4'),
          dict(JobIDRaw='2',JobName='odeedit_en_reuse_g256_B1_s4',User='janghj',State='COMPLETED',ExitCode='0:0',
               ElapsedRaw='1800',AllocTRES='cpu=8,gres/gpu=1,mem=59G',Start='2026-09-19T02:30:00',
               End='2026-09-19T03:00:00',NodeList='server4')]
    return rows


def serialize(rows):
    value=io.StringIO();writer=csv.DictWriter(value,fieldnames=list(rows[0]),delimiter='|')
    writer.writeheader();writer.writerows(rows);return value.getvalue()


class PublicationTests(unittest.TestCase):
    def test_exact_parent_cost_no_nested_step_charge(self):
        result=parse(serialize(accounting_fixture()),{'1':'PREP','2':'B1'})
        self.assertEqual(result['allocated_GPU_seconds'],5400)
        self.assertEqual(result['allocated_GPU_hours'],1.5)
        self.assertEqual(result['maximum_task_GPU_overlap'],1)
        self.assertEqual(result['utilization'],'NOT_MEASURED')

    def test_unrelated_steps_wrong_mapping_failed_resource_or_overlap_rejected(self):
        for change in ('duplicate','step','owner','name','failed','memory','GPU','elapsed','overlap'):
            rows=accounting_fixture()
            if change=='duplicate':rows.append(copy.deepcopy(rows[0]))
            if change=='step':rows[1]['JobIDRaw']='2.batch'
            if change=='owner':rows[1]['User']='other'
            if change=='name':rows[1]['JobName']='odeedit_enfc_S4_s4'
            if change=='failed':rows[1]['State']='FAILED'
            if change=='memory':rows[1]['AllocTRES']='cpu=8,gres/gpu=1,mem=100G'
            if change=='GPU':rows[1]['AllocTRES']='cpu=8,gres/gpu=2,mem=59G'
            if change=='elapsed':rows[1]['ElapsedRaw']='1801'
            if change=='overlap':
                rows[1].update(Start='2026-09-19T01:30:00',End='2026-09-19T02:00:00')
            with self.subTest(change=change),self.assertRaises(ValueError):
                parse(serialize(rows),{'1':'PREP','2':'B1'})

    def test_gfm_width_and_embedded_pipe_escaping(self):
        value=table(['A','B'],[['pipe|value','next\nline']])
        self.assertEqual([r.count('|') for r in value.splitlines()],[3,3,3])
        self.assertIn('pipe&#124;value',value)
        with self.assertRaisesRegex(ValueError,'WIDTH'):table(['A','B'],[['one']])


if __name__=='__main__':unittest.main()
