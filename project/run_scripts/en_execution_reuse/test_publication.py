"""CPU-only accounting/publication fixtures; no Slurm calls or scientific raw."""
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import report
from .accounting import parse
from .report import table
from .config import ARMS
from .preparation import create_json,member
from .publication_checks import check as publication_check
from .test_generated_teacher import fixture as teacher_fixture
from .test_reducer import fixture as reducer_fixture


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

    def test_complete_synthetic_publication_consumes_actual_schema_not_gpu(self):
        with tempfile.TemporaryDirectory(prefix='en-report-fixture-') as td:
            base=Path(td);repo=base/'repo';repo.mkdir();raw=base/'attempt/output';raw.mkdir(parents=True)
            generated=base/'prep/generated';generated.mkdir(parents=True)
            _,_,manifest=teacher_fixture(generated)
            ready=create_json(base/'prep/READY.json',dict(manifest=member(generated/'manifest.json'),
                seconds=10.,source={'commit':'PREP_CPU_FIXTURE'}))
            execution=dict(commit='B1_CPU_FIXTURE',tree='CPU_FIXTURE',
                source_root=str(Path(__file__).resolve().parents[3]))
            lock=dict(output=str(raw),execution=execution,dataset_root='CPU_FIXTURE',generated_ready=ready,
                native_reuse_lineage={'prior_seconds':3.})
            lock_path=base/'attempt/execution.lock.json';create_json(lock_path,lock)
            create_json(base/'attempt/submission.json',dict(job='2'))
            records,canonical=reducer_fixture()
            for arm in ('W0','N4',*ARMS):create_json(raw/'observers'/f'{arm}.json',canonical)
            devrows=[dict(source_row_id=f'dev-{i}',role='Dev128',loss=.1,scored_positions=1) for i in range(128)]
            devloss=sum(r['loss'] for r in devrows)/128
            for arm in ('N4',*ARMS):
                create_json(raw/'observers'/f'{arm}-Dev128.json',dict(loss=devloss,rows=devrows,receipt={'coverage':{'complete':True}}))
            for arm in ARMS:
                create_json(raw/'arms'/arm/'selection-ledger.json',dict(trials=[]))
                create_json(raw/'arms'/arm/'execution-exactness.json',dict(trials=[],full_sweep_rows=[]))
            create_json(raw/'matched-exactness.json',dict(status='CPU_FIXTURE_ONLY'))
            create_json(raw/'technical/checks.json',dict(pass_=True))
            create_json(raw/'geometry/EN-F.json',dict(status='CPU_FIXTURE_ONLY',dimension=1))
            create_json(raw/'artifact-manifest.json',dict(members=[]))
            work={arm:dict(wall_seconds=1.,schedule_entry_pre_timing_seconds=.1,
                optimizer={'gradient_sweeps':0,'objective_trial_sweeps':0,'attempted_trial_slots':0},
                current={},external_current_weight={},session=None) for arm in ARMS}
            create_json(raw/'terminal.json',dict(status='B1_COMPLETE',max_batches=1,sequential_authorized=False,
                arm_work=work,setup_timing={'model_load_seconds':.1},total_program_seconds=5.,
                peak_gpu_allocated=0,peak_gpu_reserved=0,peak_host_KiB=1))
            accounting=parse(serialize(accounting_fixture()),{'1':'PREP','2':'B1'})
            scheduler=base/'accounting.json';create_json(scheduler,accounting)
            with patch.object(report,'ROOT',base),patch.object(report,'audit',return_value={'logical_bytes':10}), \
                 patch('scripts.fixed_counterfact.load_prefix',return_value=records), \
                 patch.object(report.subprocess,'check_output',return_value='CPU_FIXTURE_ANALYSIS'):
                directory=report.run(lock_path,repo,scheduler,base/'review-local')
            text=(directory/'diagnostic-report-ko.md').read_text()
            self.assertIn('CPU_FIXTURE_ONLY',text);self.assertIn('1000/1000 (100.000%)',text)
            self.assertEqual(len(list(csv.DictReader(io.StringIO((directory/'final-table.csv').read_text())))),12)
            self.assertEqual(len(list(csv.DictReader(io.StringIO((directory/'source-conformance.csv').read_text())))),11)
            self.assertTrue((directory/'rooted-receipt.json').is_file())
            checks=json.loads((directory/'publication-checks.json').read_text())
            self.assertEqual(checks['GFM_tables'],2)
            original={p.name:p.read_bytes() for p in directory.iterdir()}
            with patch.object(report,'ROOT',base),self.assertRaises(FileExistsError):
                report.run(lock_path,repo,scheduler,base/'review-local')
            self.assertEqual(original,{p.name:p.read_bytes() for p in directory.iterdir()})

    def test_missing_link_and_bad_gfm_are_not_renderer_pass(self):
        with tempfile.TemporaryDirectory(prefix='en-render-fixture-') as td:
            root=Path(td);p=root/'diagnostic-report-ko.md'
            p.write_text('| a | b |\n|---|---|\n|1|2|3|\n[missing](missing.csv)\n')
            with self.assertRaisesRegex(ValueError,'WIDTH'):publication_check(root)
            p.write_text('| a | b |\n|---|---|\n|1|2|\n[missing](missing.csv)\n')
            with self.assertRaisesRegex(ValueError,'MISSING'):publication_check(root)


if __name__=='__main__':unittest.main()
