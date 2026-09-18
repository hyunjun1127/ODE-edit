import copy
from dataclasses import replace
import unittest
from .config import Scope, ScopeError, runtime_policy, validate_runtime_policy
from .preparation import storage_plan, GIB
from .parity import compare_exact


class Boundaries(unittest.TestCase):
    def test_held_source_args_exact(self):
        from .admission import inspect_held
        text='''UserId=janghj(1) JobState=PENDING Reason=JobHeldUser NumCPUs=8 Requeue=0
ReqNodeList=server4 Dependency=(null) TimeLimit=1-00:00:00 ReqTRES=cpu=8,mem=59G,node=1,gres/gpu=1
TresPerNode=gres/gpu:rtx_pro_6000:1
Command=/source/run.sbatch
SubmitLine=sbatch --hold /source/run.sbatch /source /attempt/lock.json GENERATED_REFERENCE_PREPARATION
'''
        command=['/source/run.sbatch','/source','/attempt/lock.json','GENERATED_REFERENCE_PREPARATION']
        inspect_held(text,command)
        for old,new in (('/attempt/lock.json','/wrong/lock.json'),('server4','server2'),
                        ('rtx_pro_6000:1','rtx_pro_6000:2'),('GENERATED_REFERENCE_PREPARATION','MATCHED_B1'),
                        ('server4','server40'),('NumCPUs=8','NumCPUs=80'),('gres/gpu=1','gres/gpu=10'),
                        ('rtx_pro_6000:1','rtx_pro_6000:10')):
            with self.subTest(old=old),self.assertRaises(ValueError):inspect_held(text.replace(old,new),command)

    def test_lock_tamper_and_B2_fail_closed(self):
        from dataclasses import asdict
        from .model import require_lock
        from project.run_scripts.single_layer_edit_preserving_correction.common import digest
        lock=dict(scope=asdict(Scope()),runtime_policy=runtime_policy(),stage='GENERATED_REFERENCE_PREPARATION',
            resources=dict(GPU=1,CPU=8,mem_MiB=60416,wall_hours=24,node='server4',export='NONE',requeue=0,GPUhour_hardcap=None),
            sequential_authorized=False,auto_continue=False,
            output='/data/janghj/ODE-edit/local/en-execution-reuse/20260919-v1/PREP/test/output')
        lock['lock_identity']=digest(lock)
        require_lock(lock)
        for field,value in (('sequential_authorized',True),('auto_continue',True),('output','/tmp/output')):
            changed=copy.deepcopy(lock);changed[field]=value
            with self.assertRaises(ValueError):require_lock(changed)
            changed['lock_identity']=digest({k:v for k,v in changed.items() if k!='lock_identity'})
            with self.assertRaises(ValueError):require_lock(changed)

    def test_single_batch(self):
        Scope().require_batch(0)
        for x in (-1, 1, 9, True):
            with self.assertRaises(ScopeError):
                Scope().require_batch(x)
        with self.assertRaises(ScopeError):
            Scope().require_next()

    def test_scope_no_expansion(self):
        for field, value in (('max_batches', 10), ('sequential_authorized', True),
                             ('auto_continue', True), ('task_gpu_cap', 2),
                             ('method_gradient_shared', True), ('trial_cap', 2),
                             ('reference_documents', 64), ('max_new_tokens', 16),
                             ('mem_mib', 65536), ('arms', ('EN-F',))):
            with self.subTest(field=field), self.assertRaises(ScopeError):
                replace(Scope(), **{field: value}).validate()

    def test_policy_no_waiver_or_optional_optimization(self):
        validate_runtime_policy(runtime_policy())
        for key, value in (('prior_T_skip_inherited', True), ('storage_waiver_inherited', True),
                           ('generation_KV_optimized', True), ('head_chunking_optimized', True),
                           ('backtrack', 0.25), ('armijo', float('nan')),
                           ('current_individual_nll_allowance', .05), ('transformers', '4.57.1')):
            policy = runtime_policy()
            policy[key] = value
            with self.subTest(key=key), self.assertRaises(ScopeError):
                validate_runtime_policy(policy)

    def test_payload_matches_design(self):
        plan = storage_plan()
        parts = plan['components_bytes']
        self.assertEqual(parts['teacher_FP32']/GIB, 78.28125)
        self.assertEqual(parts['reference_key_residual_FP32']/GIB, 16.875)
        self.assertEqual(plan['payload_teacher_key_bytes']/GIB, 95.15625)
        self.assertGreater(plan['estimated_bytes_without_unrecorded_overhead'], plan['payload_teacher_key_bytes'])
        self.assertFalse(plan['reservation'])

    def test_actual_lengths_require_full_population(self):
        for lengths in ([1]*639, [0]*640, [257]*640, [True]*640):
            with self.assertRaises(ValueError):
                storage_plan(lengths)
        plan = storage_plan([1]*640)
        self.assertEqual(plan['actual_positions'], 640)
        self.assertEqual(plan['components_bytes']['reference_key_residual_FP32'], 640*129*(14336+4096)*4)

    @staticmethod
    def result():
        return dict(data_id='EN-R512-G256-v1', input_sha256='input', native_sha256='native',
                    geometry_sha256='geometry', current_binding_sha256='current', teacher_sha256='teacher',
                    model_epoch='W0-own-WN', loss=.1, gradient_sha256='G', projected_gradient_sha256='H',
                    chi=.01, eta0=10., reference_document_ids=list(range(512)),
                    reference_position_counts=[256]*511+[3],
                    trials=[dict(trial=0, weight_sha256='trial', loss=.09, actual_p=-.02,
                                 armijo=True, guard={'accepted': True}, invariant={'accepted': True}, decision='ACCEPT')],
                    selected_sha256='trial', stop_reason='ACCEPT', history_appends=1,
                    gradient_sweeps=1, method_gradient_shared=False,
                    full_sweep_rows=[dict(gradient=True,rows_sha256='exact512rows',coverage='fixture')])

    def test_exact_parity(self):
        a = self.result()
        self.assertEqual(compare_exact(a, copy.deepcopy(a))['status'], 'EXACT_RECEIPT_MATCH')

    def test_no_protection_epsilon_for_parity(self):
        a = self.result()
        b = copy.deepcopy(a)
        b['trials'][0]['loss'] += 1e-12
        self.assertEqual(compare_exact(a, b)['status'], 'EXACTNESS_NOT_ESTABLISHED')

    def test_same_mean_different_document_rows_fails(self):
        a=self.result();b=copy.deepcopy(a);b['full_sweep_rows'][0]['rows_sha256']='different'
        self.assertEqual(compare_exact(a,b)['status'],'EXACTNESS_NOT_ESTABLISHED')

    def test_normal_empty_space_is_not_fabricated_full_sweep(self):
        a=self.result();a.update(stop_reason='REPAIR_SPACE_EMPTY',gradient_sweeps=0,gradient_sha256=None,
            selected_sha256=a['native_sha256'],trials=[],full_sweep_rows=[])
        result=compare_exact(a,copy.deepcopy(a))
        self.assertEqual(result['status'],'EXACT_RECEIPT_MATCH')
        self.assertEqual(result['gradient_coverage'],'NOT_RUN_NORMAL_EMPTY_OR_UNRESOLVED_SPACE')

    def test_receipt_cannot_hide_missing_coverage_or_shared_gradient(self):
        a = self.result()
        for edit in ('missing', 'shared', 'coverage', 'nonfinite', 'selection'):
            b = copy.deepcopy(a)
            if edit == 'missing': del b['trials'][0]['invariant']
            if edit == 'shared': b['method_gradient_shared'] = True
            if edit == 'coverage': b['reference_position_counts'][-1] = 0
            if edit == 'nonfinite': b['loss'] = float('inf')
            if edit == 'selection': b['selected_sha256'] = 'different'
            with self.subTest(edit=edit):
                self.assertEqual(compare_exact(a, b)['status'], 'EXACTNESS_NOT_ESTABLISHED')


if __name__ == '__main__':
    unittest.main()
