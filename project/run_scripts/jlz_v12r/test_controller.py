import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from . import batches,digest,write,state
from .profile import ARMS,arm_profile,history_expected,resource_order
from .run import BatchTransaction,ready,ram_copy,ram_identity,diagnostic,shared_source_guard
from .submit import arguments,launcher,resource_inventory

class TinyAdapter:
    def __init__(self):
        self.weights={4:torch.tensor([[1.,2.]])}
        self.last_virtual={'request':{'hidden':torch.tensor([3.])}}
        self.capture_virtual=True
    def guard(self):return 'unchanged-nonselected'
    def hook_signature(self):return ('native-hook',)

class ControllerFixtures(unittest.TestCase):
    def test_profiles_cold_scope_anchor8_and420history(self):
        p={a:arm_profile({'value':[]},a) for a in ARMS}
        self.assertEqual(p['L4-ONLY']['eligible_layers'],[4]);self.assertEqual(p['L4-ONLY']['anchor_layer'],8)
        self.assertEqual(sum(history_expected(a) for a in ARMS),420)
        self.assertEqual([p[a]['n_exp'] for a in ARMS],[4,4,0,4,0])
        self.assertTrue(p['BLIND']['blind']);self.assertFalse(p['MAIN']['blind'])
        p['MAIN']['value'].append(1);self.assertEqual(p['BLIND']['value'],[])
    def test_twenty_batches_seenprefix_no21(self):
        rows=list(range(2000));parts=list(batches(rows));self.assertEqual(len(parts),20)
        self.assertEqual([n for n,_,_ in parts],list(range(1,21)))
        self.assertEqual(parts[-1][1],rows[1900:]);self.assertEqual(parts[4][2],rows[:500])
    def test_resource_graph_cap2_or_serial(self):
        for cap in (1,2):
            graph=resource_order(cap);nodes=['qualification',*ARMS]
            for mask in range(1<<len(nodes)):
                done={n for i,n in enumerate(nodes) if mask&(1<<i)}
                if any(not set(graph[n])<=done for n in done):continue
                available=[n for n in nodes if n not in done and set(graph[n])<=done]
                self.assertLessEqual(len(available),cap)
            self.assertEqual(set(graph['collector']),set(nodes))
        self.assertEqual(resource_order(2)['MAIN'],['qualification'])
        self.assertEqual(resource_order(2)['NO-EXPAND'],['BLIND','L4-ONLY','BASE-2X'])
    def test_deep_cache_context_cursor_weight_history_rollback(self):
        a=TinyAdapter();H={4:torch.zeros(2,2)};bench=SimpleNamespace(contexts={'x':['original']});cursor=[1]
        before=state(a,H);cache=ram_identity(a.last_virtual)
        with self.assertRaisesRegex(ValueError,'IO_failure'):
            with BatchTransaction(a,H,bench,cursor) as tx:
                a.weights[4].add_(8);H[4].add_(4);a.last_virtual['request']['hidden'].add_(7)
                bench.contexts['x'].append('mutated');cursor.append(2)
                raise ValueError('IO_failure')
        self.assertTrue(tx.rollback_verified);self.assertEqual(state(a,H),before)
        self.assertEqual(ram_identity(a.last_virtual),cache);self.assertEqual(cursor,[1]);self.assertEqual(bench.contexts,{'x':['original']})
    def test_finish_own_nextentry_and_atomic_failure_rollback(self):
        a=TinyAdapter();H={4:torch.zeros(2,2)};bench=SimpleNamespace(contexts={});cursor=[]
        with BatchTransaction(a,H,bench,cursor) as tx:a.weights[4].add_(1);H[4].add_(1);cursor.append(100);tx.finish()
        prefix=state(a,H)
        with self.assertRaisesRegex(OSError,'atomic'):
            with BatchTransaction(a,H,bench,cursor) as tx:
                a.weights[4].add_(1);H[4].add_(1);cursor.append(200);tx.finish();tx.done=False;raise OSError('atomic')
        self.assertEqual(state(a,H),prefix);self.assertEqual(cursor,[100])
    def test_ready_profile_source_binding_before_setup(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);c={'arm_profiles':{a:arm_profile({},a) for a in ARMS}};lock={'source_commit':'frozen','config_sha256':'config'}
            value={'status':'TECHNICAL_READY','source':'frozen','config_sha256':'config','profiles_sha256':digest(c['arm_profiles'])}
            write(path/'qualification/READY.json',value);self.assertEqual(ready(path,c,lock),value)
            write(path/'qualification/SHARED_TECHNICAL_BLOCK.json',{'status':'SHARED_SOURCE_TECHNICAL_BLOCK'})
            with self.assertRaisesRegex(RuntimeError,'SHARED_SOURCE'):ready(path,c,lock)
            (path/'qualification/SHARED_TECHNICAL_BLOCK.json').unlink()
            lock['source_commit']='different'
            with self.assertRaisesRegex(RuntimeError,'EXACT_READY'):ready(path,c,lock)
    def test_explicit_memory_export_threads_and_cpu0gpu(self):
        resources=dict(cpu=8,collector_cpu=8,host_mib=59392,collector_host_mib=24576,wall='2-00:00:00',qualification_wall='04:00:00',collector_wall='04:00:00')
        for role in ('qualification',*ARMS,'collector'):
            args=arguments(role,'afterany:123',Path('/tmp/task'),resources)
            self.assertIn('--export=NONE',args);self.assertIn('--no-requeue',args);self.assertIn('--cpus-per-task=8',args)
            self.assertIn('--mem='+('24576' if role=='collector' else '59392')+'M',args)
            self.assertEqual('--gres=gpu:1' in args,role!='collector')
            script=launcher(Path('/tmp/source'),'source',role,Path('/tmp/task'),8)
            self.assertIn('JLZ_V12R_SOURCE_COMMIT=source',script);self.assertIn('OMP_NUM_THREADS=8',script)
            self.assertNotIn('odeedit',script)
    def test_implicit_pending_resource_is_counted(self):
        owner=__import__('getpass').getuser()
        def command(argv,cwd=None):
            if argv[0]=='squeue':
                self.assertNotIn('-w',argv)
                return f'1|{owner}|owned|PENDING|gpu:1|Resources\n2|{owner}|elsewhere|RUNNING|gpu:1|server3'
            return ('ReqTRES=cpu=8,gres/gpu=1 ReqNodeList=(null) NodeList=(null) ' if argv[3]=='1' else
                    'ReqTRES=cpu=8,gres/gpu=1 ReqNodeList=server3 NodeList=server3 ')
        with patch('project.run_scripts.jlz_v12r.submit.command',command):jobs=resource_inventory()['jobs']
        self.assertEqual([j['job'] for j in jobs],['1']);self.assertTrue(jobs[0]['implicit_pending_counted'])
    def test_diagnostic_oom_only_and_generic_error_block(self):
        a=TinyAdapter();H={4:torch.zeros(2,2)};entry={'history_entry':H};engine=SimpleNamespace(calls={'logical_builds':25})
        plan={'engine':engine,'R':{},'built':{'candidate':24},'mask':[]}
        with tempfile.TemporaryDirectory() as d:
            with patch('project.run_scripts.jlz_v12r.qualification.laterdiagnostic',side_effect=torch.cuda.OutOfMemoryError('bounded')):
                diagnostic(a,entry,plan,Path(d)/'oom',[0,1])
            receipt=json.loads((Path(d)/'oom/diagnostic.json').read_text());self.assertEqual(receipt['result']['status'],'TECHNICALLY_UNAVAILABLE')
            self.assertTrue(receipt['state_RNG_hooks_cache_verified'])
            with patch('project.run_scripts.jlz_v12r.qualification.laterdiagnostic',side_effect=RuntimeError('parity')):
                with self.assertRaisesRegex(RuntimeError,'parity'):diagnostic(a,entry,plan,Path(d)/'bad',[0,1])
    def test_running_arm_shared_block_precommit_rollsback_prefix(self):
        a=TinyAdapter();H={4:torch.zeros(2,2)};bench=SimpleNamespace(contexts={});cursor=[100];before=state(a,H)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);shared_source_guard(path)
            with self.assertRaisesRegex(RuntimeError,'SHARED_SOURCE'):
                with BatchTransaction(a,H,bench,cursor) as tx:
                    a.weights[4].add_(1);H[4].add_(1);cursor.append(200)
                    write(path/'qualification/SHARED_TECHNICAL_BLOCK.json',{'status':'SHARED_SOURCE_TECHNICAL_BLOCK'})
                    shared_source_guard(path);tx.finish()
            self.assertEqual(state(a,H),before);self.assertEqual(cursor,[100]);self.assertTrue(tx.rollback_verified)

if __name__=='__main__':unittest.main()
