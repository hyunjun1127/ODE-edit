"""CPU-only scheduler transcript tests. Never invokes Slurm."""
from pathlib import Path
import unittest
from .submit_program import inspection,other_capacity
from .program import require_program


class SubmissionTests(unittest.TestCase):
    def transcript(self, argv=None):
        source=Path('/task/source');lock=Path('/task/execution.lock.json')
        command=['sbatch',str(source/'project/run_scripts/single_layer_mechanism_first/run.sbatch'),str(source),str(lock)]
        text=('UserId=janghj(1001) JobName=odeedit_slmf_S10_s4 JobState=PENDING Reason=JobHeldUser '
              'TRES=cpu=8,mem=59G,gres/gpu=1 NumCPUs=8 ReqNodeList=server4 Requeue=0 '
              'WorkDir=/task/source Dependency=afterok:50974(unfulfilled) Command='+
              ' '.join(argv or command[-3:])+' StdOut=/task/log.out')
        return text,command,source,lock

    def test_valid_inspection(self):
        self.assertTrue(all(inspection(*self.transcript()).values()))

    def test_wrong_actual_argv_rejected(self):
        args=self.transcript();bad=args[0].replace('Command=/task/source/', 'Command=/wrong/')
        self.assertFalse(inspection(bad,*args[1:])['actual_full_argv'])
        bad=args[0].replace('/task/execution.lock.json','/wrong.lock')
        self.assertFalse(inspection(bad,*args[1:])['actual_full_argv'])

    def test_missing_actual_command_rejected(self):
        args=self.transcript();self.assertFalse(inspection(args[0].split(' Command=')[0],*args[1:])['actual_full_argv'])

    def test_resource_dependency_nonoverlap(self):
        result=other_capacity('50974|prior|RUNNING|gpu:1\n60000|other|PENDING|gpu:1')
        self.assertEqual(sum(r['GPU'] for r in result),1)

    def test_compressed_array_not_one_gpu(self):
        with self.assertRaisesRegex(RuntimeError,'ARRAY_CAPACITY'):
            other_capacity('60000_[0-9%2]|other|PENDING|gpu:1')

    def test_unknown_gres_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError,'GPU_CAPACITY'):
            other_capacity('60000|other|RUNNING|N/A')

    def test_typed_gpu_resource(self):
        self.assertEqual(other_capacity('60000_2|other|RUNNING|gpu:rtx6000:1')[0]['GPU'],1)

    def test_no_checkpoint_monitoring_gate_closed(self):
        lock=dict(phase='GATED_PROGRAM',maximum_batch=10,scientific_gates_required=True,
                  scheduler_writes_in_program=False,agent_monitoring_after_release=False,
                  disk_state_checkpoints=False,save_checkpoints=False)
        require_program(lock)
        for key in ('save_checkpoints','disk_state_checkpoints','agent_monitoring_after_release','scheduler_writes_in_program'):
            bad=dict(lock);bad[key]=True
            with self.assertRaises(ValueError):require_program(bad)
        for key,value in [('maximum_batch',11),('scientific_gates_required',False)]:
            bad=dict(lock);bad[key]=value
            with self.assertRaises(ValueError):require_program(bad)


if __name__=='__main__':unittest.main()
