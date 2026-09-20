import json,shlex,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from . import freeze_server3_repair as module

class SubmitTests(unittest.TestCase):
    def test_release_is_last_scheduler_operation(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d);source=a/'source';source.mkdir();archive=a/'source.tar.gz';archive.write_bytes(b'x')
            lock=dict(execution=dict(source=str(source),commit='f'*40,members=[],archive=dict(path=str(archive),sha256=module.sha(archive))),input_seals={},output=str(a/'output'))
            (a/'execution.lock.json').write_text(json.dumps(lock));calls=[];submitted=[]
            def query(argv,**kw):
                calls.append(argv)
                if argv[0]=='squeue':return ''
                if argv[0]=='sbatch':submitted.extend(argv);return '12345\n'
                if argv==['scontrol','show','job','12345','--oneliner']:
                    return ('UserId=janghj(1025) JobName=odeedit_en_adapt_B300_s3_repair Reason=JobHeldUser '
                        'gres/gpu=1 NumCPUs=8 mem=119G ReqNodeList=ubuntu Requeue=0 Dependency=(null) '
                        'Partition=gpu TimeLimit=1-00:00:00 Command='+str(source/'project/run_scripts/en_adaptive_nullspace/run-server3-repair.sbatch')+
                        ' WorkDir='+str(source)+' SubmitLine='+shlex.join(submitted))
                self.fail('unexpected scheduler query '+str(argv))
            def run(argv,**kw):
                calls.append(argv);self.assertEqual(argv,['scontrol','release','12345'])
                return SimpleNamespace(returncode=0,stdout='',stderr='')
            with patch.object(module.subprocess,'check_output',side_effect=query),patch.object(module.subprocess,'run',side_effect=run),patch.object(module.os,'statvfs',return_value=SimpleNamespace(f_bavail=20*2**30,f_frsize=1)):
                module.submit(a)
            self.assertEqual([x[0] for x in calls],['squeue','sbatch','scontrol','scontrol'])
            self.assertEqual(calls[-1],['scontrol','release','12345'])
            r=json.loads((a/'release.json').read_text());self.assertFalse(r['monitoring'])
            self.assertEqual(r['actual_initial'],'NOT_OBSERVED')
            self.assertEqual(r['status'],'MONITORING_PAUSED_AWAITING_USER')

if __name__=='__main__':unittest.main()
