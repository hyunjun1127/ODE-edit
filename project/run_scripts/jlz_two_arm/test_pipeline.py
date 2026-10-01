"""Bounded model-free regression for the autonomous two-lane DAG."""
import itertools
import unittest
import csv
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET
import re
import shlex
from .submit import specs,dependency_members,job_name
from .common import numerical
from .freeze import shell

class PipelineTests(unittest.TestCase):
    def test_two_lane_width_and_all_main_registered(self):
        ids={};graph={}
        for i in range(6):
            name,gpu,dep,args=specs(ids)[i]
            graph[name]=set(dep.split(':')[1:]) if dep else set()
            ids[name]=str(100+i)
        self.assertEqual(graph['main-A'],{'101','102'})
        self.assertEqual(graph['main-A'],graph['main-B'])
        self.assertEqual(graph['collector'],{'100','101','102','103','104'})
        # Enumerate reachable DAG states, not sum sequential dependency requests.
        states=[frozenset()];maximum=0
        while states:
            done=states.pop();ready=[k for k,d in graph.items() if k!='collector' and ids[k] not in done and d<=done]
            maximum=max(maximum,len(ready))
            if ready:
                states.extend(done|{ids[k]} for k in ready)
        self.assertEqual(maximum,2)

    def test_existing_capacity_precedes_shared_prep(self):
        self.assertEqual(specs({},['71','80_2'])[0][2],'afterany:71:80_2')
        self.assertEqual(dependency_members('afterok:123(unfulfilled),afterok:124(unfulfilled)'),{('afterok','123'),('afterok','124')})
        self.assertEqual(dependency_members('(null)'),set())

    def test_prefix_and_immutable_command(self):
        self.assertTrue(job_name('main-A').startswith('odeedit_jlz_twoarm_s4'))
        self.assertEqual(job_name('collector'),'odeedit_jlz_twoarm_collect_s4')
        text=shell('/tmp/source','project.run_scripts.jlz_two_arm.run',['--phase','main','--arm','A'])
        self.assertIn('set -euo pipefail',text);self.assertIn('HF_HUB_OFFLINE=1',text)
        self.assertNotIn('sbatch',text);self.assertNotIn('sleep',text)

    def test_slurm_submitline_is_full_argv_not_command(self):
        argv=['sbatch','--hold','/tmp/collector.sh','123,124']
        text='Command=/tmp/collector.sh SubmitLine=sbatch --hold /tmp/collector.sh 123,124 WorkDir=/tmp/source'
        self.assertEqual(re.search(r'\bCommand=(.*?)(?= [A-Z][A-Za-z]+=|$)',text)[1],'/tmp/collector.sh')
        self.assertEqual(shlex.split(re.search(r'\bSubmitLine=(.*?)(?= WorkDir=|$)',text)[1]),argv)

    def test_record_only_does_not_mask_nonfinite(self):
        r=numerical({'error':10.},{'error':.001})
        self.assertEqual(r['original_verdict'],'FAIL');self.assertEqual(r['action'],'CONTINUE_WITH_WARNING')
        with self.assertRaisesRegex(RuntimeError,'NONFINITE_DIAGNOSTIC'):numerical({'error':float('nan')},{'error':1.})

    def test_svg_reproducible_empty_and_values(self):
        from .plot import generate
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'endpoint-metrics.csv').write_text('stage,method,kind,panel,batch,preference_rate\nmain,JLZ_A,R,allseen_RP_current_or_milestone_N,1,.5\n')
            (root/'fit-reference-losses.csv').write_text('stage,arm,batch,general_mean\nmain,A,1,.01\n')
            files=generate(root);first=[(root/f).read_bytes() for f in files]
            generate(root);self.assertEqual(first,[(root/f).read_bytes() for f in files])
            for f in files:ET.parse(root/f)

if __name__=='__main__':unittest.main()
