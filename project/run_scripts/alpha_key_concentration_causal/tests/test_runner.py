import json,tempfile,unittest
from pathlib import Path
from project.run_scripts.alpha_key_concentration_causal.runner import PHASES,require_ready
from project.run_scripts.alpha_key_concentration_causal.common import save

class RunnerScope(unittest.TestCase):
    def test_no_followup_phase(self):
        self.assertEqual(PHASES,('gate','geometry','writers','reduce'))
        self.assertTrue(not {'SEQ','ORDER','FUTURE'}.intersection(PHASES))
    def test_pending_not_ready(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'ready.json';save(p,{'status':'PENDING'})
            with self.assertRaises(RuntimeError):require_ready(p)
    def test_actual_marker_required(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):require_ready(Path(d)/'missing')
    def test_create_once(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.json';save(p,{'status':'PASS'})
            with self.assertRaises(FileExistsError):save(p,{'status':'CHANGED'})
            self.assertEqual(json.loads(p.read_text())['status'],'PASS')

if __name__=='__main__':unittest.main()
