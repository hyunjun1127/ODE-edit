import tempfile
import unittest
from pathlib import Path
from .records import save
from .import_assets import sha
from .saved_generation import validate_selection

class SavedGenerationTests(unittest.TestCase):
    def test_exact_saved_selected_step_and_outcome_blindness(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);snapshot=root/'snapshot-008.pt'
            snapshot.write_bytes(b'CPU identity fixture, not a model')
            selection=root/'selection.json'
            save(selection,dict(entry='Middle',support='C',PS_NS_influence=0,endpoint_path=str(root/'endpoint.pt'),alpha=.02))
            self.assertEqual(validate_selection(snapshot,selection,sha(snapshot))['alpha'],.02)
            with self.assertRaisesRegex(ValueError,'IDENTITY'):
                validate_selection(snapshot,selection,'0'*64)
            with self.assertRaisesRegex(ValueError,'PREDECLARED'):
                validate_selection(root/'snapshot-032.pt',selection,sha(snapshot))

if __name__=='__main__':unittest.main()
