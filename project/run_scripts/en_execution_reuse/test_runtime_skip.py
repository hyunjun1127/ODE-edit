"""CPU regression only: no experiment gates, model load, or CUDA allocation."""
import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from . import matched_runner, generated_oracle
from .transaction import atomic_without_reload


class RuntimeSkipTests(unittest.TestCase):
    def test_production_route_has_no_diagnostic_calls(self):
        tree=ast.parse(Path(matched_runner.__file__).read_text())
        calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)]
        names={node.func.id for node in calls if isinstance(node.func,ast.Name)}
        attrs={node.func.attr for node in calls if isinstance(node.func,ast.Attribute)}
        self.assertTrue(names.isdisjoint({'check','current_physical','compare_exact'}))
        self.assertNotIn('byte_hash_nonselected',attrs)
        store=next(n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='GeneratedTeacherStore')
        self.assertIs(next(k.value.value for k in store.keywords if k.arg=='verify_payloads'),False)
        commit=next(n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='commit')
        self.assertIs(next(k.value.value for k in commit.keywords if k.arg=='verify_reload'),False)

    def test_no_document_gradient_CPU_transfer_and_one_final_transfer_site(self):
        # Static path assertion supplements CPU arithmetic tests, not GPU parity.
        tree=ast.parse(Path(generated_oracle.__file__).read_text())
        evaluate=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_evaluate')
        loop=next(n for n in ast.walk(evaluate) if isinstance(n,ast.For))
        transfers=[n for n in ast.walk(loop) if isinstance(n,ast.Call) and
                   isinstance(n.func,ast.Attribute) and n.func.attr=='cpu']
        self.assertEqual(transfers,[])
        final=[n for n in ast.walk(evaluate) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
               and n.func.attr=='cpu' and isinstance(n.func.value,ast.Name) and n.func.value.id=='accumulation']
        self.assertEqual(len(final),1)
        allocation=next(n for n in ast.walk(evaluate) if isinstance(n,ast.Call) and
                        isinstance(n.func,ast.Attribute) and n.func.attr=='zeros')
        device=next(k.value for k in allocation.keywords if k.arg=='device')
        self.assertEqual(ast.unparse(device),'self.device')

    def test_atomic_save_does_not_reload_or_overwrite(self):
        with tempfile.TemporaryDirectory(prefix='en-skip-save-') as directory:
            path=Path(directory)/'checkpoint.pt'
            payload={'weight':torch.arange(12,dtype=torch.float32).reshape(3,4),'M4':torch.eye(4)}
            with patch.object(torch,'load',side_effect=AssertionError('diagnostic reload')):
                receipt=atomic_without_reload(path,payload)
                original=path.read_bytes()
                self.assertEqual(receipt['bytes'],len(original))
                with self.assertRaises(FileExistsError):atomic_without_reload(path,payload)
                self.assertEqual(path.read_bytes(),original)


if __name__=='__main__':unittest.main()
