"""CPU miniature file-inventory fixtures, not real-model validation."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from .common import sha256,write_json,stat_identity,tensor_sha
import torch
from . import model_runtime as rt


class AssetBindingTests(unittest.TestCase):
    def test_projector_hash_is_original_selected_stack_not_matrix(self):
        stack=torch.arange(45,dtype=torch.float32).reshape(5,3,3)
        contract={'identity':{'projector_selected_sha256':tensor_sha(stack[:1])}}
        with patch.object(rt,'CONTRACT',contract):
            selected=rt.selected_projector(stack)
            self.assertTrue(torch.equal(selected,stack[0]))
            self.assertNotEqual(tensor_sha(selected),contract['identity']['projector_selected_sha256'])
            changed=stack.clone();changed[0,0,0]+=1
            with self.assertRaisesRegex(AssertionError,'P4_SINGLETON_STACK_IDENTITY'):
                rt.selected_projector(changed)
    def fixture(self,root):
        root=Path(root);snapshot=root/'snapshot';snapshot.mkdir()
        weight=snapshot/'model.safetensors';weight.write_bytes(b'synthetic file-hash fixture only')
        lock=root/'execution.json'
        write_json(lock,dict(snapshot='/old/model',members=[dict(path='/old/model/model.safetensors',
            bytes=weight.stat().st_size,sha256=sha256(weight))]))
        return root,snapshot,weight,lock

    def test_complete_hash_then_current_stat_receipt_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            root,snapshot,weight,lock=self.fixture(folder)
            with patch.object(rt,'ATTEMPT',root),patch.object(rt,'CONTRACT',{'paths':{'model_snapshot':str(snapshot)}}),patch.object(rt,'mapped',return_value=lock):
                first=rt.verify_model_assets()
                self.assertEqual(first['status'],'PASS');self.assertFalse(first['members'][0]['reused_stat_bound_receipt'])
                write_json(root/'inputs/model-asset-verification.json',first)
                second=rt.verify_model_assets()
                self.assertTrue(second['members'][0]['reused_stat_bound_receipt'])

    def test_same_size_changed_bytes_fail_without_valid_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            root,snapshot,weight,lock=self.fixture(folder)
            weight.write_bytes(b'x'*weight.stat().st_size)
            with patch.object(rt,'ATTEMPT',root),patch.object(rt,'CONTRACT',{'paths':{'model_snapshot':str(snapshot)}}),patch.object(rt,'mapped',return_value=lock):
                with self.assertRaisesRegex(AssertionError,'MODEL_ASSET_HASH_MISMATCH'):
                    rt.verify_model_assets()

    def test_checkpoint_stat_change_blocks_before_tensor_load(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cp=root/'B001/W-method-state.pt';cp.parent.mkdir();cp.write_bytes(b'x')
            identity=stat_identity(cp);identity.update(full_sha256=True,unchanged_during_hash=True)
            write_json(root/'results/geometry/checkpoint-input-verification.json',[identity])
            cp.write_bytes(b'changed')
            with patch.object(rt,'ATTEMPT',root),patch.object(rt,'CPROOT',root),patch.object(rt.torch,'load') as load:
                with self.assertRaisesRegex(AssertionError,'CHECKPOINT_STAT_DRIFT'):rt.checkpoint(1)
                load.assert_not_called()


if __name__=='__main__':unittest.main()
