import json
from types import SimpleNamespace
import unittest
from project.run_scripts.multilayer_joint_compensation.native_baselines.binding import validate_hparams,execute,SOURCE,verify_source


class TestNativeContracts(unittest.TestCase):
    def hp(self,method):
        name='AlphaEdit' if method=='ALPHAEDIT_NATIVE' else 'MEMIT'
        d=json.loads((SOURCE/f'hparams/{name}/Llama3-8B.json').read_text())
        d.setdefault('blue',False);d.setdefault('edit_layer',-1)
        return SimpleNamespace(**d)

    def test_source_exact_and_native_settings(self):
        manifest=verify_source();self.assertEqual(manifest['member_count'],25)
        for method in ('ALPHAEDIT_NATIVE','MEMIT_NATIVE'):
            hp=self.hp(method);validate_hparams(method,hp)
            hp.blue=True
            with self.assertRaisesRegex(ValueError,'CONTRACT'):validate_hparams(method,hp)

    def test_b0_no_model_or_history_access(self):
        for method in ('ALPHAEDIT_NATIVE','MEMIT_NATIVE'):
            result=execute(None,self.hp(method),method,None,None,[],None,None)
            self.assertEqual(result['native_entrypoint_calls'],0);self.assertEqual(result['history_append'],0)


if __name__=='__main__':unittest.main()
