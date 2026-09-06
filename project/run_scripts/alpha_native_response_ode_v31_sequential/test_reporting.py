"""Small raw-free report regressions; never launches a model or scheduler."""
import csv,hashlib,json,tempfile,unittest
from pathlib import Path
from .reporting import sha,csvwrite,bits
from .synthesis import physical_rows,ratio
from .recovery_context import seal


class Tests(unittest.TestCase):
    def test_context_cache_recovers_existing_bytes_only(self):
        value=[['{}'],['Existing native context. {}']]
        digest=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);chain=d/'chain';chain.mkdir()
            (chain/'runtime.lock.json').write_text(json.dumps({'contexts_sha256':digest}))
            log=d/'stdout';log.write_text('header\nCached context templates '+repr(value)+'\nmore output\n')
            r=seal(chain,log,d/'context');self.assertEqual(r['contexts_sha256'],digest)
            self.assertEqual(r['additional_generation_count'],0)
            with self.assertRaises(FileExistsError):seal(chain,log,d/'context')

    def test_hash_bytes_and_csv_lf(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.csv';csvwrite(p,[dict(value=1,vector=[1,2])])
            self.assertNotIn(b'\r',p.read_bytes())
            self.assertEqual(sha(p),hashlib.sha256(p.read_bytes()).hexdigest())
            with p.open() as f:self.assertEqual(list(csv.DictReader(f))[0]['value'],'1')
            with self.assertRaises(FileExistsError):csvwrite(p,[])

    def test_official_net_energy_not_velocity(self):
        writer=dict(nodes=[],actual_physical_action=dict(frobenius_net_sq=5.,layers=[
            dict(layer=4,frobenius_sq=1.,native_raw=3.),dict(layer=8,frobenius_sq=4.,native_raw=8.)]))
        out=physical_rows(writer,{'arm':'O_NATIVE'})
        self.assertEqual(out[1]['endpoint_energy_share'],.8)
        self.assertEqual(out[1]['endpoint_DeltaW_norm'],2.)
        self.assertIsNone(out[1]['integral_raw_native_velocity_action_share'])
        self.assertEqual(ratio(0,0),None)

    def test_single_h_action_and_net_distinction(self):
        action=dict(layer=8,raw_native_velocity_action=20.,normalized_native_velocity_action=2.,
            history_velocity_action=12.,L2_velocity_action=8.,frobenius_velocity_squared=4.)
        writer=dict(nodes=[dict(h=.5,active_layers=[8],layer_actions=[action],predicted_target_contribution=[-2.],
            actual_physical=[dict(layer=8,actual_step_DeltaW_squared=1.)])],
            actual_physical_action=dict(frobenius_net_sq=1.,layers=[dict(layer=8,frobenius_sq=1.,native_raw=5.)]))
        row=physical_rows(writer,{})[0]
        self.assertEqual(row['integral_raw_native_velocity_action'],10.)
        self.assertEqual(row['integral_history_velocity_action'],6.)
        self.assertEqual(row['signed_predicted_target_progress'],-1.)
        self.assertEqual(row['actual_step_norm_sum'],1.)

    def test_preference_and_tie_not_token_accuracy(self):
        rows=[]
        for prefix in ('rewrite','locality'):
            for kind in ('new','true'):
                rows.append(dict(case_id=1,prompt_index=0,kind=prefix+'_target_'+kind,nll=1.,input_identity_sha256=kind))
        self.assertFalse(bits({'rows':rows},'rewrite')[(1,0)]['success'])
        self.assertFalse(bits({'rows':rows},'locality')[(1,0)]['success'])


if __name__=='__main__':unittest.main()
