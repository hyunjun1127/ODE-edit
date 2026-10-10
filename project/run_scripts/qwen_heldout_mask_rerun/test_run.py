import unittest
from pathlib import Path
from .run import tracking_config
from official.tracking.schema import config, metrics
from official.runners.server4.qwen_run import _factual_scalars
from official.evaluation.reduce import counterfact

class Mapping(unittest.TestCase):
    def test_current_and_final500(self):
        c=tracking_config(dict(method='MEMIT',source='a'*40,config_sha256='b'*64,stream_sha256='c'*64))
        config(c)
        case={k+'_prompts_probs':[dict(target_true=2.,target_new=1.)]
              for k in ('rewrite','paraphrase','neighborhood')}
        for n,prefix,edits in [(100,'current/pre',100),(100,'current/post',100),(500,'all_seen/post',500)]:
            cases=[case]*n;v=_factual_scalars('cf',cases,counterfact(cases),prefix,edits)
            if prefix=='current/pre':v['pre_state_edits']=edits-100
            metrics(v,scientific=True,config_values=c)
    def test_baseline_context_only(self):
        s=Path(__file__).with_name('run.py').read_text()
        self.assertIn('native.module.get_context_templates(model,tok)',s)
        self.assertNotIn('native.restore_context(',s)
        self.assertNotIn('W0_first2000',s)
        self.assertNotIn('qwen_price_hparam_tier2',s)

if __name__=='__main__':unittest.main()
