import copy
import unittest
from ..contracts import digest
from ..track_b.analyze_partial import validate_rows,reduce_rows


class PartialAnalysis(unittest.TestCase):
    def fixture(self):
        records=[dict(case_id=7,requested_rewrite=dict(prompt='{} is',subject='X',
            target_new={'str':'N'},target_true={'str':'T'}),paraphrase_prompts=['X was'],neighborhood_prompts=['Y is'])]
        panel={'panels':{'Current100':[0]}}
        rows=[]
        for metric,prompt,nn,tn in [('RS','X is',1.,2.),('PS','X was',2.,2.),('NS','Y is',3.,1.)]:
            rows.append(dict(panel='Current100',metric=metric,case_id=7,prompt_index=0,
                identity=digest([7,0,prompt,'N','T']),new_nll=nn,true_nll=tn,
                success=tn<nn if metric=='NS' else nn<tn,margin=tn-nn,
                new_token_correct=1,new_token_count=2,true_token_correct=2,true_token_count=2,
                new_strict=False,true_strict=True))
        return dict(rows=rows,pairs=3,panel_identity=digest(panel)),panel,records

    def test_pair_definition_tie_and_secondary_denominators(self):
        data,panel,records=self.fixture()
        rows=validate_rows(data,panel,records)
        values=reduce_rows('B',rows)
        self.assertEqual([v['numerator'] for v in values],[1,0,1])
        self.assertEqual([v['ties'] for v in values],[0,1,0])
        self.assertEqual([v['token_numerator'] for v in values],[1,1,2])
        self.assertEqual([v['token_denominator'] for v in values],[2,2,2])

    def test_order_nonfinite_success_corruption_rejected(self):
        for mutate in (lambda d:d['rows'].reverse(),
                       lambda d:d['rows'][0].update(new_nll=float('nan')),
                       lambda d:d['rows'][1].update(success=True),
                       lambda d:d['rows'][0].update(identity='wrong')):
            data,panel,records=self.fixture();mutate(data)
            with self.assertRaises(ValueError):validate_rows(data,panel,records)

if __name__=='__main__':unittest.main()
