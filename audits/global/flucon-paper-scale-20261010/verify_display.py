"""Read-only display regression; no remote reads/model execution."""
import hashlib
import json
from pathlib import Path
import sys
from decimal import Decimal, ROUND_HALF_UP

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from official.evaluation.generation.paper_display import paper_cell, paper_generation


def verify():
    raw = json.loads((Path(__file__).parent / 'qwen-raw-recomputation.json').read_text())
    readme = (ROOT / 'README.md').read_text()
    llama = readme.split('### Llama3-8B-Instruct', 1)[1].split('### Qwen', 1)[0]
    qwen = readme.split('### Qwen2.5-7B-Instruct', 1)[1].split('### GPT-J', 1)[0]
    def cells(section, prefix):
        return next(line for line in section.splitlines() if line.startswith(prefix)).split('|')[6:8]
    expected_llama = [paper_cell(6.352242334333923, metric='Flu', raw_unit='bits'),
                      paper_cell(.24636896048599818, metric='Con', raw_unit='cosine_0_to_1')]
    expected_ft = list(paper_generation(raw['endpoints']['W20_FT_61898']['stored_summary']).values())
    assert [s.strip() for s in cells(llama, '| [W0 (')] == expected_llama
    assert [s.strip() for s in cells(qwen, '| FT |')] == expected_ft
    assert readme.count('| CF Flu ×100 | CF Con ×100 |') == 4
    assert [s.strip() for s in cells(qwen, '| [W0 (')] == list(paper_generation(raw['endpoints']['W0']['stored_summary']).values())
    evidence = json.loads((ROOT/'audits/servers/server2/flucon-paper-scale-20261010/table-rows-final.json').read_text())
    names = {'FT':'FT','MEMIT':'MEMIT','ALPHAEDIT':'AlphaEdit','ALPHAEDIT_BLUE':'AlphaEdit-BLUE','MEMIT_FE':'MEMIT-FE','SPHERE':'AlphaEdit+SPHERE'}
    verified = 0
    for row in evidence['rows']:
        if not row.get('numeric_eligible'):
            continue
        section = qwen if row['model']=='qwen25' else readme.split('### GPT-J-6B',1)[1]
        displayed = next(line for line in section.splitlines() if line.startswith('| '+names[row['method']]+' |')).split('|')
        keys = ['Score','Efficacy','Generalization','Specificity'] if row['dataset']=='cf' else ['Efficacy','Generalization','Specificity']
        start = 2 if row['dataset']=='cf' else 8
        for offset, key in enumerate(keys):
            value = row['metrics'][key]
            if row['dataset']=='cf' and key != 'Score':
                denominator = row['denominators'][{'Efficacy':'rewrite','Generalization':'paraphrase','Specificity':'neighborhood'}[key]]
                numerator = round(value*denominator/100)
                assert abs(value-numerator*100/denominator)<1e-10
                number = Decimal(numerator)*100/denominator
            else:
                number = Decimal(str(value))
            assert displayed[start+offset].strip() == str(number.quantize(Decimal('.01'),rounding=ROUND_HALF_UP)), (row['job_id'],key)
        verified += 1
    assert verified == 13
    sh1 = json.loads((ROOT/'audits/servers/server1/flucon-paper-scale-20261010/completed-rows.json').read_text())
    for row in sh1['rows']:
        if row['dataset']!='zsre':
            continue
        displayed = next(line for line in llama.splitlines() if line.startswith('| '+names[row['method']]+' |')).split('|')
        for offset,key in enumerate(['Efficacy','Generalization','Specificity']):
            assert displayed[8+offset].strip()==str(Decimal(str(row['metrics'][key])).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))
    jobs = json.loads((ROOT/'audits/servers/server1/flucon-paper-scale-20261010/table-rows.json').read_text())
    for jid in ('62259','62260','62261','62262','62263'):
        assert readme.count('('+jid+')')==2, jid
    policy = json.loads((ROOT / 'control/main-results-policy.json').read_text())
    assert 'generation_paper_display' in policy
    assert paper_cell('DEFERRED', metric='Flu', raw_unit='bits') == 'DEFERRED'
    return {'corrected_scale_cells': 4, 'new_W0_generation_cells':2, 'SH2_factual_rows_reverified':verified,
            'SH1_zsRE_rows_verified':6, 'SH1_generation_registered_rows':5, 'unit_headers': 4, 'status': 'PASS',
            'README_sha256': hashlib.sha256(readme.encode()).hexdigest(),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'GPU_forwards': 0, 'raw_mutations': 0}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
