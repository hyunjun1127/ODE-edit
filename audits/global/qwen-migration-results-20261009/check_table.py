"""Read-only check of owner W20 receipts against the GH README integration."""
import csv
import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
README = (ROOT / 'README.md').read_text()
LABEL = {'FT': 'FT', 'MEMIT': 'MEMIT', 'ALPHAEDIT': 'AlphaEdit',
         'ALPHAEDIT_BLUE': 'AlphaEdit-BLUE', 'MEMIT_FE': 'MEMIT-FE',
         'SPHERE': 'AlphaEdit+SPHERE'}


def display(value):
    return str(Decimal(str(value)).quantize(Decimal('.01'), rounding=ROUND_HALF_UP))


def table(model):
    title = {'llama3': 'Llama3-8B-Instruct', 'gptj': 'GPT-J-6B'}[model]
    section = README.split('### ' + title + '\n', 1)[1].split('\n### ', 1)[0]
    return {cells[0]: cells[1:] for line in section.splitlines()
            if line.startswith('| ') and (cells := [x.strip() for x in line.split('|')[1:-1]])}


def main():
    inputs = [
        'audits/servers/server1/official-baselines-20261008/results-review-20261009/table-rows.csv',
        'audits/servers/server2/qwen-migration-results-20261009/gptj-results.json',
    ]
    checked = []
    for row in csv.DictReader((ROOT / inputs[0]).open()):
        if row['W20_verified'] != 'true':
            continue
        assert row['requests'] == '2000'
        cols = ['Score', 'Efficacy', 'Generalization', 'Specificity'] if row['dataset'] == 'cf' else ['Efficacy', 'Generalization', 'Specificity']
        expected = [display(row[k]) for k in cols]
        actual = table('llama3')[LABEL[row['method']]]
        assert (actual[:4] if row['dataset'] == 'cf' else actual[6:9]) == expected, row
        checked.append(['llama3', row['dataset'], row['method'], row['job_id']])
    for row in json.loads((ROOT / inputs[1]).read_text())['rows']:
        s = row['summary']
        assert s['requests'] == 2000 and row['committed_batches'] == 20
        actual = table('gptj')[LABEL[row['method']]]
        if row['dataset'] == 'cf':
            # Uniform R1/P2/N10: recover exact integer fractions from verified
            # request-macro rates; reject anything not on that fraction lattice.
            rates = []
            for key, den in zip(['Efficacy', 'Generalization', 'Specificity'], [2000, 4000, 20000]):
                numerator = round(s[key] * den / 100)
                exact = Decimal(numerator) * 100 / den
                assert abs(float(exact) - s[key]) < 1e-10
                rates.append(exact)
            expected = [display(s['Score'])] + list(map(display, rates))
            assert actual[:4] == expected, (row['method'], actual, expected)
            assert actual[4:6] == ['DEFERRED', 'DEFERRED']
        else:
            assert actual[6:9] == [display(s[k]) for k in ['Efficacy', 'Generalization', 'Specificity']]
        checked.append(['gptj', row['dataset'], row['method'], row['job_id']])
    assert len(checked) == 20
    assert table('llama3')['MEMIT-FE'][:4] == ['ING: 61773'] * 4
    qwen = README.split('### Qwen2.5-7B-Instruct\n')[1].split('### GPT-J-6B')[0]
    assert 'PENDING: 617' not in qwen and 'RESOURCE_BLOCKED_STORAGE_KEEP_SOURCE' in qwen
    print(json.dumps({'status': 'PASS', 'completed_chains': checked,
        'input_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in inputs},
        'GPU': 0, 'model_forward': 0, 'scope': 'compact receipt/table consistency, not new raw or online validation'}, indent=2))


if __name__ == '__main__':
    main()
