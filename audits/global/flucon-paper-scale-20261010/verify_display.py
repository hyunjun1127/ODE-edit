"""Read-only display regression; no remote reads/model execution."""
import hashlib
import json
from pathlib import Path
import sys

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
    policy = json.loads((ROOT / 'control/main-results-policy.json').read_text())
    assert 'generation_paper_display' in policy
    assert paper_cell('DEFERRED', metric='Flu', raw_unit='bits') == 'DEFERRED'
    return {'corrected_cells': 4, 'unit_headers': 4, 'status': 'PASS',
            'README_sha256': hashlib.sha256(readme.encode()).hexdigest(),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'GPU_forwards': 0, 'raw_mutations': 0}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
