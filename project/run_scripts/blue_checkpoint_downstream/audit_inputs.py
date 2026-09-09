"""CPU-only sealed source/data audit; never opens checkpoint payloads or a model."""
import argparse
import ast
import hashlib
import json
import os
import subprocess
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_members(root, members):
    for member in members:
        relative = Path(member['relative'])
        assert not relative.is_absolute() and '..' not in relative.parts
        path = root / relative
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == member['bytes'] and sha(path) == member['sha256'], str(path)


def parser(source, class_name, method_name, argument):
    cls = next(x for x in ast.parse(source).body if isinstance(x, ast.ClassDef) and x.name == class_name)
    method = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == method_name)
    namespace = {}
    # Only the fully reviewed pure string/label method; no source imports or module execution.
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<reviewed-parser-fixture>', 'exec'), namespace)
    return namespace[method_name](None, argument)


def audit(imports, dataset_receipts):
    source = imports / 'source'
    evaluator = imports / 'evaluator-source'
    assert sha(source/'source-seal.json') == '37dacc4677e7952f16377800d3e85889b312d6352f38bd31b78f3744da28e5aa'
    assert sha(source/'checkpoint-manifest.json') == 'e4625ab025e6bf57c30a5c3a1e6eece01557cd2d204c3266a37777368368884a'
    assert sha(evaluator/'source-manifest.json') == '21badc13a1ceed8ac25edf545fcf0b3a3b9e6b69577240bf712acde84b9581e5'
    sm = json.loads((source/'source-seal.json').read_text())
    em = json.loads((evaluator/'source-manifest.json').read_text())
    verify_members(source, sm['members'])
    verify_members(evaluator, em['members'])
    cp = json.loads((source/'checkpoint-manifest.json').read_text())
    assert cp['count'] == len(cp['checkpoints']) == 72
    assert sum(x['file']['bytes'] for x in cp['checkpoints']) == cp['total_checkpoint_bytes'] == 62011141768
    for row in cp['checkpoints']:
        assert set(row['weights']) == {f'model.layers.{layer}.mlp.down_proj.weight' for layer in row['layers']}
        assert all(x['dtype'] == 'torch.float32' and x['shape'] == [4096,14336] for x in row['weights'].values())
        assert row['sample_root'] == '5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729'
    data = json.loads(subprocess.check_output(['python3', '-B', str(dataset_receipts/'verify.py'), '/mnt/raid5/janghj/EasyEdit/glue_eval/dataset'], text=True))
    expected = json.loads((dataset_receipts/'source-audit.json').read_text())
    assert data['members'] == expected['members'] and data['datasets'] == expected['datasets']
    assert data['member_root'] == 'e9328a5d351816cb9ba89454d228f7ade841c526313dae9c0d1a0e72a8ab00fc'
    rte = (evaluator/'glue_eval/rte_eval.py').read_text()
    nli = (evaluator/'glue_eval/nli_eval.py').read_text()
    mmlu = (evaluator/'glue_eval/mmlu_eval.py').read_text()
    facts = {
        'rte_true_prediction': parser(rte, 'RTEEval', '_get_answer', 'answer: True'),
        'rte_false_prediction': parser(rte, 'RTEEval', '_get_answer', 'answer: False'),
        'nli_entailment': parser(nli, 'NLIEval', '_get_label', 'entailment'),
        'nli_not_entailment': parser(nli, 'NLIEval', '_get_label', 'not_entailment'),
        'mmlu_A_without_newline': parser(mmlu, 'MMLUEval', '_get_answer', 'A'),
        'mmlu_A_with_newline': parser(mmlu, 'MMLUEval', '_get_answer', 'A\n'),
    }
    assert facts == dict(rte_true_prediction=1, rte_false_prediction=0, nli_entailment=1,
                         nli_not_entailment=0, mmlu_A_without_newline=-1, mmlu_A_with_newline=0)
    return dict(status='EVALUATOR_SEMANTIC_HOLD', asset_identity='PASS', source_members=len(sm['members']),
                evaluator_members=len(em['members']), dataset_member_root=data['member_root'], checkpoints_declared=72,
                checkpoint_payload_verified=False, parser_fixtures=facts,
                blocker='RTE True->1 compared to raw GLUE label; GLUE entailment=0. No silent remap.',
                mmlu_note='f1 generation newline-sensitive parser; f1_new alternative probability prediction; distinct',
                tasks=['sst2','mrpc','cola','rte','mmlu','nli'], eval_slice=[10,110], fewshot=0, gen_len=5,
                model_load=0, gpu=0, submission=0, scientific_promotion=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--imports', type=Path, required=True)
    p.add_argument('--dataset-receipts', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.imports, a.dataset_receipts)
    fd = os.open(a.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False))
