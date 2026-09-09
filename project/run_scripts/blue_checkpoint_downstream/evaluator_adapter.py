"""Direct six-class BLUE evaluation with pinned local data and RTE scoring only.

Unused GLUEEval wrapper/util imports are not called. Original source is loaded
from its sealed directory; no evaluator formula, prompt or parser is copied.
"""
from collections import Counter
import hashlib
import importlib
import io
import json
from pathlib import Path
import pickle
import sys

from rte_scoring import score_source_rows

TASKS = {'sst2': ('sst_eval', 'SSTEval'), 'mrpc': ('mrpc_eval', 'MRPCEval'),
         'cola': ('cola_eval', 'COLAEval'), 'rte': ('rte_eval', 'RTEEval'),
         'mmlu': ('mmlu_eval', 'MMLUEval'), 'nli': ('nli_eval', 'NLIEval')}
CANONICAL_MODEL_NAME = 'meta-llama/Meta-Llama-3-8B-Instruct'


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(',', ':'), allow_nan=False).encode()


class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError('PICKLE_GLOBAL_FORBIDDEN')

    def persistent_load(self, pid):
        raise ValueError('PERSISTENT_ID_FORBIDDEN')


def read_rows(path):
    stream = io.BytesIO(Path(path).read_bytes())
    rows = PrimitiveUnpickler(stream).load()
    assert not stream.read() and isinstance(rows, list)
    return rows


class DatasetBinding:
    def __init__(self, root, expected):
        self.root = Path(root).resolve()
        self.expected = expected
        self.rows = {}
        for task in TASKS:
            path = self.root/f'{task}.pkl'
            member = next(x for x in expected['members'] if x['name'] == path.name)
            assert path.is_file() and not path.is_symlink()
            assert hashlib.sha256(path.read_bytes()).hexdigest() == member['sha256']
            rows = read_rows(path)
            selected = rows[10:110]
            digest = hashlib.sha256('\n'.join(hashlib.sha256(canonical(r)).hexdigest()
                                              for r in selected).encode()).hexdigest()
            assert digest == expected['datasets'][task]['eval100_ordered_row_hash']
            self.rows[task] = rows

    def split(self, filename, number_of_few_shots, number_of_tests):
        # Exact source slice, with an absolute allowlisted root instead of cwd.
        assert number_of_few_shots == 0 and number_of_tests == 100
        path = Path(filename)
        assert str(path.parent) == 'glue_eval/dataset' and path.stem in TASKS
        assert path.suffix == '.pkl'
        rows = self.rows[path.stem]
        return rows[:10][:number_of_few_shots], rows[10:][:number_of_tests]


def bind_classes(source_root, binding):
    root = Path(source_root).resolve()
    sys.path.insert(0, str(root))
    helper = importlib.import_module('glue_eval.useful_functions')
    assert Path(helper.__file__).resolve() == root/'glue_eval/useful_functions.py'
    assert helper.FEW_SHOT_TEST_SPLIT == 10
    assert helper.MODEL_NAME_TO_MAXIMUM_CONTEXT_LENGTH_MAP['meta-llama-3-8b-instruct'] == 4096
    helper.load_data_split = binding.split
    classes = {}
    for task, (module_name, class_name) in TASKS.items():
        module = importlib.import_module(f'glue_eval.{module_name}')
        assert Path(module.__file__).resolve() == root/f'glue_eval/{module_name}.py'
        assert module.load_data_split == binding.split
        classes[task] = getattr(module, class_name)
    return classes


def metric_summary(task, original, raw_rows, gold_rows):
    assert len(raw_rows) == len(gold_rows) == 100 and original['total'] == 100
    if task == 'rte':
        return score_source_rows(raw_rows, gold_rows, original)
    # Keep source metrics authoritative. Count alternative invalids from the
    # exact literal prediction label, without parsing generated text again.
    if task == 'mmlu':
        labels = [r['answer'] for r in gold_rows]
        valid_labels = {'A','B','C','D'}
    elif task == 'nli':
        labels = [1 if r['label'] == 'entailment' else 0 for r in gold_rows]
        valid_labels = {'True','False'}
    else:
        labels = [r['label'] for r in gold_rows]
        valid_labels = {'positive','negative'} if task == 'sst2' else {'Yes','No'}
    alternative_invalid = sum(r['highest_probability_answer'] not in valid_labels for r in raw_rows)
    return dict(status='SOURCE_EXACT_REFERENCE_METRIC', source_metrics=original,
                generation=dict(correct=original['correct'], total=100,
                                invalid=original['invalid'], accuracy=original['correct']/100,
                                weighted_f1=original['f1'], mcc=original['mcc']),
                alternative=dict(correct=sum(bool(r['correct_new']) for r in raw_rows), total=100,
                                 invalid=alternative_invalid,
                                 accuracy=sum(bool(r['correct_new']) for r in raw_rows)/100,
                                 weighted_f1=original['f1_new']),
                support=dict(Counter(str(x) for x in labels)), f1=original['f1'], f1_new=original['f1_new'],
                official_full_benchmark=False)


def evaluate_task(task, classes, binding, model, tokenizer):
    assert model.config._name_or_path == CANONICAL_MODEL_NAME
    evaluator = classes[task](model, tokenizer, number_of_tests=100, number_of_few_shots=0)
    assert evaluator.eval_dataset == binding.rows[task][10:110]
    original, rows = evaluator.evaluate(gen_len=5, print_logs=False)
    summary = metric_summary(task, original, rows, binding.rows[task][10:110])
    return summary, rows
