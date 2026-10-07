"""Task-private adapter of SH1's read-only observer API; no fitting or retry."""
import copy
from pathlib import Path

from project.run_scripts.experiment_generation_eval.assets import load_assets
from project.run_scripts.experiment_generation_eval.common import (
    EVAL_SEED, PROFILE, SCHEMA, digest, immutable_write, require,
)
from project.run_scripts.experiment_generation_eval.metrics import generation_payload
from project.run_scripts.experiment_generation_eval.observer import (
    GenerationObserver as SharedObserver, _cache_signature,
)
from .generation_common import member, read, tensor_sha, verify


OCCURRENCE_FIELDS = ('ordered_occurrence', 'occurrence_index', 'occurrence', 'ordinal')


class GenerationObserver:
    """Bridge public rows/receipts to the existing six-arm runner interface.

    Only BASE_MEMIT may create the shared cold W0 READY. Other arms fail closed
    when it is absent; dependency admission belongs to the runner, not polling.
    Reusable cold identity excludes arm-local history/context; the callback
    still checks that each arm's own native state remains unchanged.
    """
    def __init__(self, config, lock, view, engine, tokenizer, records, out, arm):
        self.config, self.view, self.engine = config, view, engine
        self.out, self.arm, self.gen = Path(out), arm, config['generation']
        require(self.gen['schema'] == SCHEMA and self.gen['profile'] == PROFILE
                and self.gen['eval_seed'] == EVAL_SEED
                and self.gen['W0_owner'] == 'BASE_MEMIT', 'GENERATION_BRIDGE_PROFILE')
        self.records, self.by_case = [], {}
        records = list(records)
        require(len(records) == 2000, 'GENERATION_BRIDGE_FIRST2000_REQUIRED')
        for ordinal, original in enumerate(records, 1):
            require(type(original.get('case_id')) is int
                    and original['case_id'] not in self.by_case, 'GENERATION_BRIDGE_CASE_OCCURRENCE')
            require(all(type(original[key]) is int and original[key] == ordinal
                        for key in OCCURRENCE_FIELDS if key in original),
                    'GENERATION_BRIDGE_OCCURRENCE_ORDER')
            record = copy.deepcopy(original)
            record['occurrence_index'] = ordinal
            self.records.append(record)
            self.by_case[record['case_id']] = record
        self.cohort_sha = digest(self.records)
        self.cold_state = copy.deepcopy(self.gen['W0_state_identity'])
        require(self.cold_state and self.gen['model_identity'], 'GENERATION_BRIDGE_FULL_COLD_IDENTITY')
        self.cache = Path(self.gen['W0_cache'])
        binding = dict(generation_assets=self.gen['generation_assets'],
                       asset_paths=self.gen.get('asset_paths', {}))
        self.assets = load_assets(binding)
        require(self.assets.sha == self.gen['reference_assets_sha256'],
                'GENERATION_BRIDGE_REFERENCE_IDENTITY')
        self.shared = SharedObserver(view.model, tokenizer, self.assets,
            dict(model_identity=self.gen['model_identity'], generation_source_sha=self.gen['source_sha'],
                 profile=PROFILE, eval_seed=EVAL_SEED), self.out / 'generation-raw',
            state_callback=self._native_signature)
        self._w0, self._receipts = None, {}

    def _native_signature(self):
        history = self.engine.history()
        module = self.engine.module
        return dict(history={str(layer): (value.data_ptr(), value._version,
                    tuple(value.shape), str(value.dtype), str(value.device))
                    for layer, value in history.items()},
            history_storage=_cache_signature(getattr(self.engine, 'H', None)),
            native_cache=_cache_signature(getattr(module, 'cache_c', None)),
            covariance_cache=_cache_signature(getattr(module, 'COV_CACHE', None)),
            context_sha256=digest(module.CONTEXT_TEMPLATES_CACHE),
            counts=copy.deepcopy(self.engine.counts), next_batch=self.engine.next_batch)

    def _selected(self, records):
        selected = []
        for original in records:
            record = self.by_case.get(original.get('case_id'))
            require(record is not None, 'GENERATION_BRIDGE_UNKNOWN_OCCURRENCE')
            require(all(type(original[key]) is int and original[key] == record['occurrence_index']
                        for key in OCCURRENCE_FIELDS if key in original),
                    'GENERATION_BRIDGE_OCCURRENCE_ORDER')
            plain = lambda value: {key: item for key, item in value.items() if key not in OCCURRENCE_FIELDS}
            require(plain(original) == plain(record), 'GENERATION_BRIDGE_RECORD_IDENTITY')
            selected.append(copy.deepcopy(record))
        require(len({record['occurrence_index'] for record in selected}) == len(selected),
                'GENERATION_BRIDGE_DUPLICATE_OCCURRENCE')
        return selected

    def _cold_weights(self):
        actual = {str(layer): tensor_sha(weight) for layer, weight in self.view.weights.items()}
        require(actual == {str(layer): self.config['cold_W'][str(layer)] for layer in self.view.sites},
                'GENERATION_BRIDGE_ACTUAL_COLD_W0')

    def _ready_identity(self):
        return dict(runtime=self.shared.runtime_sha, model_identity=self.gen['model_identity'],
            source_sha=self.gen['source_sha'], reference_assets_sha256=self.assets.sha,
            profile=PROFILE, eval_seed=EVAL_SEED, cohort_sha256=self.cohort_sha,
            ordered_occurrences=list(range(1, 2001)), cold_state_sha256=digest(self.cold_state))

    def _validate_W0(self, receipt):
        require(receipt['identity']['runtime'] == self.shared.runtime_sha
                and receipt['identity']['state_sha256'] == digest(self.cold_state)
                and receipt['identity']['endpoint'] == 'W0'
                and receipt['identity']['cohort'] == 'FIRST2000'
                and receipt['identity']['ordered_occurrences'] == list(range(1, 2001))
                and [row['case_id'] for row in receipt['rows']] == [row['case_id'] for row in self.records],
                'GENERATION_BRIDGE_W0_FULL_COHORT')

    def _wrap(self, receipt, state_identity, out=None):
        require(receipt['identity']['runtime'] == self.shared.runtime_sha
                and receipt['identity']['state_sha256'] == digest(state_identity)
                and receipt['identity_sha256'] == digest(receipt['identity']),
                'GENERATION_BRIDGE_SHARED_STATE_IDENTITY')
        # The shared mapper checks builtin finite scalars without text/tokens.
        generation_payload('current/post', receipt['summary'])
        result = dict(summary=copy.deepcopy(receipt['summary']), cases=copy.deepcopy(receipt['rows']),
            identity=copy.deepcopy(receipt['identity']), identity_sha256=receipt['identity_sha256'],
            shared_summary=copy.deepcopy(receipt['summary']), full_receipt=copy.deepcopy(receipt),
            rows_path=receipt['rows_path'], work=copy.deepcopy(receipt['work']),
            shared_state_identity=copy.deepcopy(state_identity),
            raw_directory=str(Path(receipt['rows_path']).resolve().parents[1]),
            RNG_restored=receipt['RNG_restored'], observer_no_mutation=receipt['observer_no_mutation'])
        self._receipts[digest(result['cases'])] = result
        if out is not None:
            path = Path(out) / 'receipt.json'
            immutable_write(path, result)
            result['receipt_path'] = str(path.resolve())
        return result

    def load_W0(self):
        self._cold_weights()
        if self._w0 is not None:
            return copy.deepcopy(self._w0)
        ready_path = self.cache / 'READY.json'
        identity = self._ready_identity()
        if ready_path.exists():
            ready = read(ready_path)
            require(ready['status'] == 'READY' and ready['identity'] == identity
                    and ready['identity_sha256'] == digest(identity)
                    and ready['producer_arm'] == 'BASE_MEMIT', 'GENERATION_BRIDGE_W0_READY_IDENTITY')
            receipt = self.shared.read_observed(verify(ready['endpoint']))
        else:
            require(self.arm == 'BASE_MEMIT', 'GENERATION_BRIDGE_W0_NOT_READY_NO_POLL')
            receipt = self.shared.observe(self.records, 'W0', cohort='FIRST2000',
                                          state_identity=self.cold_state)
            self.shared.read_observed(receipt['rows_path'])
            self._validate_W0(receipt)
            immutable_write(ready_path, dict(status='READY', identity=identity,
                identity_sha256=digest(identity), producer_arm='BASE_MEMIT',
                endpoint=member(receipt['rows_path']), producer_work=receipt['work'],
                checkpoint_saved=False, raw_local_only=True))
        self._validate_W0(receipt)
        self._w0 = self._wrap(receipt, self.cold_state, self.out / 'generation-W0')
        return copy.deepcopy(self._w0)

    def endpoint(self, records, *, out, endpoint, model_state, cohort_label):
        selected = self._selected(records)
        if endpoint == 'B1_PRE':
            require(self._w0 is not None and [r['occurrence_index'] for r in selected] == list(range(1, 101))
                    and model_state['W'] == {str(layer): self.config['cold_W'][str(layer)]
                        for layer in self.view.sites}, 'GENERATION_BRIDGE_B1_PRE_COLD_SUBSET')
            return self.subset_receipt(self._w0, records, endpoint=endpoint,
                                       cohort_label=cohort_label, out=out)
        receipt = self.shared.observe(selected, endpoint, cohort=cohort_label,
                                      state_identity=model_state)
        return self._wrap(receipt, model_state, out)

    def subset_receipt(self, observed, records, *, endpoint='CURRENT_SUBSET', cohort_label='CURRENT', out=None):
        receipt = self.shared.subset(observed['full_receipt'], self._selected(records),
                                     endpoint, cohort=cohort_label)
        return self._wrap(receipt, observed['shared_state_identity'], out)

    def subset(self, cases, records):
        """Compatibility summary API; production keeps subset_receipt instead."""
        observed = self._receipts.get(digest(cases))
        require(observed is not None, 'GENERATION_BRIDGE_SUBSET_PARENT_RECEIPT_REQUIRED')
        return self.subset_receipt(observed, records)['summary']

    @staticmethod
    def payload(prefix, summary):
        return generation_payload(prefix, summary)
