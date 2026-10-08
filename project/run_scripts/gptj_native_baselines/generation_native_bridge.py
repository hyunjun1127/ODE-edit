"""SH2 final-only adapter of SH1's distinct native case/global-stream API.

No W0 generation, old partial-row reuse, route qualification, MB selection or
fallback is performed. NativeExecution is an actual observation receipt, not
qualification evidence. F/C share the one generated text through SH1 scoring.
"""
import copy
import json
from pathlib import Path

from project.run_scripts.experiment_generation_eval.assets import load_assets
from project.run_scripts.experiment_generation_eval.common import EVAL_SEED, SCHEMA, immutable_write
from project.run_scripts.experiment_generation_eval.metrics import generation_payload
from project.run_scripts.experiment_generation_eval.native_observer import NativeGenerationObserver
from project.run_scripts.experiment_generation_eval.native_profile import PROFILE, ROUTE
from project.run_scripts.experiment_generation_eval.observer import _cache_signature
from .generation_common import digest, expected_counts, member, read, require, verify
from .generation_bridge import OCCURRENCE_FIELDS
from .generation_native_common import NONCE, SCHEDULE, TASK


class GenerationObserver:
    def __init__(self, config, lock, view, engine, tokenizer, records, out, arm, tracker=None):
        self.config, self.lock, self.view, self.engine = config, lock, view, engine
        self.out, self.arm, self.gen, self.tracker = Path(out), arm, config['generation'], tracker
        self._final_attempted = False
        require(config['task_id'] == TASK and config['instruction_id'] == NONCE
            and self.gen['schema'] == SCHEMA and self.gen['profile'] == PROFILE
            and self.gen['generation_route'] == ROUTE and self.gen['eval_seed'] == EVAL_SEED
            and self.gen['evaluation_schedule'] == SCHEDULE,
            'NATIVE_GENERATION_BRIDGE_PROFILE_AUTHORITY')
        require(not any(key in self.gen for key in ('repair', 'qualification_plan',
            'qualification_receipt_member', 'qualification_owner', 'generation_microbatch',
            'old_w0_reuse', 'W0_cache')), 'NATIVE_GENERATION_LEGACY_DEPENDENCY_FORBIDDEN')
        self.records, self.by_case = [], {}
        records = list(records)
        require(len(records) == 2000, 'NATIVE_GENERATION_FIRST2000')
        for ordinal, original in enumerate(records, 1):
            require(type(original.get('case_id')) is int and original['case_id'] not in self.by_case
                and all(type(original[key]) is int and original[key] == ordinal
                    for key in OCCURRENCE_FIELDS if key in original),
                'NATIVE_GENERATION_RECORD_ORDER')
            record = dict(copy.deepcopy(original), occurrence_index=ordinal)
            self.records.append(record)
            self.by_case[record['case_id']] = record
        self.cohort_sha = digest(self.records)
        self.assets = load_assets(dict(generation_assets=self.gen['generation_assets'],
            asset_paths=self.gen.get('asset_paths', {})))
        require(self.assets.sha == self.gen['reference_assets_sha256'],
            'NATIVE_GENERATION_REFERENCE_IDENTITY')
        require(not (self.out/'generation-raw').exists(), 'NATIVE_GENERATION_CREATE_ONCE_RAW')
        self.shared = NativeGenerationObserver(view.model, tokenizer, self.assets,
            dict(model_identity=self.gen['model_identity'], generation_source_sha=self.gen['source_sha'],
                reference_assets_sha256=self.assets.sha, profile=PROFILE, generation_route=ROUTE,
                eval_seed=EVAL_SEED), self.out/'generation-raw',
            state_callback=self._native_signature, progress_callback=self._progress)
        self.runtime_aux = dict(source_commit=lock['source_commit'], config_sha256=lock['config_sha256'],
            generation_source_sha=self.gen['source_sha'], profile=PROFILE, route=ROUTE,
            shared_runtime_identity=copy.deepcopy(self.shared.runtime_identity),
            shared_runtime_sha256=self.shared.runtime_sha, evaluation_schedule=SCHEDULE,
            W0_generation='NOT_SCHEDULED', intermediate_generation='NOT_SCHEDULED',
            final_requests=2000, qualification_performed=False, no_fallback=True,
            old_partial_reuse=False, sampling_scope='ENDPOINT_GLOBAL_BATCH_STREAM',
            checkpoint_saved=False, raw_local_only=True)
        immutable_write(self.out/'generation-native-runtime.json', self.runtime_aux)
        self.runtime_aux_member = member(self.out/'generation-native-runtime.json')

    def _native_signature(self):
        from .generation_run import native_contexts
        history, module = self.engine.history(), self.engine.module
        return dict(history={str(layer): (value.data_ptr(), value._version,
                tuple(value.shape), str(value.dtype), str(value.device))
                for layer, value in history.items()},
            history_storage=_cache_signature(getattr(self.engine, 'H', None)),
            native_cache=_cache_signature(getattr(module, 'cache_c', None)),
            covariance_cache=_cache_signature(getattr(module, 'COV_CACHE', None)),
            context_sha256=digest(native_contexts(self.engine, self.arm)),
            counts=copy.deepcopy(self.engine.counts), next_batch=self.engine.next_batch)

    def _selected(self, records):
        selected = []
        for original in records:
            record = self.by_case.get(original.get('case_id'))
            require(record is not None and all(type(original[key]) is int
                and original[key] == record['occurrence_index']
                for key in OCCURRENCE_FIELDS if key in original),
                'NATIVE_GENERATION_OCCURRENCE_ORDER')
            plain = lambda value: {key:item for key,item in value.items() if key not in OCCURRENCE_FIELDS}
            require(plain(original) == plain(record), 'NATIVE_GENERATION_RECORD_IDENTITY')
            selected.append(copy.deepcopy(record))
        require(len({row['occurrence_index'] for row in selected}) == len(selected),
            'NATIVE_GENERATION_DUPLICATE_OCCURRENCE')
        return selected

    def _progress(self, payload):
        from .generation_native_tracking import log_generation_progress
        from project.run_scripts.experiment_tracking.schema import metrics
        require(payload.get('phase') == 'generation_evaluation'
            and payload.get('generation_progress/total_cases') == 2000,
            'NATIVE_GENERATION_W20_PROGRESS_ENDPOINT')
        values = dict(copy.deepcopy(payload), phase='W20_generation')
        metrics(values, scientific=True)
        with (self.out/'generation-progress.jsonl').open('a') as journal:
            journal.write(json.dumps(values, sort_keys=True, allow_nan=False)+'\n')
        if self.tracker is not None:
            log_generation_progress(self.tracker, values)

    def _persisted_commits(self, physical_state):
        """Generation cannot precede the actual durable twenty-commit ledger."""
        previous = None
        members = []
        for number in range(1, 21):
            path = self.out/f'batch-{number:02d}'/'commit.json'
            receipt = read(path)
            require(receipt['task'] == TASK and receipt['arm'] == self.arm
                and receipt['batch'] == number and receipt['seen_requests'] == number*100
                and receipt['source'] == self.lock['source_commit']
                and receipt['config'] == digest(self.config)
                and receipt['generation_schedule'] == SCHEDULE
                and receipt['native_counts'] == expected_counts(self.arm)
                and receipt['checkpoint_saved'] is False and len(receipt['case_ids']) == 100,
                'NATIVE_GENERATION_PERSISTED_COMMIT_IDENTITY')
            require(previous is None or receipt['before'] == previous,
                'NATIVE_GENERATION_PERSISTED_COMMIT_LINK')
            previous = receipt['after']
            members.append(member(path))
        require(previous == physical_state, 'NATIVE_GENERATION_PERSISTED_W20_STATE')
        return members

    def _wrap(self, receipt, state_identity, out=None):
        require(receipt['identity']['runtime'] == self.shared.runtime_sha
            and receipt['identity']['state_sha256'] == digest(state_identity)
            and receipt['identity_sha256'] == digest(receipt['identity'])
            and receipt['RNG_restored'] is True and receipt['observer_no_mutation'] is True,
            'NATIVE_GENERATION_SHARED_STATE_IDENTITY')
        require('native_execution_member' in receipt
            and not any(key in receipt for key in ('qualification_receipt_member', 'compatibility_member')),
            'NATIVE_GENERATION_EXECUTION_NOT_QUALIFICATION')
        execution = read(verify(receipt['native_execution_member']))
        require(execution['identity'] == receipt['identity'] and execution['profile'] == PROFILE
            and execution['route'] == ROUTE and execution['qualification_performed'] is False
            and execution['native_execution_complete'] is True and execution['no_fallback'] is True,
            'NATIVE_GENERATION_ACTUAL_EXECUTION_IDENTITY')
        generation_payload('all_seen/post', receipt['summary'])
        result = dict(summary=copy.deepcopy(receipt['summary']), cases=copy.deepcopy(receipt['rows']),
            identity=copy.deepcopy(receipt['identity']), identity_sha256=receipt['identity_sha256'],
            shared_summary=copy.deepcopy(receipt['summary']), full_receipt=copy.deepcopy(receipt),
            rows_path=receipt['rows_path'], work=copy.deepcopy(receipt['work']),
            shared_state_identity=copy.deepcopy(state_identity), RNG_restored=True, observer_no_mutation=True,
            native_execution_member=copy.deepcopy(receipt['native_execution_member']),
            generation_native_runtime_member=self.runtime_aux_member,
            shared_runtime_identity=copy.deepcopy(self.shared.runtime_identity),
            generation_profile=PROFILE, generation_route=ROUTE,
            sampling_scope='ENDPOINT_GLOBAL_BATCH_STREAM', qualification_performed=False,
            no_fallback=True, generation_schedule=SCHEDULE,
            raw_directory=str(Path(receipt['rows_path']).resolve().parents[1]))
        if out is not None:
            path = Path(out)/'receipt.json'
            immutable_write(path, result)
            result['receipt_path'] = str(path.resolve())
        return result

    def endpoint(self, records, *, out, endpoint, model_state, cohort_label):
        from project.run_scripts.gptj_cake_blue_prune_rect.metrics import state
        selected = self._selected(records)
        require(not self._final_attempted and endpoint == 'W20' and cohort_label == 'ALL_SEEN'
            and len(selected) == 2000 and [r['occurrence_index'] for r in selected] == list(range(1, 2001)),
            'NATIVE_GENERATION_ONE_W20_FULL_FIRST2000')
        require(self.engine.next_batch == 21, 'NATIVE_GENERATION_AFTER_TWENTY_NATIVE_COMMITS')
        require(state(self.view, self.engine.history()) == model_state,
            'NATIVE_GENERATION_ACTUAL_FINAL_STATE')
        commits = self._persisted_commits(model_state)
        self._final_attempted = True
        receipt = self.shared.observe(selected, 'W20', cohort='ALL_SEEN', state_identity=model_state)
        require(receipt['identity']['ordered_occurrences'] == list(range(1, 2001))
            and receipt['identity']['state_sha256'] == digest(model_state),
            'NATIVE_GENERATION_FINAL_COHORT_STATE')
        wrapped = self._wrap(receipt, model_state)
        wrapped['native_commit_members'] = commits
        wrapped['committed_batch20_member'] = commits[-1]
        if out is not None:
            path = Path(out)/'receipt.json'
            immutable_write(path, wrapped)
            wrapped['receipt_path'] = str(path.resolve())
        return wrapped

    def load_W0(self):
        raise RuntimeError('NATIVE_FINAL_ONLY_W0_GENERATION_NOT_SCHEDULED')

    def subset_receipt(self, *args, **kwargs):
        raise RuntimeError('NATIVE_FINAL_ONLY_INTERMEDIATE_GENERATION_NOT_SCHEDULED')

    def subset(self, *args, **kwargs):
        raise RuntimeError('NATIVE_FINAL_ONLY_INTERMEDIATE_GENERATION_NOT_SCHEDULED')
