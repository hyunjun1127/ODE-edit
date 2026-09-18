"""Versioned own-trajectory S adapter. Existing M/native files remain unchanged."""
import copy
from pathlib import Path
import torch
from .runtime import Runtime
from .common import digest, tensor_sha, write, Timer, save_tensor
from .alltoken import WEIGHT, model_guard, FullWeightLlamaOracle
from .binding import pack
from project.run_scripts.bg_tw_reference.ep_tw.model_adapter import capture_native_fit, _token_contracts
from project.run_scripts.baseline_mechanism_first.fixtures import capture_rng, restore_rng

class SequentialRuntime(Runtime):
    def __init__(self, lock, output):
        super().__init__(lock, output)
        self.memory_version = self.M._version
        self.projector_version = self.P._version

    def guard(self):
        after = model_guard(self.model)
        strip = lambda g: (tuple(x for x in g[0] if x[1] != WEIGHT), g[1])
        if strip(after) != strip(self.base_guard):
            raise RuntimeError('NONSELECTED_STATE_MUTATION')
        if any(p.requires_grad or p.grad is not None for p in self.model.parameters()):
            raise RuntimeError('UNFROZEN_OR_GRAD_STATE')
        if self.module.CONTEXT_TEMPLATES_CACHE != self.context or self.module.COV_CACHE:
            raise RuntimeError('NATIVE_CONTEXT_CACHE_MUTATION')
        if self.M._version != self.memory_version or self.P._version != self.projector_version:
            raise RuntimeError('INNER_HISTORY_OR_PROJECTOR_MUTATION')

    def reset(self):
        # Only the shared B1 native reuse path may request a cold reset.
        if self.M.count_nonzero() or tensor_sha(self.W) != self.identity['W0']:
            raise RuntimeError('S_COLD_RESET_AFTER_TRAJECTORY_STARTED')
        restore_rng(self.rng); self.guard()
        return dict(W=tensor_sha(self.W), M=tensor_sha(self.M), rng=digest(capture_rng()), independent_cold=True)

    def native_batch(self, records, directory, batch):
        self.oracles = []
        if batch == 1:
            return super().native(records, directory, reuse=True)
        before = dict(W=tensor_sha(self.W), M=tensor_sha(self.M), rng=digest(capture_rng()), independent_cold=False)
        with Timer(self.timing, 'native_fit'):
            result = capture_native_fit(self.fitter, self.model, self.tok, self.hp, self.M, self.P,
                                        self.requests(records), layer=4)
        self.guard()
        r = result['receipt']
        if (r['entry_weight_sha256'] != before['W'] or r['history_sha256'] != before['M'] or
            r['compute_z'] != len(records) or r['history_append'] != 0 or r['solve'] != 1):
            raise RuntimeError('OWN_TRAJECTORY_NATIVE_BINDING')
        source = save_tensor(Path(directory)/'native-capsule.pt', result)
        write(Path(directory)/'native-binding.json', dict(source=source, mode='FRESH_OWN_ENTRY', entry=before,
              endpoint=tensor_sha(result['weight']), case_ids=[r['case_id'] for r in records], receipt=r,
              native_fit_new_calls=1, native_target_new_calls=len(records)))
        return result

    def past_oracle(self, records):
        if not records:
            return None, []
        packs, rows, lookup = [], [], {}
        token = _token_contracts()
        for record in records:
            r = record['requested_rewrite']; case = record['case_id']
            prompt = token.prompt_token_ids(self.etok, r['prompt'].format(r['subject']))
            for branch, field in (('new', 'target_new'), ('old', 'target_true')):
                label = r.get(field, {}).get('str')
                if not label: continue
                target = token.target_token_ids(self.etok, label)
                ids = tuple(prompt + target[:-1])
                if ids not in lookup:
                    lookup[ids] = len(packs); packs.append(pack(ids))
                rows.append(dict(cache=lookup[ids], positions=list(range(len(prompt)-1, len(ids))), labels=target,
                    case_id=case, kind='canonical', branch=branch, context=0,
                    sequence_id=f'{case}:canonical:{branch}'))
        oracle = FullWeightLlamaOracle(self.model, packs)
        self.oracles.append(oracle)
        return oracle, rows

    def finalize_batch(self, records):
        self.guard()
        with Timer(self.timing, 'history_finalize'):
            receipt = self.fitter.finalize(self.model, self.tok, self.requests(records), [(4,self.hp,self.M,self.P)])
        self.memory_version = self.M._version
        self.guard()
        return receipt

    def rollback_entry(self, weight, memory, rng):
        self.oracles = []
        with torch.no_grad(): self.M.copy_(memory)
        self.memory_version = self.M._version
        self.module.CONTEXT_TEMPLATES_CACHE = copy.deepcopy(self.context)
        self.module.COV_CACHE = {}
        self.copy_weight(weight); restore_rng(rng); self.guard()
        if not torch.equal(self.M, memory) or digest(capture_rng()) != digest(rng):
            raise RuntimeError('ENTRY_ROLLBACK_STATE_MISMATCH')
