"""RAM-only native batch transaction; no reconstructible state is persisted."""
import copy

import torch

from project.run_scripts.jlz_realization.common import require
from project.run_scripts.jlz_realization.writer import rng_equal, rng_restore, rng_snapshot
from .metrics import HOOK_FIELDS, state


def nonselected(view):
    selected = {id(value) for value in view.weights.values()}
    return tuple((name, id(value), value.data_ptr(), value._version,
                  tuple(value.shape), str(value.dtype))
                 for name, value in view.model.named_parameters() if id(value) not in selected)


class NativeTransaction:
    """Successful finish persists W/H. Any exception restores batch entry.

    An escaping commit-file IO error after finish invalidates it automatically;
    invalidate() also permits an explicit abort without an exception. Prior
    successful batch prefixes are not reset.
    Nonselected weights are guarded rather than cloned: mutating them is a typed
    technical violation and cannot be called a verified rollback.
    """
    def __init__(self, view, engine, bench, cursor=None):
        self.view, self.engine, self.bench, self.cursor = view, engine, bench, cursor
        self.done = False
        self.rollback_verified = False
        self.rollback_ledger_verified = False

    def __enter__(self):
        require(not self.done, 'TRANSACTION_REUSE')
        require(all(callable(getattr(self.engine, name, None)) for name in
                    ('snapshot_ledger', 'restore_ledger', 'ledger_identity')),
                'NATIVE_LOGICAL_LEDGER_API_REQUIRED')
        self.ledger_snapshot = self.engine.snapshot_ledger()
        self.ledger_before = self.engine.ledger_identity(self.ledger_snapshot)
        self.before = state(self.view, self.engine.history())
        self.guard, self.hooks = nonselected(self.view), self.view.hook_signature()
        self.hook_maps = [(module, {field: dict(getattr(module, field, {}))
                                  for field in HOOK_FIELDS},
                           getattr(module, '_is_full_backward_hook', None))
                          for _, module in self.view.model.named_modules()]
        self.rng = rng_snapshot()
        self.context = copy.deepcopy(self.bench.contexts)
        self.native_context = copy.deepcopy(self.engine.contexts())
        self.context_snapshot = (self.engine.snapshot_context()
                                 if hasattr(self.engine, 'snapshot_context') else None)
        self.cursor_before = None if self.cursor is None else list(self.cursor)
        self.W = {layer: weight.detach().cpu().clone()
                  for layer, weight in self.view.weights.items()}
        self.H = {layer: value.detach().cpu().clone()
                  for layer, value in self.engine.history().items()}
        return self

    def finish(self):
        require(nonselected(self.view) == self.guard
                and self.view.hook_signature() == self.hooks
                and self.bench.contexts == self.context
                and self.engine.contexts() == self.native_context,
                'NATIVE_COMMIT_GUARD')
        self.done = True

    def invalidate(self):
        self.done = False

    def __exit__(self, kind, value, tb):
        if kind is not None:
            self.done = False
        try:
            if not self.done:
                # Restore original native H aliases/initialized flags before
                # copying saved H contents. Cursor/caches are not cost counters.
                protected_unchanged = self.engine.restore_ledger(self.ledger_snapshot)
                with torch.no_grad():
                    for layer, weight in self.view.weights.items():
                        weight.copy_(self.W[layer])
                    history = self.engine.history()
                    if self.H:
                        require(set(history) == set(self.H), 'NATIVE_ROLLBACK_HISTORY_LAYOUT')
                        for layer, current in history.items():
                            current.copy_(self.H[layer])
                    elif history:
                        self.engine.reset_history_to_uninitialized()
                self.bench.contexts = copy.deepcopy(self.context)
                for module, mappings, full_backward in self.hook_maps:
                    for field, mapping in mappings.items():
                        target = getattr(module, field, None)
                        if target is not None:
                            target.clear()
                            target.update(mapping)
                    module._is_full_backward_hook = full_backward
                if self.context_snapshot is not None:
                    require(hasattr(self.engine, 'restore_context'),
                            'NATIVE_CONTEXT_RESTORE_API')
                    self.engine.restore_context(self.context_snapshot)
                else:
                    require(self.engine.contexts() == self.native_context,
                            'NATIVE_CONTEXT_RESTORE_NOT_VERIFIED')
                if self.cursor is not None:
                    self.cursor[:] = self.cursor_before
                rng_restore(self.rng)
                require(protected_unchanged is not False
                        and self.engine.ledger_identity() == self.ledger_before,
                        'NATIVE_LOGICAL_LEDGER_ROLLBACK_MISMATCH')
                require(state(self.view, self.engine.history()) == self.before
                        and nonselected(self.view) == self.guard
                        and self.view.hook_signature() == self.hooks
                        and self.bench.contexts == self.context
                        and self.engine.contexts() == self.native_context
                        and (self.cursor is None or self.cursor == self.cursor_before)
                        and rng_equal(self.rng), 'NATIVE_ROLLBACK_MISMATCH')
                self.rollback_ledger_verified = True
                self.rollback_verified = True
        finally:
            self.W.clear()
            self.H.clear()
            self.context_snapshot = None
            self.ledger_snapshot = None
            self.hook_maps.clear()
        return False
