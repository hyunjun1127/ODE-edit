"""R512 plus a selected, fixed history block, with the parent's controller API."""
import hashlib
import json
import time
import torch
from project.run_scripts.en_adaptive_nullspace.objective import Objective as ReferenceObjective
from .factors import HistoryReplay


class Objective:
    def __init__(self, model, tokenizer, generated_root, reference_inputs, history_root, basis, expected_map_seal=None):
        self.reference = ReferenceObjective(model, generated_root, reference_inputs)
        self.replay = HistoryReplay(model, tokenizer, history_root, basis, expected_map_seal=expected_map_seal)
        self.store = self.reference.store
        self.selected_ids, self.weights = [], []
        self.bank_identity = None
        self.history_sweeps = []

    def rebind(self, expected):
        self.reference.rebind(expected)
        self.replay.rebind(expected)

    def prepare(self, weight, version_rows, *, arm, batch):
        start = time.monotonic()
        bank = self.replay.prepare_pool(weight, version_rows,
            select_gss=arm == 'EN_ADAPT_H_GSS_REC', current_batch=batch,
            recency=arm == 'EN_ADAPT_H_GSS_REC')
        self.selected_ids = list(bank['selected_ids'])
        self.weights = [float(v) for v in bank['weights']]
        self.bank_identity = hashlib.sha256(json.dumps(dict(ids=self.selected_ids,
            weights=self.weights, versions=[dict(version_id=r['version_id'],
            created_batch=r['created_batch'], teacher_binding=r.get('teacher_binding'))
            for r in version_rows if r['version_id'] in set(self.selected_ids)]),
            sort_keys=True, allow_nan=False).encode()).hexdigest()
        _, gradient, reference = self.reference.evaluate(weight, gradient=True, kind='native')
        bank['reference_gradient'] = gradient.clone()
        h = bank['gradient']
        nr = float(gradient.norm())
        nh = float(h.norm()) if h is not None else 0.
        inner = float(torch.dot(gradient.reshape(-1),h.reshape(-1))) if h is not None else 0.
        if bank['gradient'] is not None:
            if bank['gradient'].dtype != torch.float64 or bank['gradient'].shape != gradient.shape:
                raise ValueError('HISTORY_GRADIENT_SCHEMA')
            gradient.add_(bank['gradient'])
        if not torch.isfinite(gradient).all():
            raise FloatingPointError('NONFINITE_COMBINED_GRADIENT')
        receipt = dict(J=reference['L_R'] + bank['L_H'], L_R=reference['L_R'],
            L_H=bank['L_H'], reference=reference, history=bank['receipt'],
            bank_identity=self.bank_identity, history_ids=self.selected_ids,
            history_weights=self.weights, block_coefficients=[1., 1.],
            combined_gradient='G_R + G_H; cross terms retained by parent geometry',
            gradient_blocks=dict(G_R_norm=nr,G_H_norm=nh,inner_product=inner,
                cross_term=2*inner,cosine=inner/(nr*nh) if nr*nh else None,
                combined_norm=float(gradient.norm())),
            seconds=time.monotonic()-start)
        self.history_sweeps.append(dict(batch=batch, role='gradient', **bank['receipt']))
        return gradient, receipt, bank

    def evaluate(self, weight):
        start = time.monotonic()
        _, _, reference = self.reference.evaluate(weight, kind='candidate')
        value, hist = self.replay.evaluate(weight, self.selected_ids, self.weights)
        self.history_sweeps.append(dict(role='candidate', **hist))
        return dict(J=reference['L_R']+value, L_R=reference['L_R'], L_H=value,
                    reference=reference, history=hist, bank_identity=self.bank_identity,
                    history_ids=self.selected_ids, history_weights=self.weights,
                    seconds=time.monotonic()-start)

    def capture(self, version_rows, weight):
        return self.replay.capture(version_rows, weight)
