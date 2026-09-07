"""D preparatory CPU workflow over an already captured, frozen raw bundle.

API: ``FrozenDiagnosisBundle.seal(...)`` then ``analyze_frozen_bundle(...)``.
The parent owns capture/runtime, authorized paths, publication, and GPU budget.
This module never loads a model, restores historical weights, or executes replay.
``read_publication_availability(repo)`` reads two pinned Git CSV blobs and checks
their exact local historical paths once; missing D inputs do not hold S.
"""
from dataclasses import dataclass
from copy import deepcopy
import csv
import hashlib
import io
import json
import math
import re
import subprocess

import torch

from project.run_scripts.native_response_ode_v31.algebra import FrozenNormalization
from .contracts import PUBLICATION_HEAD
from .normalization_views import source_residual, tensor_identity
from .diagnosis import (capture_repeatability, same_state_attribution,
                        cross_objective_scores, historical_availability)


class DiagnosisInputBoundary(RuntimeError):
    pass


_PUBLICATION_PREFIX = (
    'experiment-reports/servers/server2/'
    'alpha-native-response-v31-sequential-routing-2026-09-06-v1/main-four-terminal-v1/'
)
_SHA_FIELDS = ('entry_W_sha256', 'entry_M_sha256', 'fixed_z_sha256',
               'request_order_sha256', 'semantic_inventory_sha256',
               'context_sha256', 'metric_identity_sha256')
_PROVENANCE = ('EXACT_STATE_REPLAY', 'SEALED_EXACT_STATE', 'NEW_REPLAY_ANALOGUE',
               'RETROSPECTIVE_CONTROL', 'CPU_SYNTHETIC_FIXTURE')


def _canonical_sha(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _cpu_tensor(value, dtype, label):
    if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
            or value.dtype != dtype or not torch.isfinite(value).all()):
        raise DiagnosisInputBoundary('CPU_FINITE_DTYPE_' + label)
    return value.detach().clone().contiguous()


@dataclass(frozen=True)
class FrozenDiagnosisBundle:
    """One immutable CPU capture, with source provenance supplied by its owner.

    A CPU checksum validates input bytes, not the external model/state provenance;
    the output retains that distinction. Endpoint captures are optional, so absent
    endpoints remain missing rather than substituting an entry or predicted state.
    """
    target32: torch.Tensor
    entry_captures32: tuple
    raw_responses32: torch.Tensor
    q_layers: torch.Tensor
    metric: torch.Tensor
    frobenius_metric: object
    layer_ids: tuple
    request_ids: tuple
    evidence: dict
    endpoint_terminals: dict
    _member_hashes: dict
    _metadata_sha256: str

    @classmethod
    def seal(cls, *, target32, entry_captures32, raw_responses32, q_layers,
             metric, layer_ids, request_ids, evidence, endpoint_terminals=None,
             frobenius_metric=None):
        target = _cpu_tensor(target32, torch.float32, 'TARGET')
        if target.ndim != 2 or len(entry_captures32) != 3:
            raise DiagnosisInputBoundary('TARGET_SHAPE_OR_CAPTURE_COUNT')
        captures = tuple(_cpu_tensor(value, torch.float32, 'ENTRY') for value in entry_captures32)
        if any(value.shape != target.shape for value in captures):
            raise DiagnosisInputBoundary('ENTRY_CAPTURE_SHAPE')
        raw = _cpu_tensor(raw_responses32, torch.float32, 'RAW_JVP')
        q = _cpu_tensor(q_layers, torch.float64, 'Q')
        metric = _cpu_tensor(metric, torch.float64, 'METRIC')
        layers, ids = tuple(layer_ids), tuple(str(value) for value in request_ids)
        if (raw.shape != (len(layers), *target.shape) or q.shape != (len(layers),)
                or metric.shape != (len(layers), len(layers))
                or len(set(layers)) != len(layers) or any(layer not in (4, 5, 6, 7, 8) for layer in layers)
                or len(ids) != target.shape[1] or len(set(ids)) != len(ids)
                or bool((q <= 0).any())):
            raise DiagnosisInputBoundary('RAW_REQUEST_LAYER_INVENTORY')
        gf = (_cpu_tensor(frobenius_metric, torch.float64, 'FROBENIUS_METRIC')
              if frobenius_metric is not None else None)
        if gf is not None and gf.shape != metric.shape:
            raise DiagnosisInputBoundary('FROBENIUS_METRIC_SHAPE')
        evidence = deepcopy(evidence)
        if (evidence.get('track') != 'D' or evidence.get('provenance_mode') not in _PROVENANCE
                or not evidence.get('fixture') or not evidence.get('model')
                or any(not re.fullmatch(r'[0-9a-f]{64}', str(evidence.get(key, ''))) for key in _SHA_FIELDS)
                or not re.fullmatch(r'[0-9a-f]{40}', str(evidence.get('source_sha', '')))
                or not isinstance(evidence.get('qN_ref'), (int, float))
                or not math.isfinite(evidence['qN_ref']) or evidence['qN_ref'] < 0
                or len(layers) and evidence['qN_ref'] <= 0
                or not isinstance(evidence.get('cache_c_new'), bool)):
            raise DiagnosisInputBoundary('D_PROVENANCE_IDENTITY_REQUIRED')
        endpoints = {str(name): _cpu_tensor(value, torch.float32, 'ENDPOINT')
                     for name, value in (endpoint_terminals or {}).items()}
        if any(value.shape != target.shape for value in endpoints.values()):
            raise DiagnosisInputBoundary('ENDPOINT_CAPTURE_SHAPE')
        members = dict(target32=target, raw_responses32=raw, q_layers=q, metric=metric,
                       **{f'entry_capture_{i}': value for i, value in enumerate(captures)},
                       **{f'endpoint:{name}': value for name, value in endpoints.items()})
        if gf is not None:
            members['frobenius_metric'] = gf
        hashes = {name: tensor_identity(value) for name, value in members.items()}
        metadata = dict(layer_ids=layers, request_ids=ids, evidence=evidence)
        return cls(target, captures, raw, q, metric, gf, layers, ids, evidence, endpoints,
                   hashes, _canonical_sha(metadata))

    def assert_frozen(self):
        members = dict(target32=self.target32, raw_responses32=self.raw_responses32,
            q_layers=self.q_layers, metric=self.metric,
            **{f'entry_capture_{i}': value for i, value in enumerate(self.entry_captures32)},
            **{f'endpoint:{name}': value for name, value in self.endpoint_terminals.items()})
        if self.frobenius_metric is not None:
            members['frobenius_metric'] = self.frobenius_metric
        if ({name: tensor_identity(value) for name, value in members.items()} != self._member_hashes
                or _canonical_sha(dict(layer_ids=self.layer_ids, request_ids=self.request_ids,
                                       evidence=self.evidence)) != self._metadata_sha256):
            raise DiagnosisInputBoundary('FROZEN_DIAGNOSIS_INPUT_MUTATED')


def analyze_frozen_bundle(bundle, *, availability=None, include_all_row=True):
    """Return JSON-ready D tables/receipt. Do not write or create missing data.

    Missing bundle is a valid preparation status, not an S dependency. Existing
    endpoints are cross-scored only from their supplied real terminal captures.
    The execution lambda is locked .1; no new D hyperparameter axis is created.
    """
    if bundle is None:
        return dict(status='WAITING_FOR_FROZEN_D_RAW_BUNDLE',
            availability=deepcopy(availability), diagnosis_attribution_status='NOT_RECORDED_RAW_BUNDLE_UNAVAILABLE',
            endpoint_status='NOT_RECORDED_ACTUAL_ENDPOINT_CAPTURE', tables={},
            S_blocked_by_D=False, model_load_count=0, replay_action_count=0,
            actual_write_count=0, scientific_promotion=False)
    if not isinstance(bundle, FrozenDiagnosisBundle):
        raise DiagnosisInputBoundary('SEALED_DIAGNOSIS_BUNDLE_REQUIRED')
    bundle.assert_frozen()
    source = FrozenNormalization.capture(bundle.target32, bundle.entry_captures32[0],
                                         bundle.evidence['entry_W_sha256'])
    residual = source_residual(bundle.target32, bundle.entry_captures32[0])
    # Do not infer per-capture context/tokenization metadata from the fixed-z
    # inventory. If the capture owner did not record it, retain the schema gap.
    semantic = bundle.evidence.get('capture_semantic_identities', ['NOT_RECORDED'] * 3)
    if not isinstance(semantic, (tuple, list)) or len(semantic) != 3:
        raise DiagnosisInputBoundary('CAPTURE_SEMANTIC_METADATA_COUNT')
    repeatability = capture_repeatability(bundle.target32, bundle.entry_captures32,
                                         semantic_identities=semantic)
    attribution = same_state_attribution(source_normalization=source,
        residual32=residual, raw_responses32=bundle.raw_responses32,
        q_layers=bundle.q_layers, layer_ids=bundle.layer_ids, metric=bundle.metric,
        lambda_response=.1, request_ids=bundle.request_ids,
        frobenius_metric=bundle.frobenius_metric, include_all_row=include_all_row)
    responses, influences, summaries = [], [], []
    for view_id, view in attribution['views'].items():
        responses.extend(dict(normalization_id=view_id, **row) for row in view['request_rows'])
        influences.extend(dict(normalization_id=view_id, **row) for row in view.get('all_row_influence', []))
        summaries.append(dict(normalization_id=view_id, lambda_response=.1,
            V=view['V'], gain=view['gain'], qN=view['qN'], response_sq=view['response_sq'],
            coefficients_observed=view['coefficients_observed'], **view['summary']))
    cross_scores = {name: cross_objective_scores(bundle.target32, terminal, source)
                    for name, terminal in bundle.endpoint_terminals.items()}
    # Entry is labelled as entry, never substituted for a missing final endpoint.
    entry_score = cross_objective_scores(bundle.target32, bundle.entry_captures32[0], source)
    bundle.assert_frozen()
    body = dict(status='CPU_FROZEN_RAW_ATTRIBUTION_COMPLETE',
        input_member_hashes=deepcopy(bundle._member_hashes),
        input_metadata_sha256=bundle._metadata_sha256, provenance=deepcopy(bundle.evidence),
        external_model_state_provenance_independently_validated_here=False,
        availability=deepcopy(availability), S_blocked_by_D=False,
        capture_repeatability=repeatability, entry_cross_objective=entry_score,
        endpoint_cross_objective=cross_scores,
        endpoint_status='PROVIDED_ACTUAL_CAPTURE_CROSS_SCORED' if cross_scores else 'NOT_RECORDED_ACTUAL_ENDPOINT_CAPTURE',
        tables=dict(request_response_contributions=responses, all_row_influence=influences,
                    normalization_same_state=summaries),
        native_angle=attribution['native_angle'],
        frobenius_angle=attribution.get('frobenius_angle'),
        actual_write_count=0, model_load_count=0, model_forward_count=0,
        replay_action_count=0, controller_influence_count=0, input_mutation_count=0,
        scientific_promotion=False)
    return dict(body, receipt_identity_sha256=_canonical_sha(body))


def read_publication_availability(repo):
    """Pinned Git-only publication reader; no fetch/SSH/live-output access."""
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse',
        PUBLICATION_HEAD+'^{commit}'], text=True).strip()
    if commit != PUBLICATION_HEAD:
        raise DiagnosisInputBoundary('PUBLICATION_COMMIT_IDENTITY')
    tables, inventory = {}, []
    for name in ('checkpoint_inventory.csv', 'checkpoint_context_support.csv'):
        path = _PUBLICATION_PREFIX+name
        content = subprocess.check_output(['git', '-C', str(repo), 'show', PUBLICATION_HEAD+':'+path])
        table = list(csv.DictReader(io.StringIO(content.decode('utf-8'))))
        tables[name] = table
        inventory.append(dict(path=path, bytes=len(content), sha256=hashlib.sha256(content).hexdigest(), rows=len(table)))
    availability = historical_availability(checkpoint_rows=tables['checkpoint_inventory.csv'],
        context_rows=tables['checkpoint_context_support.csv'])
    return dict(publication_head=commit, publication_members=inventory,
                availability=availability, remote_access_count=0, git_mutation_count=0)
