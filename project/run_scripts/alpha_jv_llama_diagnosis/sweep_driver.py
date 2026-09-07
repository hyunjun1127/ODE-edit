"""Bounded S orchestration over the shared, immutable fixture and trajectory.

No model loader, Slurm submission, native solve implementation, or D replay is
present. Callers supply an already authorized model and prepared cold fixture.
Injected callbacks permit compact CPU lifecycle tests without a model backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Mapping
import math
import weakref

from project.run_scripts.native_response_ode_v31.algebra import FrozenNormalization
from project.run_scripts.native_response_ode_v31.native_binding import NativeDictionary
from project.run_scripts.ordered_response_barrier_ode.fp32_overlay import tensor_set_sha256, tensor_sha256
from .contracts import BindingBoundary
from .fixture import EntrySnapshot, FixedTargetBundle, RNGSnapshot
from .normalization_views import NormalizationView
from .sweep import MODELS, endpoint_binding, endpoints, paths
from .trajectory import run_joint
from .raw_store import tree_identity


def _normalization(target, terminal, entry_sha):
    return NormalizationView.from_source(FrozenNormalization.capture(target, terminal, entry_sha), "N0_SOURCE")


def _official(family):
    # This public path evaluates its active endpoint, then restores on return.
    return family.run_official(fixed_z=family.fixed_z)


@dataclass(frozen=True)
class SweepCallbacks:
    dictionary_factory: Callable = NativeDictionary
    target_bundle_capture: Callable = FixedTargetBundle.capture
    normalization_factory: Callable = _normalization
    trajectory: Callable = run_joint
    official: Callable = _official
    first_fidelity: Callable | None = None
    fixed_target_sink: Callable | None = None
    component_sink: Callable | None = None


def _assert_entry(snapshot: EntrySnapshot, model, module, pointers):
    snapshot.assert_sealed()
    current = {name: model.get_parameter(name) for name in snapshot.weights}
    if (tensor_set_sha256(current) != snapshot.W_sha256
            or tensor_sha256(module.cache_c) != snapshot.M_sha256
            or bool(module.cache_c_new) != snapshot.cache_c_new
            or {name: value.data_ptr() for name, value in current.items()} != pointers):
        raise BindingBoundary("S_RETURNED_WITHOUT_ENTRY_RESTORE")


def _bind_reference(dictionary, reference):
    if dictionary.qref is not None or dictionary.qfref is not None:
        raise BindingBoundary("S_DICTIONARY_NOT_FRESH")
    dictionary.qref, dictionary.qfref = reference


@contextmanager
def _sink_guard(family, snapshot, model, module, pointers, *, bundle=None, normalization=None,
                dictionary=None, temporary_fidelity=False):
    """Persistence/telemetry callbacks do not acquire scientific authority."""
    rng = RNGSnapshot()
    versions = {name: value._version for name, value in family.parameters.items()}
    def identity():
        target = getattr(family, "fixed_z", None)
        bound = dict(target_values=target.values if target is not None else None,
                     target_identity=getattr(target, "identity_sha256", None),
                     semantic=getattr(family, "semantic_inventory", None),
                     contexts=getattr(family, "contexts", None),
                     request_order=getattr(family, "request_order_sha256", None))
        if bundle is not None:
            bound["bundle"] = dict(values=bundle.artifact.values,
                identity=bundle.artifact.identity_sha256,
                semantic=getattr(bundle, "semantic_inventory", getattr(bundle, "semantic", None)),
                contexts=getattr(bundle, "contexts_sha256", None))
        if dictionary is not None:
            bound["qref"] = (dictionary.qref, dictionary.qfref)
        if normalization is not None:
            bound["normalization"] = normalization.receipt()
        return tree_identity(bound)
    before = identity()
    try:
        yield
        _assert_entry(snapshot, model, module, pointers)
        if (identity() != before or not rng.matches()
                or (not temporary_fidelity and versions != {name: value._version for name, value in family.parameters.items()})):
            raise BindingBoundary("S_OBSERVER_BINDING_MUTATION")
    finally:
        rng.restore()


def run_model_sweep(*, model_alias: str, model, module, entry_snapshot: EntrySnapshot,
                    family_factory: Callable, family_kwargs: Mapping[str, Any], output: Path,
                    callbacks: SweepCallbacks | None = None, raw_sink=None, endpoint_sink=None):
    """Produce nine S endpoints through seven actual independent joint paths.

    The same prepared cold W/M and fixed-z bundle are used by Official and all
    paths. Prefix finalization is delegated to run_joint's guarded observation
    scope; no writer or evaluator is rerun here to reconstruct a returned state.
    This function neither assigns nor interprets a GPU-hour authorization.
    """
    callbacks = callbacks or SweepCallbacks()
    if model_alias not in MODELS:
        raise BindingBoundary("S_MODEL_MAPPING")
    if "model" in family_kwargs or "module" in family_kwargs:
        raise BindingBoundary("S_FACTORY_ARGUMENT_OVERRIDE")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    pointers = {name: model.get_parameter(name).data_ptr() for name in entry_snapshot.weights}
    family_refs = []
    dictionary_refs = []
    family_count = 0

    def emit(component, phase, payload, family):
        if callbacks.component_sink is not None:
            before = tree_identity(payload)
            with _sink_guard(family, entry_snapshot, model, module, pointers):
                callbacks.component_sink(dict(component=component, phase=phase, model=model_alias), payload)
            if tree_identity(payload) != before:
                raise BindingBoundary("S_COMPONENT_RECEIPT_MUTATION")

    def make_family():
        nonlocal family_count
        family = entry_snapshot.make_family(family_factory, model=model, module=module, **dict(family_kwargs))
        if any(family is previous() for previous in family_refs):
            raise BindingBoundary("S_MUTABLE_FAMILY_REUSED")
        family_refs.append(weakref.ref(family))
        family_count += 1
        return family

    def make_dictionary(family):
        dictionary = callbacks.dictionary_factory(family)
        if dictionary.family is not family or any(dictionary is previous() for previous in dictionary_refs):
            raise BindingBoundary("S_MUTABLE_DICTIONARY_REUSED")
        dictionary_refs.append(weakref.ref(dictionary))
        return dictionary

    try:
        initial = make_family()
        emit("fixed_target_and_entry", "begin", {}, initial)
        initial.compute_fixed_z()
        bundle = callbacks.target_bundle_capture(initial)
        initial_terminal = initial.terminal()
        normalization = callbacks.normalization_factory(bundle.artifact.values, initial_terminal,
                                                        entry_snapshot.W_sha256)
        normalization.assert_frozen()
        if normalization.entry_sha != entry_snapshot.W_sha256 or normalization.normalization_id != "N0_SOURCE":
            raise BindingBoundary("S_NORMALIZATION_BINDING")
        reference_dictionary = make_dictionary(initial)
        builds = reference_dictionary.build(initial_terminal, 0)
        if len(builds) != 5:
            raise BindingBoundary("S_ENTRY_FIVE_DIRECTION_REFERENCE")
        reference_receipt = reference_dictionary.capture_reference(builds)
        reference = (reference_dictionary.qref, reference_dictionary.qfref)
        if not all(value is not None and math.isfinite(value) and value >= 0 for value in reference):
            raise BindingBoundary("S_ENTRY_NATIVE_REFERENCE_INVALID")
        del builds
        if callbacks.fixed_target_sink is not None:
            receipt_before = tree_identity(reference_receipt)
            with _sink_guard(initial, entry_snapshot, model, module, pointers, bundle=bundle,
                             normalization=normalization, dictionary=reference_dictionary):
                callbacks.fixed_target_sink(bundle, entry_snapshot, normalization, reference_receipt)
            if tree_identity(reference_receipt) != receipt_before:
                raise BindingBoundary("S_FIXED_REFERENCE_RECEIPT_MUTATION")
        fidelity = None
        if callbacks.first_fidelity is not None:
            emit("first_fidelity", "begin", {}, initial)
            with _sink_guard(initial, entry_snapshot, model, module, pointers, bundle=bundle,
                             normalization=normalization, dictionary=reference_dictionary, temporary_fidelity=True):
                fidelity = callbacks.first_fidelity(initial, reference_dictionary, normalization)
            emit("first_fidelity", "end", fidelity, initial)
        initial.metric_observer = reference_dictionary
        entry_evaluation = initial.evaluate_endpoint()
        _assert_entry(entry_snapshot, model, module, pointers)
        emit("fixed_target_and_entry", "end", dict(reference=reference_receipt, normalization=normalization.receipt(),
                entry_evaluation=entry_evaluation,
                target_generation_accounting=getattr(initial, "target_generation_accounting", None),
                evaluation_seconds=getattr(initial, "evaluation_seconds", None)), initial)
        initial.metric_observer = None
        del initial, reference_dictionary, initial_terminal

        official_family = make_family()
        bundle.attach(official_family)
        official_dictionary = make_dictionary(official_family)
        _bind_reference(official_dictionary, reference)
        official_family.metric_observer = official_dictionary
        emit("O_NATIVE", "begin", {}, official_family)
        official = callbacks.official(official_family)
        _assert_entry(entry_snapshot, model, module, pointers)
        if official.get("status") != "TERMINAL_VALID":
            raise BindingBoundary("S_OFFICIAL_TERMINAL_COMPLETENESS")
        if endpoint_sink is not None:
            weights = getattr(official_family, "_captured_endpoint_weights", None)
            method_tensor = getattr(official_family, "_captured_endpoint_method_state", None)
            method_flag = getattr(official_family, "_captured_endpoint_cache_c_new", None)
            if weights is None or method_tensor is None or not isinstance(method_flag, bool):
                raise BindingBoundary("S_OFFICIAL_ENDPOINT_CAPTURE_REQUIRED")
            method_state = dict(alpha_cache=method_tensor, cache_c_new=method_flag)
            captured_before = tree_identity((weights, method_state, official_family.last_terminal))
            with _sink_guard(official_family, entry_snapshot, model, module, pointers, bundle=bundle,
                             normalization=normalization, dictionary=official_dictionary):
                endpoint_sink("O_NATIVE", "O_NATIVE", official, weights, official_family.last_terminal,
                              entry_snapshot, method_state, ())
            if tree_identity((weights, method_state, official_family.last_terminal)) != captured_before:
                raise BindingBoundary("S_ENDPOINT_SINK_MUTATION")
        emit("O_NATIVE", "end", dict(endpoint=official, actual_physical=getattr(official_family, "last_physical_action", None),
                                     evaluation_seconds=getattr(official_family, "evaluation_seconds", None)), official_family)
        official_family.metric_observer = None
        del official_family, official_dictionary

        results, endpoint_results = [], {}
        specs = {row.candidate_id: row for row in endpoints()}
        for path in paths():
            family = make_family()
            bundle.attach(family)
            dictionary = make_dictionary(family)
            _bind_reference(dictionary, reference)
            family.metric_observer = dictionary
            normalization.assert_frozen()
            path_endpoints = [specs[label] for label in path.endpoint_ids]
            prefix_schedule = {row.completed_nodes: row.candidate_id for row in path_endpoints
                               if row.derived_observation_only}
            terminal_labels = [row.candidate_id for row in path_endpoints if not row.derived_observation_only]
            if len(terminal_labels) != 1:
                raise BindingBoundary("S_PATH_TERMINAL_LABEL")
            def path_raw_sink(node, payload):
                if raw_sink is not None:
                    raw_sink(path.path_id, node, payload)
            def path_endpoint_sink(label, endpoint, weights, activation, increments=()):
                if endpoint_sink is not None:
                    prefix = specs[label].derived_observation_only
                    method_tensor = entry_snapshot.alpha_cache if prefix else getattr(family, "_captured_endpoint_method_state", None)
                    method_flag = entry_snapshot.cache_c_new if prefix else getattr(family, "_captured_endpoint_cache_c_new", None)
                    if method_tensor is None or not isinstance(method_flag, bool):
                        raise BindingBoundary("S_TERMINAL_ENDPOINT_CAPTURE_REQUIRED")
                    method_state = dict(alpha_cache=method_tensor, cache_c_new=method_flag)
                    captured_before = tree_identity((weights, method_state, activation, increments))
                    endpoint_sink(path.path_id, label, endpoint, weights, activation,
                                  entry_snapshot, method_state, increments)
                    if tree_identity((weights, method_state, activation, increments)) != captured_before:
                        raise BindingBoundary("S_ENDPOINT_SINK_MUTATION")
            emit(path.path_id, "begin", dict(config=path.config.receipt()), family)
            result = callbacks.trajectory(family, dictionary, normalization, path.config,
                output=output / path.path_id, arm=terminal_labels[0], prefixes=prefix_schedule,
                raw_sink=path_raw_sink if raw_sink is not None else None,
                endpoint_sink=path_endpoint_sink if endpoint_sink is not None else None)
            # Do not repair missing restore and report PASS. Check before cleanup.
            _assert_entry(entry_snapshot, model, module, pointers)
            normalization.assert_frozen()
            if (dictionary.qref, dictionary.qfref) != reference:
                raise BindingBoundary("S_ENTRY_REFERENCE_CHANGED")
            if (result.get("status") != "TERMINAL_VALID" or len(result.get("nodes", ())) != path.config.N
                    or set(result["endpoints"]) != set(path.endpoint_ids)):
                raise BindingBoundary("S_PATH_ENDPOINT_COMPLETENESS")
            for endpoint in path_endpoints:
                label = endpoint.candidate_id
                if label in endpoint_results:
                    raise BindingBoundary("S_DUPLICATE_ENDPOINT")
                observation = result["endpoints"][label]
                binding = endpoint_binding(endpoint, path)
                for key in ("effective_T", "effective_N", "h", "parent_T", "parent_N"):
                    if observation.get(key) != binding[key]:
                        raise BindingBoundary("S_ENDPOINT_CLOCK_MISMATCH")
                if endpoint.derived_observation_only and (
                        observation.get("history_append_count") != 0
                        or observation.get("persistent_endpoint_capture_count") != 0):
                    raise BindingBoundary("S_PREFIX_LIFECYCLE_MISMATCH")
                endpoint_results[label] = dict(observation=observation, binding=binding)
            results.append(dict(path_id=path.path_id, result=result))
            emit(path.path_id, "end", dict(result=result, actual_physical=getattr(family, "last_physical_action", None),
                evaluation_seconds=getattr(family, "evaluation_seconds", None)), family)
            family.metric_observer = None
            del family, dictionary
        if len(endpoint_results) != 9 or len(results) != 7:
            raise BindingBoundary("S_NINE_ENDPOINT_COMPLETENESS")
        return dict(status="S_NINE_ENDPOINTS_COMPLETE", model=model_alias,
            fixed_z_bundle_sha256=bundle.artifact.identity_sha256,
            entry_W_sha256=entry_snapshot.W_sha256, entry_M_sha256=entry_snapshot.M_sha256,
            normalization=normalization.receipt(), reference=reference_receipt,
            reference_capture_count=1, fixed_z_capture_count=1, fixed_z_recompute_count=0,
            family_construction_count=family_count, independent_joint_path_count=len(results),
            JV_endpoint_count=len(endpoint_results), entry_evaluation=entry_evaluation,
            Official_endpoint=official, endpoints=endpoint_results, trajectories=results,
            first_fidelity=fidelity,
            qref_inherited_into_branch_count=8, entry_restore=True,
            D_restore_dependency=False, scientific_promotion=False)
    finally:
        # The original error remains an error; cleanup is not successful replay.
        entry_snapshot.restore(model, module)
