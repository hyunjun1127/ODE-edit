"""Bounded S orchestration over the shared, immutable fixture and trajectory.

No model loader, Slurm submission, native solve implementation, or D replay is
present. Callers supply an already authorized model and prepared cold fixture.
Injected callbacks permit compact CPU lifecycle tests without a model backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
import math
import weakref

from project.run_scripts.native_response_ode_v31.algebra import FrozenNormalization
from project.run_scripts.native_response_ode_v31.native_binding import NativeDictionary
from project.run_scripts.ordered_response_barrier_ode.fp32_overlay import tensor_set_sha256, tensor_sha256
from .contracts import BindingBoundary
from .fixture import EntrySnapshot, FixedTargetBundle
from .normalization_views import NormalizationView
from .sweep import MODELS, endpoint_binding, endpoints, paths
from .trajectory import run_joint


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
        initial.metric_observer = reference_dictionary
        entry_evaluation = initial.evaluate_endpoint()
        _assert_entry(entry_snapshot, model, module, pointers)
        initial.metric_observer = None
        del initial, reference_dictionary, initial_terminal

        official_family = make_family()
        bundle.attach(official_family)
        official_dictionary = make_dictionary(official_family)
        _bind_reference(official_dictionary, reference)
        official_family.metric_observer = official_dictionary
        official = callbacks.official(official_family)
        _assert_entry(entry_snapshot, model, module, pointers)
        if official.get("status") != "TERMINAL_VALID":
            raise BindingBoundary("S_OFFICIAL_TERMINAL_COMPLETENESS")
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
            result = callbacks.trajectory(family, dictionary, normalization, path.config,
                output=output / path.path_id, arm=terminal_labels[0], prefixes=prefix_schedule,
                raw_sink=raw_sink, endpoint_sink=endpoint_sink)
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
            qref_inherited_into_branch_count=8, entry_restore=True,
            D_restore_dependency=False, scientific_promotion=False)
    finally:
        # The original error remains an error; cleanup is not successful replay.
        entry_snapshot.restore(model, module)
