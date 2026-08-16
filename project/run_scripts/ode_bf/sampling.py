"""Outcome-free, stateless pretrained replay schedules with separate lineages."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

from .contracts import ODEBFContractError, canonical_hash, positive_integer


class SampleLineage(str, Enum):
    CONTROLLER = "controller"
    TERMINAL_CONFIRMATION = "terminal-confirmation"
    REPORT_ONLY = "report-only"


FORBIDDEN_SOURCE_CLASSES = {
    "counterfact-efficacy",
    "counterfact-paraphrase",
    "counterfact-locality",
    "counterfact-generation",
    "zsre-efficacy",
    "zsre-paraphrase",
    "zsre-locality",
    "session03",
}


@dataclass(frozen=True, slots=True)
class PopulationItem:
    item_sha256: str
    stratum: str
    source_class: str

    def __post_init__(self) -> None:
        if len(self.item_sha256) != 64:
            raise ODEBFContractError("pretrained sample identity is not SHA-256")
        if not self.stratum or not self.source_class:
            raise ODEBFContractError("pretrained sample metadata is incomplete")
        if self.source_class.lower() in FORBIDDEN_SOURCE_CLASSES:
            raise ODEBFContractError("held-out or foreign source entered replay population")


@dataclass(frozen=True, slots=True)
class LineageSeal:
    lineage: SampleLineage
    population_sha256: str
    seed: int
    sample_count: int
    allowed_item_sha256: tuple[str, ...]
    strata: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.population_sha256) != 64:
            raise ODEBFContractError("population digest is not SHA-256")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ODEBFContractError("sampling seed is invalid")
        positive_integer("sample_count", self.sample_count)
        if len(self.allowed_item_sha256) < self.sample_count:
            raise ODEBFContractError("lineage pool is smaller than its sample count")
        if len(set(self.allowed_item_sha256)) != len(self.allowed_item_sha256):
            raise ODEBFContractError("lineage pool contains duplicates")
        if not self.strata or len(set(self.strata)) != len(self.strata):
            raise ODEBFContractError("lineage strata are empty or repeated")

    def identity(self) -> str:
        return canonical_hash(
            {
                "lineage": self.lineage.value,
                "population_sha256": self.population_sha256,
                "seed": self.seed,
                "sample_count": self.sample_count,
                "allowed_item_sha256": list(self.allowed_item_sha256),
                "strata": list(self.strata),
            }
        )


@dataclass(frozen=True, slots=True)
class SamplingSeal:
    population_sha256: str
    items: tuple[PopulationItem, ...]
    lineages: tuple[LineageSeal, ...]

    def __post_init__(self) -> None:
        if len(self.population_sha256) != 64:
            raise ODEBFContractError("population digest is not SHA-256")
        identities = tuple(item.item_sha256 for item in self.items)
        if len(set(identities)) != len(identities):
            raise ODEBFContractError("sampling population contains duplicates")
        by_identity = {item.item_sha256: item for item in self.items}
        if len(self.lineages) != len(SampleLineage) or {
            lineage.lineage for lineage in self.lineages
        } != set(SampleLineage):
            raise ODEBFContractError("controller/terminal/report lineages are all required")
        pools: list[set[str]] = []
        for lineage in self.lineages:
            if lineage.population_sha256 != self.population_sha256:
                raise ODEBFContractError("lineage population identity differs")
            if not set(lineage.allowed_item_sha256).issubset(by_identity):
                raise ODEBFContractError("lineage refers to an absent population item")
            item_strata = {by_identity[item].stratum for item in lineage.allowed_item_sha256}
            if not set(lineage.strata).issubset(item_strata):
                raise ODEBFContractError("lineage stratum has no eligible item")
            pools.append(set(lineage.allowed_item_sha256))
        if any(pools[left].intersection(pools[right]) for left in range(3) for right in range(left + 1, 3)):
            raise ODEBFContractError("controller/terminal/report lineage pools must be disjoint")

    def identity(self) -> str:
        return canonical_hash(
            {
                "population_sha256": self.population_sha256,
                "items": [
                    {
                        "item_sha256": item.item_sha256,
                        "stratum": item.stratum,
                        "source_class": item.source_class,
                    }
                    for item in self.items
                ],
                "lineages": [lineage.identity() for lineage in self.lineages],
            }
        )


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    lineage: SampleLineage
    outer_batch_index: int
    correction_cycle: int
    waypoint: int
    replay_batch_id: int
    item_sha256: tuple[str, ...]
    schedule_digest: str


class StatelessReplaySchedule:
    """A pure schedule: reject/accept cannot mutate a cursor or RNG state."""

    def __init__(self, seal: SamplingSeal) -> None:
        self.seal = seal
        self._lineages = {item.lineage: item for item in seal.lineages}
        self._items = {item.item_sha256: item for item in seal.items}
        self._state_digest = canonical_hash({"seal": seal.identity(), "mutable_state": None})

    @property
    def state_digest(self) -> str:
        return self._state_digest

    @staticmethod
    def _validate_index(name: str, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ODEBFContractError(f"{name} must be a nonnegative integer")
        return value

    def batch(
        self,
        lineage: SampleLineage,
        *,
        outer_batch_index: int,
        correction_cycle: int,
        waypoint: int,
        replay_batch_id: int,
    ) -> ReplayBatch:
        indices = {
            "outer_batch_index": self._validate_index("outer_batch_index", outer_batch_index),
            "correction_cycle": self._validate_index("correction_cycle", correction_cycle),
            "waypoint": self._validate_index("waypoint", waypoint),
            "replay_batch_id": self._validate_index("replay_batch_id", replay_batch_id),
        }
        sealed = self._lineages[lineage]
        by_stratum: dict[str, list[str]] = {stratum: [] for stratum in sealed.strata}
        for identity in sealed.allowed_item_sha256:
            item = self._items[identity]
            if item.stratum in by_stratum:
                by_stratum[item.stratum].append(identity)
        context = canonical_hash(
            {
                "lineage_identity": sealed.identity(),
                **indices,
            }
        )
        ordered: list[str] = []
        # Deterministic stratified round-robin.  Arm identity is deliberately absent,
        # which gives matched arms common random numbers.
        ranked = {
            stratum: sorted(
                values,
                key=lambda identity: hashlib.sha256(
                    f"{sealed.seed}|{context}|{stratum}|{identity}".encode("utf-8")
                ).hexdigest(),
            )
            for stratum, values in by_stratum.items()
        }
        offset = 0
        while len(ordered) < sealed.sample_count:
            made_progress = False
            for stratum in sealed.strata:
                values = ranked[stratum]
                if offset < len(values):
                    ordered.append(values[offset])
                    made_progress = True
                    if len(ordered) == sealed.sample_count:
                        break
            if not made_progress:
                raise ODEBFContractError("sealed replay lineage cannot fill its batch")
            offset += 1
        payload = {
            "lineage": lineage.value,
            **indices,
            "items": ordered,
            "seal": sealed.identity(),
        }
        return ReplayBatch(
            lineage,
            indices["outer_batch_index"],
            indices["correction_cycle"],
            indices["waypoint"],
            indices["replay_batch_id"],
            tuple(ordered),
            canonical_hash(payload),
        )


def assert_cross_arm_common_random_numbers(
    batches_by_arm: Mapping[str, ReplayBatch],
) -> str:
    if len(batches_by_arm) < 2:
        raise ODEBFContractError("matched replay identity needs at least two arms")
    digests = {batch.schedule_digest for batch in batches_by_arm.values()}
    item_orders = {batch.item_sha256 for batch in batches_by_arm.values()}
    if len(digests) != 1 or len(item_orders) != 1:
        raise ODEBFContractError("matched arms received different replay batches")
    return next(iter(digests))


def load_cpu_sampling_seal(path: str) -> SamplingSeal:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    observed = value.pop("root_digest", None)
    if observed != canonical_hash(value):
        raise ODEBFContractError("CPU sampling seal root digest differs")
    if (
        value.get("schema_version")
        != "ode-edit-s04-ode-bf-p0-cpu-sampling-seal/v1"
        or value.get("p1_decision_eligible") is not False
        or value.get("heldout_counterfact_or_zsre_items") != 0
    ):
        raise ODEBFContractError("CPU sampling seal scope differs")
    identities = tuple(value.get("items", ()))
    if len(identities) != 30 or canonical_hash(list(identities)) != value.get("population_sha256"):
        raise ODEBFContractError("CPU sampling population identity differs")
    strata = tuple(value.get("strata", ()))
    items = tuple(
        PopulationItem(identity, strata[index % len(strata)], value["source_class"])
        for index, identity in enumerate(identities)
    )
    lineages: list[LineageSeal] = []
    for lineage in SampleLineage:
        spec = value["lineages"].get(lineage.value)
        if not isinstance(spec, dict):
            raise ODEBFContractError("CPU sampling lineage is absent")
        start = int(spec["pool_start"])
        end = int(spec["pool_end"])
        lineages.append(
            LineageSeal(
                lineage,
                value["population_sha256"],
                int(spec["seed"]),
                int(spec["sample_count"]),
                identities[start:end],
                strata,
            )
        )
    return SamplingSeal(value["population_sha256"], items, tuple(lineages))


def load_p1_sampling_seal(
    population_path: str | Path,
    *,
    stream_path: str | Path,
) -> SamplingSeal:
    """Load the sealed 160-item P1 population without opening prompt fields."""

    from .p1_selection import (
        verify_p1_population_seal,
        verify_p1_stream_seal,
    )

    stream_value = json.loads(Path(stream_path).read_text(encoding="utf-8"))
    stream = verify_p1_stream_seal(stream_value)
    population_value = json.loads(Path(population_path).read_text(encoding="utf-8"))
    value = verify_p1_population_seal(population_value, stream=stream)
    items = tuple(
        PopulationItem(
            str(item["request_sha256"]),
            str(item["stratum"]),
            str(value["source_class"]),
        )
        for item in value["items"]
    )
    by_identity = {item.item_sha256: item for item in items}
    lineages: list[LineageSeal] = []
    for lineage in SampleLineage:
        spec = value["lineages"].get(lineage.value)
        if not isinstance(spec, Mapping):
            raise ODEBFContractError("P1 sampling lineage is absent")
        allowed = tuple(str(item) for item in spec["allowed_item_sha256"])
        strata = tuple(
            sorted({by_identity[identity].stratum for identity in allowed})
        )
        lineages.append(
            LineageSeal(
                lineage,
                str(value["population_sha256"]),
                int(spec["seed"]),
                int(spec["sample_count"]),
                allowed,
                strata,
            )
        )
    return SamplingSeal(str(value["population_sha256"]), items, tuple(lineages))
