"""Resolve one immutable PRICE configuration without importing torch or a tracker."""
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType

ROOT = Path(__file__).resolve().parents[1] / "hparams" / "ours"
MODELS = ("llama3", "qwen25", "gptj")


def plain(value):
    """A detached JSON-compatible copy for receipts and W&B config."""
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def canonical_sha256(value):
    data = json.dumps(plain(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


@dataclass(frozen=True, slots=True)
class ResolvedConfig(Mapping):
    _data: Mapping

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


def require_config(config):
    if not isinstance(config, ResolvedConfig):
        raise TypeError("PRICE_REQUIRES_RESOLVED_CONFIG: use official.ours.config.resolve")
    return config


def metadata(config):
    require_config(config)
    return dict(config_sha256=config["config_sha256"], config_arm=config["arm"],
                resolved_config=plain(config))


def wandb_config(config):
    """Pass to wandb.init(config=...) or to the fit wandb_run connector."""
    return {"ours": metadata(config)}


def bind_fit(adapter, supplied=None, wandb_run=None):
    config = require_config(adapter.profile)
    if supplied is not None and require_config(supplied) != config:
        raise ValueError("PRICE_ADAPTER_FIT_CONFIG_MISMATCH")
    if wandb_run is not None:
        wandb_run.config.update(wandb_config(config), allow_val_change=False)
    return config


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("DUPLICATE_CONFIG_KEY: " + key)
        value[key] = item
    return value


def _read(path):
    data = path.read_bytes()
    value = json.loads(data, object_pairs_hook=_object)
    return value, hashlib.sha256(data).hexdigest()


def _validate(value, rule, location):
    """Validate the JSON Schema keywords used in the bundled schema (stdlib only)."""
    kinds = rule.get("type")
    matches = dict(object=isinstance(value, dict), array=isinstance(value, list),
                   number=type(value) in (int, float), integer=type(value) is int,
                   string=isinstance(value, str), boolean=type(value) is bool,
                   null=value is None)
    if kinds and not any(matches[k] for k in ([kinds] if isinstance(kinds, str) else kinds)):
        raise ValueError("CONFIG_TYPE: " + location)
    if "enum" in rule and value not in rule["enum"]:
        raise ValueError("CONFIG_ENUM: " + location)
    if "const" in rule and (type(value) is not type(rule["const"]) or value != rule["const"]):
        raise ValueError("CONFIG_FIXED_SEMANTICS: " + location)
    if type(value) in (int, float):
        if not math.isfinite(value):
            raise ValueError("CONFIG_FINITE: " + location)
        for key, invalid in (("minimum", lambda x: value < x),
                             ("exclusiveMinimum", lambda x: value <= x),
                             ("maximum", lambda x: value > x),
                             ("exclusiveMaximum", lambda x: value >= x)):
            if key in rule and invalid(rule[key]):
                raise ValueError("CONFIG_RANGE: " + location)
    if isinstance(value, str):
        if len(value) < rule.get("minLength", 0) or (
                "pattern" in rule and not re.fullmatch(rule["pattern"], value)):
            raise ValueError("CONFIG_STRING: " + location)
    if isinstance(value, list):
        if len(value) < rule.get("minItems", 0) or len(value) > rule.get("maxItems", len(value)):
            raise ValueError("CONFIG_ARRAY_LENGTH: " + location)
        if rule.get("uniqueItems") and len({canonical_sha256(v) for v in value}) != len(value):
            raise ValueError("CONFIG_UNIQUE_ITEMS: " + location)
        for i, item in enumerate(value):
            _validate(item, rule.get("items", {}), f"{location}[{i}]")
    if isinstance(value, dict):
        properties = rule.get("properties", {})
        if set(rule.get("required", [])) - value.keys():
            raise ValueError("CONFIG_MISSING_KEYS: " + location)
        if rule.get("additionalProperties") is False and value.keys() - properties.keys():
            raise ValueError("CONFIG_UNKNOWN_KEYS: " + location)
        for key, item in value.items():
            _validate(item, properties.get(key, {}), location + "." + key)


def resolve(model, arm=None, *, root=ROOT):
    """Load an independent writer snapshot plus explicit PRICE values and one arm.

    Baseline files and deprecated PRICE files are never read. K_eval and legacy
    aliases are derived, so callers cannot tune them independently.
    """
    if model not in MODELS or (arm is not None and
            (not isinstance(arm, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", arm))):
        raise ValueError("CONFIG_MODEL_OR_ARM")
    root = Path(root)
    schema, schema_sha = _read(root / "schema.json")
    writer, writer_sha = _read(root / model / "writer.json")
    price, price_sha = _read(root / model / "price.json")
    _validate(writer, schema["$defs"]["writer"], "writer")
    _validate(price, schema["$defs"]["price"], "price")
    if price["model"] != model:
        raise ValueError("CONFIG_MODEL_BINDING")
    sources = dict(schema_sha256=schema_sha, writer_file_sha256=writer_sha,
                   price_file_sha256=price_sha, arm_file_sha256=None)
    if arm is not None:
        patch, sources["arm_file_sha256"] = _read(root / "arms" / (arm + ".json"))
        _validate(patch, schema["$defs"]["arm"], "arm")
        if patch["base"] != model:
            raise ValueError("CONFIG_ARM_MODEL_MISMATCH")
        price.update(patch["override"])
        _validate(price, schema["$defs"]["price"], "resolved_price")
    if price["K_grace"] >= price["max_updates"]:
        raise ValueError("CONFIG_GRACE_MUST_PRECEDE_MAX_UPDATES")
    if price["cap_mode"] == "native" and (price["c"] is None or price["c"] <= 0):
        raise ValueError("CONFIG_NATIVE_CAP_POSITIVE")
    hp = writer["hparams"]
    binding = schema["x-model-bindings"][model]
    if hp["model_name"] != binding["model_name"] or any(
            price[key] != binding[key] for key in
            ("model_type", "model_profile", "expected_hidden", "expected_intermediate")):
        raise ValueError("CONFIG_WRITER_ARCHITECTURE_BINDING")
    layers = hp["layers"]
    if layers != sorted(layers) or hp["v_loss_layer"] < layers[-1]:
        raise ValueError("CONFIG_WRITER_LAYER_ORDER")
    result = dict(price, writer="memit", writer_hparams=hp, writer_source=writer["source"],
                  eligible_layers=layers, anchor_layer=layers[-1], native_anchor_layer=layers[-1],
                  nll_layer=hp["v_loss_layer"], nll_readout_layer=hp["v_loss_layer"],
                  lambda_C=hp["mom2_update_weight"], K_eval=price["max_updates"] + 1,
                  beta_max_native_scale=price["beta_max_scale"],
                  configuration_sources=sources, arm=arm or price["arm"])
    if result['model_type'] in ('qwen2', 'gptj'):
        result['price_m1_anchor_guard'] = False
        result['subject_position_policy'] = 'LOOKUP_ZERO_DOCUMENT_PREFIX'
    result["config_sha256"] = canonical_sha256(result)
    return ResolvedConfig(_freeze(result))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--arm")
    args = parser.parse_args()
    print(json.dumps(plain(resolve(args.model, args.arm)), indent=2, sort_keys=True))
