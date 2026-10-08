"""Import exact pinned frozen blobs under an isolated namespace for CPU comparisons."""
import ast
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import types

FIXTURE = json.loads((Path(__file__).parent / "fixtures/price_hparams_oracle.json").read_text())
_MODULES = {}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def load(name):
    if name in _MODULES:
        return _MODULES[name]
    module = types.ModuleType("project.run_scripts." + name)
    _MODULES[name] = module
    if name in ("jlz_realization.common", "jlz_native_writer_aware.common", "jlz_realized_subject.common"):
        # Unchanged assertion/hash utilities, no numerical solver substitutions.
        from official.ours.common import tensor_sha
        module.require, module.tensor_sha = require, tensor_sha
        return module
    row = FIXTURE["files"][name]
    assert hashlib.sha256(row["text"].encode()).hexdigest() == row["sha256"]
    if name == "jlz_interference_l1.cap_common":
        # Import no historical scheduler/task code; use literal namespace constants
        # from the pinned file for arm_profile's membership checks.
        for node in ast.parse(row["text"]).body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                key = node.targets[0].id
                if key in ("ARMS", "MODELS"):
                    setattr(module, key, ast.literal_eval(node.value))
        module.require = require
        return module
    module.__package__ = module.__name__.rsplit(".", 1)[0]

    def importer(target, globals=None, locals=None, fromlist=(), level=0):
        if level:
            target = importlib.util.resolve_name("." * level + target, globals["__package__"])
        prefix = "project.run_scripts."
        if target.startswith(prefix):
            return load(target[len(prefix):])
        return builtins.__import__(target, globals, locals, fromlist, 0)

    module.__dict__["__builtins__"] = dict(vars(builtins), __import__=importer)
    exec(compile(row["text"], f"frozen://{FIXTURE['commit']}/{row['path']}", "exec"), module.__dict__)
    return module
