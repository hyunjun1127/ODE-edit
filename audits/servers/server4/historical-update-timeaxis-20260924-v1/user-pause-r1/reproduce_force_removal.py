"""Read-only CPU reproducer of a frozen-source routing defect; no model imports."""
import ast
import contextlib
import json
import types
from pathlib import Path


def main():
    repo = Path(__file__).resolve().parents[5]
    source = repo / "project/run_scripts/historical_update_timeaxis/backend.py"
    tree = ast.parse(source.read_text())
    backend = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Backend")
    method = next(n for n in backend.body if isinstance(n, ast.FunctionDef) and n.name == "state")
    namespace = {"contextlib": contextlib}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    calls = []
    fake = types.SimpleNamespace(endpoint=lambda t: calls.append(t))
    try:
        with namespace["state"](fake, {"kind": "ACTUAL", "actual_checkpoint": 0}, force_removal=(1, [0])):
            raise AssertionError("Unexpected success")
    except KeyError as exc:
        assert exc.args == ("construction_endpoint",), exc
        assert not calls
        print(json.dumps({"defect_reproduced": True, "exception": "KeyError: construction_endpoint", "endpoint_calls": 0, "model_loads": 0, "gpu_calls": 0}))
    else:
        raise AssertionError("Expected routing defect not reproduced")


if __name__ == "__main__":
    main()
