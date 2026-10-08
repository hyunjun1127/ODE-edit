"""Verify installed source bytes and forbid dependencies on historical task code."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify():
    lock = json.loads((ROOT / "SOURCES.json").read_text())
    for row in lock["files"]:
        path = ROOT / row["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("SOURCE_CHANGED: " + row["path"])
    count = 0
    for path in ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        count += 1
        for node in ast.walk(tree):
            names = ([n.name for n in node.names] if isinstance(node, ast.Import) else
                     [node.module or ""] if isinstance(node, ast.ImportFrom) and not node.level else [])
            if any(n.startswith(("project.", "scripts.", "easyeditor.")) for n in names):
                raise ValueError("EXTERNAL_REPOSITORY_CODE_IMPORT: " + str(path))
    from official.experiments.prepare import load_plan
    load_plan()
    return dict(source_files=len(lock["files"]), python_files=count,
                source_integrity="PASS", external_task_imports=0,
                GPU_qualification="NOT_RUN")


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
