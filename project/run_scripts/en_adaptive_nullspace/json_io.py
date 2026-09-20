"""Strict compact receipts: scalar conversion only, atomic create-once publication."""
import json
import os
from pathlib import Path
import tempfile
import numpy as np


def scalar(value):
    if isinstance(value, np.generic):
        item = value.item()
        if type(item) in (bool, int, float, str) or item is None:
            return item
    raise TypeError(f"Unsupported compact JSON type: {type(value).__name__}")


def save(path, value):
    # Validate everything before creating any output; never serialize tensor state.
    encoded = json.dumps(value, ensure_ascii=False, indent=2,
                         allow_nan=False, default=scalar).encode('utf-8')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic no-replace publication on same filesystem.
    finally:
        os.unlink(temporary)  # Only this function's newly owned temporary file.
