"""Offline, explicit repair of a defective native FE moment-cache asset.

The author runner and YAML do not import this module. A separate experiment may
point its native cache path to the derived NPZ, with this receipt in its identity.
This is numerical regularization of stored moments, not recomputed statistics.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from official.baselines.fe_covariance import prepare_covariance
from official.experiments.prepare import file_sha, write_new


def derive(source, destination, *, expected_sha256):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError('NATIVE_COVARIANCE_NEW_ASSET_REQUIRED')
    source_sha = file_sha(source)
    if source_sha != expected_sha256:
        raise ValueError('NATIVE_COVARIANCE_SOURCE_SHA')
    with np.load(source, allow_pickle=False) as stored:
        arrays = {key: stored[key].copy() for key in stored.files}
    count = int(arrays['mom2.count'])
    summed = arrays['mom2.mom2']
    if count <= 0 or summed.dtype != np.float32:
        raise ValueError('NATIVE_COVARIANCE_SOURCE_FORMAT')
    # Exactly match native SecondMoment.moment() before repairing.
    original = torch.from_numpy(summed) / count
    corrected, repair = prepare_covariance(original, policy='project_roundoff')
    if not repair['repaired']:
        raise ValueError('NATIVE_COVARIANCE_HEALTHY_KEEP_ORIGINAL')
    arrays['mom2.mom2'] = (corrected * count).numpy()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.partial')
    receipt_path = destination.with_suffix('.json')
    if temporary.exists() or receipt_path.exists():
        raise ValueError('NATIVE_COVARIANCE_OUTPUT_EXISTS')
    try:
        with temporary.open('xb') as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        with np.load(temporary, allow_pickle=False) as stored:
            if set(stored.files) != set(arrays):
                raise ValueError('NATIVE_COVARIANCE_METADATA_KEYS')
            for key in arrays:
                if key != 'mom2.mom2' and not np.array_equal(stored[key], arrays[key]):
                    raise ValueError('NATIVE_COVARIANCE_METADATA_CHANGED')
            reloaded = torch.from_numpy(stored['mom2.mom2'].copy()) / int(stored['mom2.count'])
        difference = float((reloaded - corrected).abs().max())
        if reloaded.dtype != torch.float64 or not torch.isfinite(reloaded).all():
            raise ValueError('NATIVE_COVARIANCE_ROUNDTRIP_DTYPE_FINITE')
        factor, info = torch.linalg.cholesky_ex(reloaded)
        del factor
        if int(info) != 0:
            raise ValueError('NATIVE_COVARIANCE_ROUNDTRIP_NOT_POSITIVE_DEFINITE')
        if file_sha(source) != source_sha:
            raise ValueError('NATIVE_COVARIANCE_SOURCE_CHANGED')
        # Exclusive publication: never overwrite another prepared asset.
        os.link(temporary, destination)
        receipt = dict(schema='native-fe-derived-moment-v1',
            scope='Numerical regularization of defective stored moments; not fresh covariance sampling',
            source=dict(path=str(source), sha256=source_sha, bytes=source.stat().st_size),
            derived=dict(path=str(destination), sha256=file_sha(destination), bytes=destination.stat().st_size),
            count=count, unchanged_metadata=True, native_normalization_dtype='float64',
            max_roundtrip_difference=difference, roundtrip_cholesky_info=int(info), repair=repair)
        write_new(receipt_path, receipt)
        descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return receipt
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=args.threads):
        result = derive(args.source, args.destination, expected_sha256=args.expected_sha256)
    print(json.dumps(result), flush=True)
