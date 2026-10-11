"""Immutable C0 preparation for sequential FE; contains no target computation."""
import os
from pathlib import Path

import numpy as np
import torch

from official.baselines.fe_covariance import prepare_covariance
from official.experiments.prepare import write_new
from .server1.common import member, read, verify


def prepare(c, assets, output, *, resume=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / 'covariance-assets.json'
    prior = read(manifest) if resume else None
    if prior is not None:
        assert prior['config_sha256'] == c['config_sha256']
        assert set(prior['layers']) == set(assets['C0'])
    else:
        assert not manifest.exists(), 'FE_PRIOR_ALREADY_PREPARED_USE_RESUME'
    covariances, receipts = {}, {}
    policy = c.get('covariance_policy', 'strict')
    for layer, row in assets['C0'].items():
        source = row['member']
        verify(source)
        saved = prior['layers'][layer] if prior else None
        if saved is not None:
            assert saved['source'] == source and saved['policy'] == policy
        if saved is not None and saved['repaired']:
            cov = torch.load(verify(saved['derived']), map_location='cpu', weights_only=True)
            assert cov.dtype == torch.float64 and list(cov.shape) == saved['shape']
            assert torch.isfinite(cov).all()
            receipt = saved
        else:
            with np.load(source['path'], allow_pickle=False) as a:
                count = int(a['mom2.count'])
                assert count > 0 and count == row['validation']['masked_token_vector_count']
                assert int(a['sample_size']) == c['hparams']['mom2_n_samples']
                summed = a['mom2.mom2']
                assert summed.dtype == np.float32
                cov, receipt = prepare_covariance(torch.from_numpy(summed.copy()) / count, policy=policy)
            receipt = dict(receipt, source=source)
            if receipt['repaired']:
                folder = output / 'covariance'
                folder.mkdir(exist_ok=True)
                dest = folder / f'layer-{layer}.pt'
                temporary = dest.with_suffix('.partial')
                assert not dest.exists() and not temporary.exists()
                try:
                    with temporary.open('xb') as stream:
                        torch.save(cov, stream)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, dest)
                    _sync(folder)
                finally:
                    temporary.unlink(missing_ok=True)
                receipt['derived'] = member(dest)
            if saved is not None:
                assert receipt == saved, 'FE_PRIOR_RESUME_CHANGED'
        assert tuple(cov.shape) == (assets['model']['identity']['intermediate'],) * 2
        covariances[row['module']] = cov
        receipts[layer] = receipt
    if not resume:
        write_new(manifest, dict(config_sha256=c['config_sha256'], layers=receipts))
        with manifest.open('rb') as stream:
            os.fsync(stream.fileno())
        _sync(output)
    return covariances, receipts


def _sync(folder):
    descriptor = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
