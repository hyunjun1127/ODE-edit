"""Optional, computation-neutral milestone snapshots for native FE main.py.

This module performs no optimization, evaluation, covariance repair, or resume.
Saved weights and history retain their native dtype. It uses no ODE runner.
"""
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(cfg, model, cache_c, contexts, completed_batches, requests):
    folder = cfg.get('diagnostic_checkpoint_dir')
    if not folder or completed_batches % 5:
        return
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f'B{completed_batches:02d}.pt'
    partial = destination.with_suffix('.pt.partial')
    if destination.exists() or partial.exists():
        raise FileExistsError(destination)
    names = {f'{cfg.llms.rewrite_module_tmp.format(layer)}.weight' for layer in cfg.llms.layers}
    selected = {name: p for name, p in model.named_parameters() if name in names}
    if set(selected) != names:
        raise ValueError('MISSING_CHECKPOINT_WEIGHTS')
    versions = {name: p._version for name, p in model.named_parameters()}
    rng = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
               cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    estimated = cache_c.numel() * cache_c.element_size() + sum(p.numel()*p.element_size() for p in selected.values()) + 16*1024**2
    reserve = int(cfg.get('diagnostic_reserve_bytes', 64*1024**3))
    if os.statvfs(folder).f_bavail * os.statvfs(folder).f_frsize < reserve + estimated:
        raise OSError('NATIVE_CHECKPOINT_STORAGE_ADMISSION')
    try:
        payload = dict(schema='native-fe-sequential-diagnostic-v1', completed_batches=completed_batches,
                       completed_edits=completed_batches*cfg.bs,
                       weights={name: p.detach().cpu().clone() for name, p in selected.items()},
                       history=cache_c.detach().cpu().clone(), contexts=contexts, rng=rng,
                       config=OmegaConf.to_container(cfg, resolve=True),
                       seen_case_ids=[r['case_id'] for r in requests[:completed_batches*cfg.bs]],
                       identity_sha256=cfg.diagnostic_identity_sha256,
                       z_policy=('CURRENT_BATCH_ENTRY_NO_FUTURE_TARGETS'
                                 if cfg.get('diagnostic_z_reference', 'batch_entry') == 'batch_entry'
                                 else 'CURRENT_BATCH_ONLY_INITIAL_MODEL'))
        with partial.open('xb') as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        checksum = sha256(partial)
        os.replace(partial, destination)
        receipt = dict(path=str(destination), sha256=checksum, bytes=destination.stat().st_size,
                       completed_batches=completed_batches, completed_edits=completed_batches*cfg.bs,
                       identity_sha256=cfg.diagnostic_identity_sha256)
        with destination.with_suffix('.json').open('x') as handle:
            json.dump(receipt, handle, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        directory = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        print('NATIVE_FE_CHECKPOINT', json.dumps(receipt), flush=True)
    finally:
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'])
        if rng['cuda']:
            torch.cuda.set_rng_state_all(rng['cuda'])
    if versions != {name: p._version for name, p in model.named_parameters()}:
        raise RuntimeError('MODEL_CHANGED_DURING_SNAPSHOT')
