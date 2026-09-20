"""Namespace adapter over the executed SH3 loader; independent fixed10k chain."""
import json
from pathlib import Path
import torch
from scripts.fixed_counterfact import load_prefix
from project.run_scripts.en_adaptive_nullspace.runtime import Runtime as ParentRuntime
from project.run_scripts.en_adaptive_nullspace.native import requests_from_records
from project.run_scripts.en_adaptive_nullspace.metrics import digest
from .io import save

ARMS = ('EN_ADAPT_H_RES', 'EN_ADAPT_H_GSS_REC')


class Runtime(ParentRuntime):
    def __init__(self, output, config):
        if config['arm'] not in ARMS or config['batches'] != 100 or config['batch_size'] != 100:
            raise ValueError('EXACT_TWO_ARM_FIXED10K_SCOPE')
        if config['save_checkpoints'] or config['cross_job_prefix_sharing']:
            raise ValueError('NO_CP_NO_CROSS_JOB_STATE')
        expected=dict(bank_capacity=512,pending_capacity=100,hot_pool_capacity=612,
            seed=20260920,map_dimensions=[32,32],map_replicas=2,map_feature_dimension=2048,
            history_microbatch=1,half_life_batches=5.12,history_coefficient=1.,reference_coefficient=1.,
            primary_epsilon=.05,native_chunk_size=16,current_only_Q=True,
            full_reference_documents=512,history_target_truncation=None,history_panel_size=128,
            full_observer_batches=[2,5,10,20,30,40,50,60,70,80,90,100])
        if any(config.get(k)!=v for k,v in expected.items()) or not config.get('map_seal'):
            raise ValueError('EXACT_GSS_RUNTIME_POLICY_AND_MAP_SEAL')
        super().__init__(output)
        self.records = load_prefix(self.ready_manifest['dataset_root'], 10000)
        self.requests = requests_from_records(self.records)
        # Parent receipt describes its unchanged loader's 300-record prefix;
        # this explicitly sealed adapter extends only the consumed data horizon.
        self.identity.update(records=digest(self.records),
            case_ids=[r['case_id'] for r in self.records],
            batches=[[r['case_id'] for r in self.records[k:k+100]] for k in range(0, 10000, 100)],
            arm=config['arm'], batches_per_arm=100, independent_cold_W0_zero_M4=True,
            parent_loader_record_count=300, effective_record_count=10000,
            cross_job_prefix_sharing=False, precision='NOT_ESTABLISHED')
        save(Path(output) / 'runtime-fixed10k.json', self.identity)
        if len(set(self.identity['case_ids'])) != 10000:
            raise ValueError('FIXED10K_UNIQUE_CASE_IDS')
        if not torch.equal(self.W.detach().cpu(), self.W0):
            raise ValueError('COLD_ENTRY_REQUIRED')
