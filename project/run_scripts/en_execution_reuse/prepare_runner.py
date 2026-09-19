"""Single-job generated256 preparation; never native-fit/optimizer/B2."""
import argparse
import json
from pathlib import Path
import resource
import time
import traceback
import torch
from .model import Runtime, require_lock
from .generated_reference import GeneratedReferenceBuilder, configured_eos
from .generated_teacher import GeneratedTeacherStore, generation_policy
from .preparation import create_json, sha, member
from project.run_scripts.single_layer_edit_preserving_correction.common import digest


def run(lock):
    require_lock(lock)
    if lock['stage'] != 'GENERATED_REFERENCE_PREPARATION':
        raise ValueError('PREPARATION_ONLY')
    out = Path(lock['output'])
    out.mkdir(parents=True, exist_ok=False)
    start, stage = time.monotonic(), 'source'
    rt = None
    try:
        for item in lock['execution']['members'] + lock['external_members']:
            p = Path(item['path'])
            if p.stat().st_size != item['bytes'] or sha(p) != item['sha256']:
                raise ValueError('SOURCE_ASSET_DRIFT:'+str(p))
        create_json(out/'execution-entry.json', dict(lock=lock, model_validation='NOT_RUN_AT_ENTRY'))
        stage = 'W0_load'
        rt = Runtime(lock, out)
        before = rt.byte_hash_nonselected()
        create_json(out/'nonselected-before.json', before)
        binding = dict(source_sha256=lock['execution']['archive']['sha256'],
            model_revision=lock['model_revision'], tokenizer_revision=lock['model_revision'],
            model_config_sha256=lock['model_config_sha256'],
            model_weights_sha256=lock['model_weights_identity_sha256'],
            tokenizer_sha256=lock['tokenizer_identity_sha256'], w0_sha256=rt.identity['W0_header_bytes'],
            bos_token_id=rt.tok.bos_token_id,
            generation=generation_policy(configured_eos(rt.model), kv_cache=False),
            mask_policy='all_ones_int64', position_policy='contiguous_zero_based_int64',
            teacher_policy='canonical_W0_TF_FP32_full_vocab_log_softmax',
            selected_parameter='model.layers.4.mlp.down_proj.weight',
            runtime=dict(torch=lock['torch'], transformers=lock['transformers'], dtype='float32',
                         attention='eager', matmul_tf32=False, cudnn_tf32=False, source=lock['execution']['commit']))
        create_json(out/'teacher-binding.json', binding)
        stage = 'generated256_teacher_and_upstream'
        builder = GeneratedReferenceBuilder(rt.model, binding, out/'generated', lock['reference_inputs']['path'])
        def progress(index, capsule, work):
            print(json.dumps(dict(event='GENERATED_DOCUMENT_COMPLETE', index=index, role=capsule['role'],
                T=capsule['actual_length'], seconds=work['document_wall_seconds'])), flush=True)
        manifest = builder.build(progress=progress)
        stage = 'complete_store_CPU_verification'
        store = GeneratedTeacherStore(out/'generated', manifest['path'], expected_manifest_sha256=manifest['sha256'],
            inputs_path=lock['reference_inputs']['path'], expected_binding=binding, require_upstream_cache=True)
        rt.guard()
        after = rt.byte_hash_nonselected()
        if before != after:
            raise RuntimeError('PREPARATION_NONSELECTED_BYTES_MUTATED')
        create_json(out/'nonselected-after.json', after)
        create_json(out/'READY.json', dict(status='GENERATED_REFERENCE_READY_NOT_CORRECTION_VALIDATION',
            manifest=manifest, store=store.receipt, binding=member(out/'teacher-binding.json'),
            actual_correction_parity='NOT_RUN', native_fit_calls=0, model_forwards='DOCUMENT_WORK_LEDGER',
            generated_documents=640, source=lock['execution'], seconds=time.monotonic()-start,
            peak_gpu_allocated=torch.cuda.max_memory_allocated(), peak_gpu_reserved=torch.cuda.max_memory_reserved(),
            peak_host_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            B2_authorized=False, automatic_continuation=False, runtime=rt.identity))
    except BaseException as exc:
        create_json(out/'failure.json', dict(status='TECHNICAL_FAILURE', stage=stage, error=repr(exc),
            traceback=traceback.format_exc(), seconds=time.monotonic()-start, partial_preserved=True,
            native_fit_calls=0, automatic_restart=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--max-batches', type=int, required=True)
    args = parser.parse_args()
    if args.max_batches != 1:
        raise ValueError('B1_ONLY')
    run(json.loads(args.lock.read_text()))
