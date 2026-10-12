"""Conditional author-L2=0 BASIC-EOT final CF generation, source-only adapter.

prepare NEW_ROOT BINDINGS | run CONFIG LOCK | collect CONFIG LOCK --output FILE
No editing, selection, submission, downloads, or editing-runtime mutation.
"""
import argparse
from contextlib import contextmanager,ExitStack
from pathlib import Path
import tempfile
from unittest.mock import patch
import torch
from official.diagnostics.fe_qwen_current import author_basic_eot_benchmark as author
from official.diagnostics.fe_qwen_current import flucon as base,flucon_prepare
from official.diagnostics.fe_qwen_current import failure_eot_probe as eot
from official.experiments.prepare import digest
from official.runners.server1.common import read,verify,member,require

TASK='fe-qwen-author-basic-eot-final-cf-flucon-20261012'
MANIFEST_SHA='6d9a713ab7eaa10871f277e10a3974e0bd5b140c265be80052c279f258876ca8'
_load_reference=base.load_reference
_tracking=base.tracking_values


def audit_final(final_member):
    """Run the actual author paired collector again, not stored summary trust.

    Its fullSHA/CP integrity audit is read-only mmap/chunked, including H hashes.
    Only a temporary SMALL report is written. Generation restore never reads H.
    """
    final=read(verify(final_member))
    require(final.get('status')=='BOTH_AUTHOR_BASIC_EOT_W20_CPU_AUDITED' and final.get('both_W20_complete') is True,
        'BOTH_AUTHOR_FINAL_W20_REQUIRED')
    rows=final['results'];require(len(rows)==2 and sorted(r['dataset'] for r in rows)==['cf','zsre'],'AUTHOR_FINAL_PAIR')
    by={r['dataset']:r for r in rows}
    paths={d:verify(row['result']) for d,row in by.items()}
    with tempfile.TemporaryDirectory(prefix='fe-author-flucon-audit-') as folder:
        actual=author.collect(paths['cf'],paths['zsre'],Path(folder)/'reaudit.json')
        require(read(verify(actual))==final,'AUTHOR_PAIRED_INDEPENDENT_REAUDIT')
    cf=read(paths['cf']);b=read(verify(cf['bindings']));c=read(verify(b['config']))
    require(cf['identity']['dataset']=='cf' and b['main_stream']['sha256']==base.STREAM,'ORIGINAL_CF_STREAM')
    author.settings(cf['identity'])
    for path in paths.values():
        require(not (path.parent/'QUALITY_EARLY_STOP.json').exists() and not (path.parent/'FAILURE.json').exists(),'FAILED_FINAL_FORBIDDEN')
    # Compatibility view for the unchanged shared generator/preparer. Config
    # points to the actual NEW bindings, never a relabeled old L2-10 config.
    return dict(row=dict(checkpoint=cf['checkpoint'],config=cf['bindings']),
        config=dict(stream=b['main_stream'],model_assets=c['model_assets']),identity=cf['identity'],
        terminal=by['cf']['result'],pointer=cf['pointer'],commit=cf['commits'][-1])


def validate_payload(p,original):
    require(set(p)=={'weights','cache_c','contexts','z_member','batch','cursor','identity','rng','integrity_sha256'},'CURRENT_PAYLOAD_SCHEMA')
    require(p['batch']==20 and p['identity']==original['identity'],'AUTHOR_CF_W20_IDENTITY')
    author.settings(p['identity']);require(p['identity']['dataset']=='cf','NO_ZSRE_OR_HELDOUT_GENERATION')
    b=read(verify(original['config']));historical=read(verify(b['runtime']))['resolved']
    _,audit=author.execution_config(historical,b)
    cur=p['cursor'];native=cur['native_config']
    require(native==audit['resolved'] and p['identity']['author_settings_sha256']==audit['resolved_sha256'],'AUTHOR_CP_NATIVE_CONFIG')
    require(p['identity']['author_commit']==author.AUTHOR and p['identity']['config_sha256']==digest(b)
        and p['identity']['stream']==b['main_stream'] and b['main_stream']['sha256']==base.STREAM,'AUTHOR_CP_INPUT_BINDING')
    require(cur['edits']==2000 and digest(cur['case_ids'])==base.ORDER and cur['raw']==read(verify(original['commit']))['raw'],'AUTHOR_W20_CURSOR')
    require(cur['input_adapter']['sha256']==member(eot.__file__)['sha256'] and p['identity']['adapter']['sha256']==member(eot.__file__)['sha256'],'FROZEN_C36_ADAPTER')
    require(p['z_member']==dict(policy='BASIC_EOT_CURRENT_ENTRY',durable_target_cache=False),'CURRENT_Z_POLICY')
    integrity=p['integrity_sha256'];require(type(integrity) is str and len(integrity)==64 and all(x in '0123456789abcdef' for x in integrity),'PAYLOAD_INTEGRITY_FIELD')
    if 'payload_integrity_sha256' in original:require(integrity==original['payload_integrity_sha256'],'PAYLOAD_INTEGRITY_BINDING')
    names={base.HP['rewrite_module_tmp'].format(l)+'.weight' for l in base.HP['layers']}
    require(set(p['weights'])==names,'SELECTED_NAMES')
    for w in p['weights'].values():
        require(w.dtype==torch.bfloat16 and w.device.type=='cpu' and w.is_contiguous() and tuple(w.shape)==base.SELECTED_SHAPE and base.finite_tensor(w),'SELECTED_BF16')
    # No H/context/RNG traversal, cloning, restoration or consumption here.
    # Full checkpoint SHA and paired-author collector authenticate those bytes.
    return p['weights'],integrity


def load_reference(path):
    value=read(path)
    if 'manifest' in value:
        manifest=value['manifest'];verify(manifest)
    else:manifest=member(path)
    require(manifest['sha256']==MANIFEST_SHA,'EXACT_ORIGINAL_REFERENCE_MANIFEST')
    return _load_reference(path) # identity/files/scorer versions/NLTK hashes


def tracking_values(c,source):
    values=_tracking(c,source)
    values.update(task_id=TASK,arm='AUTHOR-L2-0-clamp1-basicEOT-finalCF-generation',
        baseline='FE-author-hparams-current-z-basicEOT',writer='fe_native_batch_entry_z')
    return base.validate_tracking(values)


@contextmanager
def bindings():
    # Fresh evaluation process ONLY. Scoped adapter substitutions restore on
    # exit, never edit old files or any running editing process/global state.
    with ExitStack() as stack:
        for name,value in dict(TASK=TASK,audit_final=audit_final,validate_payload=validate_payload,
            load_reference=load_reference,tracking_values=tracking_values).items():
            stack.enter_context(patch.object(base,name,value))
        yield


def prepare(output,inputs):
    with bindings():return flucon_prepare.prepare(output,inputs)


def run(config,lock):
    with bindings():return base.run(config,lock)


def collect(config,lock,output):
    with bindings():return base.collect(config,lock,output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','run','collect'])
    p.add_argument('first');p.add_argument('second');p.add_argument('--output');a=p.parse_args()
    if a.mode=='prepare':print(prepare(a.first,a.second))
    elif a.mode=='run':run(a.first,a.second)
    else:collect(a.first,a.second,a.output)
