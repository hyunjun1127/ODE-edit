"""Bounded CPU final reduction or exact archive; failures keep all payloads."""
import argparse
import json
from pathlib import Path
from official.experiments.prepare import file_sha, write_new
from official.evaluation import reduce
from official.runners.server2.qwen_plan import rows

def read(p):return json.loads(Path(p).read_text())

def archive_once(root,cell):
    if (root/'kept.json').exists() and cell in read(root/'kept.json'):
        write_new(root/'archive'/cell/'pending-keep.json',dict(status='ARCHIVE_PENDING_KEEP_SOURCE',
            reason='KEPT_ORIGINAL_FT_PROVENANCE_NO_RETROACTIVE_ADOPTION',
            original=read(root/'kept.json')[cell],transfers=0,deletions=0))
        return
    out=root/'runs'/cell
    if not (out/'terminal.json').is_file() or not (root/'receiver.json').is_file():
        write_new(root/'archive'/cell/'pending-keep.json',dict(status='ARCHIVE_PENDING_KEEP_SOURCE',
            reason='FINAL_TERMINAL_OR_RECEIVER_INPUT_UNAVAILABLE',transfers=0,deletions=0))
        return
    from project.run_scripts.server2_qwen_archive import archive
    archive(root,cell)

def collect(root):
    results=[]
    for row in rows():
        cell=row['logical_main_row'];out=root/'runs'/cell
        config=row['config']
        if (root/'mask-profile.json').exists() and (root/'configs'/f'{cell}.json').exists():
            config=read(root/'configs'/f'{cell}.json')
        if (root/'kept.json').exists() and cell in read(root/'kept.json'):
            original=read(root/'kept.json')[cell]
            assert file_sha(original['config_path'])==original['config_file_sha256']
            config=read(original['config_path']);out=Path(original['root'])/'runs'/cell
        if not (out/'terminal.json').exists():
            results.append(dict(cell=cell,status='NOT_COMPLETE_NO_TERMINAL'));continue
        terminal=read(out/'terminal.json')
        try:
            assert terminal['status']=='W20_COMPLETE' and terminal['completed_edits']==2000
            assert terminal['config_sha256']==config['config_sha256']
            assert len(terminal['commits'])==20
            for i,m in enumerate(terminal['commits'],1):
                assert file_sha(m['path'])==m['sha256'] and read(m['path'])['completed_batch']==i
            factual=terminal['calculation_evidence']['factual']
            assert file_sha(factual['path'])==factual['sha256']
            cases=read(factual['path']);assert len(cases)==2000
            summary=getattr(reduce,'counterfact' if row['config']['dataset']=='cf' else 'zsre')(cases)
            for name,m in terminal['calculation_evidence'].items():assert file_sha(m['path'])==m['sha256']
            results.append(dict(cell=cell,status='W20_RAW_CPU_VERIFIED',job_id=terminal['actual_job_id'],
                dataset=row['config']['dataset'],method=row['config']['method'],summary=summary,
                checkpoint_identity=terminal['checkpoint_identity'],raw=factual,
                source_root=str(out.parent.parent),original_config_sha256=config['config_sha256'],
                archive='VERIFIED_SOURCE_REMOVED' if (root/'archive'/cell/'source-removed.json').exists() else 'KEEP_OR_ARCHIVE_PENDING'))
        except Exception as exc:
            results.append(dict(cell=cell,status='REDUCTION_FAILED_KEEP_SOURCE',error_type=type(exc).__name__))
    write_new(root/'collector/result.json',dict(rows=results,model_loads=0,actual_GPU=0,
        qualification='NOT_RUN_USER_DISABLED',W_B_remote_delivery='NOT_REQUERIED',automatic_retry=False))
    if (root/'mask-profile.json').exists():
        from official.runners.server2.qwen_mask_ft_eval import collect as collect_ft
        collect_ft(root)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--cell');p.add_argument('--collector',action='store_true');a=p.parse_args()
    if a.collector:collect(a.root)
    else:archive_once(a.root,a.cell)
