"""SH4 source deletion receipt를 이전 전체 해시 검증 catalog와 대조한다."""
import json
from pathlib import Path
from verify_initial import CONTROL, sha, save


def main():
    p = CONTROL / 'incoming/jvp1k-deletion-receipt.json'
    expected = '73b502dedfa2a3d7631cc90053126a55414ed9a89a77e3a2f8742edebd77fa5d'
    assert sha(p) == expected
    deletion = json.loads(p.read_text())
    receipt_path = CONTROL / 'jvp1k-v1-VERIFIED_DESTINATION.json'
    receipt = json.loads(receipt_path.read_text())
    assert deletion['receiver_receipt_sha256'] == sha(receipt_path)
    catalog_path = Path(receipt['destination_manifest'])
    assert sha(catalog_path) == receipt['destination_manifest_sha256']
    catalog = json.loads(catalog_path.read_text())
    mapped = {m['source_path']: m['destination'] for m in catalog['members']}
    assert deletion['status'] == 'SOURCE_EXACT_FILES_REMOVED_DESTINATION_PRESERVED'
    assert len(mapped) == len(deletion['removed']) == deletion['count'] == 6
    assert {m['source'] for m in deletion['removed']} == set(mapped)
    for row in deletion['removed']:
        target = mapped[row['source']]
        assert row['removed'] is True
        assert (row['destination'], row['bytes'], row['sha256']) == (target['path'], target['bytes'], target['sha256'])
    for m in catalog['checkpoint_members']:
        prior = m['staging_verification']
        retained = Path(m['destination_path'])
        st = retained.stat()
        assert not retained.is_symlink()
        assert (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns) == (prior['dev'], prior['inode'], prior['bytes'], prior['mtime_ns'])
    assert sum(m['bytes'] for m in deletion['removed']) == deletion['logical_bytes'] == 41460748986
    result = dict(status='SOURCE_DELETION_RECEIPT_RECONCILED_DESTINATION_RETAINED',
        deletion_receipt_sha256=expected, destination_receipt_sha256=sha(receipt_path),
        catalog_sha256=sha(catalog_path), count=6, bytes=41460748986,
        verification='Prior full destination SHA plus current stable inode/size/mtime; exact source/destination/digest mapping.',
        source_unlink_evidence='SH4 signed-by-role receipt; no new source-host observation by SH2.',
        source_observed_available_delta=deletion['observed_available_delta'],
        filesystem_delta_attribution='Shared filesystem; not exclusively attributable.',
        sh2_delete_rsync_gpu_model_slurm_actions=0)
    save(CONTROL / 'jvp1k-deletion-ack.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
