import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import fixed_counterfact as f


class FixedStreamTests(unittest.TestCase):
    def test_existing_files_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'asset'
            f.create_once(p, b'original')
            f.create_once(p, b'original')
            with self.assertRaises(ValueError):
                f.create_once(p, b'replacement')
            self.assertEqual(p.read_bytes(), b'original')

    def test_prefix_is_ordered_and_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = [{'case_id': i} for i in range(10000)]
            f.create_once(Path(tmp) / 'counterfact.json', f.encoded(records))
            with patch.object(f, 'verify') as verify:
                for n in [1, 1000, 3000, 10000]:
                    self.assertEqual(f.load_prefix(tmp, n), records[:n])
                self.assertEqual(verify.call_count, 4)
                for n in [0, -1, 10001, 1000000, True, 3.5]:
                    with self.assertRaises(ValueError):
                        f.load_prefix(tmp, n)

    def test_record_hash_order_and_inventory(self):
        records = [{'case_id': i, 'paraphrase_prompts': ['a', 'b'],
                    'neighborhood_prompts': ['n'] * 10} for i in range(10000)]
        sample = {'records': [{'case_id': i, 'ordinal': i, 'batch_index': i // 100 + 1,
                              'batch_ordinal': i % 100, 'raw_record_sha256': f.sha(f.encoded(r))}
                             for i, r in enumerate(records)]}
        self.assertEqual(f.validate_records(records, sample)['rewrite'], 10000)
        bad = copy.deepcopy(records)
        bad[0], bad[1] = bad[1], bad[0]
        with self.assertRaises(ValueError):
            f.validate_records(bad, sample)
        bad = copy.deepcopy(records)
        bad[0]['paraphrase_prompts'][0] = 'changed'
        with self.assertRaises(ValueError):
            f.validate_records(bad, sample)


if __name__ == '__main__':
    unittest.main()
