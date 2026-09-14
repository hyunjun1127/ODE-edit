import unittest
from .refresh_allocation import allocation_rows,concurrency

class AllocationTests(unittest.TestCase):
    def test_overlapping_allocations_not_compute_utilization(self):
        header='JobID|JobIDRaw|State|ExitCode|ElapsedRaw|Submit|Start|End|AllocTRES|ReqMem|NodeList|MaxRSS|\n'
        rows='1_0|2|COMPLETED|0:0|10|2026-09-14T00:00:00|2026-09-14T00:00:00|2026-09-14T00:00:10|cpu=8,gres/gpu=1|60416Mn|server4||\n'
        rows+='1_1|1|COMPLETED|0:0|10|2026-09-14T00:00:00|2026-09-14T00:00:05|2026-09-14T00:00:15|cpu=8,gres/gpu=1|60416Mn|server4||\n'
        values=allocation_rows(header+rows)
        self.assertEqual(sum(r['allocated_gpu_seconds'] for r in values),20)
        c=concurrency(values)
        self.assertEqual((c['max_concurrent_GPUs'],c['at_least_two_GPU_seconds'],c['interval_gpu_seconds']),(2,5,20))
        self.assertEqual(c['GPU_compute_occupancy'],'NOT_MEASURED_ALLOCATION_IS_NOT_UTILIZATION')

    def test_active_interval_not_imputed(self):
        c=concurrency([dict(start='2026-09-14T00:00:00',end='Unknown',gpus=1)])
        self.assertEqual(c['status'],'NOT_TERMINAL')

if __name__=='__main__':unittest.main()
