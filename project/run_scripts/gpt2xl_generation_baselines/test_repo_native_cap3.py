"""CPU-only allocation guards; never calls the scheduler or changes a job."""
import unittest
from .repo_native_cap3 import require_unallocated_pending


class Cap3Allocation(unittest.TestCase):
    def row(self):
        return dict(JobState='PENDING',RunTime='00:00:00',StartTime='Unknown',
            NodeList='',AllocTRES='(null)')

    def test_present_explicit_empty_fields_are_valid(self):
        for node in ('','(null)'):
            for allocation in ('','(null)'):
                row=self.row();row.update(NodeList=node,AllocTRES=allocation)
                require_unallocated_pending(row)

    def test_missing_allocation_fields_are_not_observed_empty(self):
        for key in ('NodeList','AllocTRES','JobState','RunTime','StartTime'):
            row=self.row();del row[key]
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                require_unallocated_pending(row)

    def test_started_allocated_or_unknown_metadata_fails_closed(self):
        for key,value in (('JobState','RUNNING'),('JobState','CONFIGURING'),
                ('RunTime','00:00:01'),('StartTime','2026-10-08T22:00:00'),
                ('NodeList','devbox'),('AllocTRES','cpu=8,mem=64G,gres/gpu=1'),
                ('AllocTRES','UNKNOWN')):
            row=self.row();row[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(RuntimeError):
                require_unallocated_pending(row)


if __name__=='__main__':unittest.main()
