"""Synthetic CPU-only identity guards; no real source checkpoint touched."""
import unittest
from delete_verified import identity
from inventory import C,os,Path
class Guards(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.root=C/'synthetic-guard-fixture-v1';cls.root.mkdir(exist_ok=False)
 def specimen(self,name):
  root=self.root/name;root.mkdir();p=root/'W-M.pt'
  with p.open('xb') as f:f.write(b'not a tensor: synthetic identity fixture')
  s=p.stat();return p,dict(path=str(p),realpath=str(p),uid=s.st_uid,nlink=1,dev=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns)
 def test_exact(self):
  p,e=self.specimen('exact');self.assertEqual(identity(e),p)
 def test_symlink_rejected(self):
  p,e=self.specimen('symlink');q=p.parent/'other';q.symlink_to(p);e['path']=str(q)
  with self.assertRaises(AssertionError):identity(e)
 def test_hardlink_rejected(self):
  p,e=self.specimen('hardlink');os.link(p,p.parent/'other')
  with self.assertRaises(AssertionError):identity(e)
 def test_stat_change_rejected(self):
  p,e=self.specimen('stat');os.utime(p,ns=(p.stat().st_atime_ns,e['mtime_ns']+10))
  with self.assertRaises(AssertionError):identity(e)
 def test_unapproved_filename_rejected(self):
  p,e=self.specimen('name');e['path']=str(p.parent/'report.md')
  with (p.parent/'report.md').open('x') as f:f.write('synthetic')
  e['realpath']=e['path']
  with self.assertRaises(AssertionError):identity(e)
if __name__=='__main__':unittest.main()
