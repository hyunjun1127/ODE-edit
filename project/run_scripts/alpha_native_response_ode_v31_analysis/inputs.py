import csv,hashlib,io,json,subprocess

COMMIT='0d0a0131e4a6a2a645dfa6530377d420a084d136'
PREFIX='experiment-reports/servers/server2/alpha-native-response-v31-sequential-routing-2026-09-06-v1/main-four-terminal-v1'
EXECUTION='77358b1546d1baf83b3e251afcce663b08d7bfd7'

def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(obj):return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

class Publication:
    def __init__(self,repo):self.repo=repo;self.inventory={}
    def blob(self,name):
        if '/' in name or name.startswith('.'):raise ValueError('MEMBER_PATH')
        data=subprocess.check_output(['git','-C',str(self.repo),'show',f'{COMMIT}:{PREFIX}/{name}'])
        self.inventory[name]=dict(path=f'{PREFIX}/{name}',commit=COMMIT,bytes=len(data),sha256=sha(data),
            original_server2_filesystem_mode='NOT_REHASHED_GIT_BLOB_ONLY')
        return data
    def json(self,name):return json.loads(self.blob(name))
    def rows(self,name):return list(csv.DictReader(io.StringIO(self.blob(name+'.csv').decode())))
    def verify(self):
        raw=self.blob('package-manifest.json');manifest=json.loads(raw)
        receipt=self.json('rooted-package-receipt.json')
        assert sha(raw)==receipt['manifest_sha256']
        assert sha(canonical(manifest))==receipt['root_sha256']
        assert sha(self.blob('factual-report-ko.md'))==receipt['report_sha256']
        for m in manifest['members']:
            b=self.blob(m['path']);assert len(b)==m['bytes'] and sha(b)==m['sha256'],m['path']
        assert len(manifest['members'])==48
        return dict(status='PASS_GIT_PUBLICATION_REHASH',members=48,receipt=receipt,
            external_raw_checkpoint_rehash='NOT_PERFORMED_SERVER2_UNAVAILABLE',
            no_remote_live_result_access=True)
