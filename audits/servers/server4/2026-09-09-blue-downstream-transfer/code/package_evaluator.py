"""BLUE evaluator 소스만 원본 bytes로 봉인. import/평가/데이터 재생성 없음."""
import ast,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BLUE=Path('/data/janghj/BLUE')
def sha(b):return hashlib.sha256(b).hexdigest()
def save(p,x):
    with p.open('x') as f:json.dump(x,f,sort_keys=True,indent=2)
def main():
    head=subprocess.check_output(['git','-C',str(BLUE),'rev-parse','HEAD'],text=True).strip()
    assert head=='311b076a92e4ed0f14f5c8b4909732da781bc5f7'
    target=ROOT/'evaluator-source-package';target.mkdir(exist_ok=False)
    rels=[str(p.relative_to(BLUE)) for p in sorted((BLUE/'glue_eval').glob('*.py'))]+['util/perplexity.py','README.md']
    if (BLUE/'util/__init__.py').is_file():rels.append('util/__init__.py')
    members=[]
    for rel in rels:
        b=(BLUE/rel).read_bytes();assert b==subprocess.check_output(['git','-C',str(BLUE),'show',head+':'+rel])
        imports=[]
        if rel.endswith('.py'):
            tree=ast.parse(b)
            for n in ast.walk(tree):
                if isinstance(n,ast.Import):imports.extend(a.name for a in n.names)
                elif isinstance(n,ast.ImportFrom):imports.append(n.module)
        p=target/rel;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(b)
        members.append(dict(relative=rel,bytes=len(b),sha256=sha(b),imports=imports))
    auditroot=Path('/data/janghj/ODE-edit/local/state/downstream-dataset-20260909-v1')
    data=json.loads(subprocess.check_output(['python3',str(auditroot/'verify.py'),'/data/janghj/EasyEdit/glue_eval/dataset'],text=True))
    assert data['member_root']=='e9328a5d351816cb9ba89454d228f7ade841c526313dae9c0d1a0e72a8ab00fc'
    save(ROOT/'dataset-independent-verification.json',data)
    save(target/'source-manifest.json',dict(source_head=head,source_tree=subprocess.check_output(['git','-C',str(BLUE),'rev-parse','HEAD^{tree}'],text=True).strip(),members=members,dataset_member_root=data['member_root'],dataset_root_on_SH2='/mnt/raid5/janghj/EasyEdit/glue_eval/dataset',dataset_copy=0,source_changes=0,source_kind='BLUE glue_eval reference; not EasyEdit upstream',source_protocol=dict(number_of_tests=100,fewshot=0,gen_len=5,split='load_data_split reserved[:10],eval[10:110]'),metric_semantic_audit='SH2/GH owner; source byte identity is NOT metric correctness PASS',extra_imports_note='dialogue/sentiment/perplexity copied as wrapper import closure, not extra execution authorization',module_imports_run=0,model_GPU_forward=0))
    manifest=target/'source-manifest.json';print(json.dumps(dict(files=len(members),manifest_sha256=sha(manifest.read_bytes()),bytes=sum(m['bytes'] for m in members))))
if __name__=='__main__':main()
