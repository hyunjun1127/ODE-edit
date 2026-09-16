"""Reorganize sealed aggregate evidence only; no experiment/evaluator calls."""
import argparse
import json
import re
from .common import *

def prepare():
    AUDIT.mkdir(parents=True,exist_ok=False)
    docs=[]
    for rel in ['messages/head/2026-09-16-sh4-cake-baseline-cap-report-separation.md',
                'plans/global/2026-09-16-sh4-cake-baseline-cap-report-separation.json']:
        p=WT/rel;docs.append(dict(ref(p),reading='FULL_READ'))
        target=LOCAL/'authoritative'/p.name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write(p.read_bytes())
    trees=protected_trees();assert all(r['git_tree']==r['evidence_tree'] for r in trees)
    data=dict(instruction_id=TASK,host='server4',session='01a04939-b5c7-7a03-ba2d-ef3343d62cfd',
       base_main=git('rev-parse','HEAD'),base_tree=git('rev-parse','HEAD^{tree}'),documents=docs,
       protected_packages=trees,scheduler_queries=0,model_forwards=0,raw_payload_reads=0,
       new_server4_README=not (BASE/'README.md').exists(),SH2_publication_preserved=git('merge-base','--is-ancestor','8aabac92fea441e4499d1426db2611cc3d3e7daa','HEAD')=='')
    save(AUDIT/'full-read.json',data)
    write(WT/'messages/acks/server4/2026-09-16-cake-baseline-cap-report-separation.md',
       '# FULL_READ / report-separation M0\n\n'
       f'Instruction `{TASK}`. main4d9bd910/tree9f8e4151와 지시문·dispatch를 전체 읽었다. Envelope SHA{docs[0]["sha256"]}; dispatch SHA{docs[1]["sha256"]}.\n\n'
       '새 clean worktree에서 CAKE+baseline 전용 v2와 alpha-cap 독립 v2, 링크 전용 목록을 실제 생성한다. 기존 v1은 Git tree identity로 결속하며 원raw 재해시/모델 재평가 없이 compact CSV/PNG만 재사용한다. SH2 publication과 공유 dirty는 보존한다. Scheduler/새GPU/forward/Slurm/rsync/delete0. 별도 승인 대기 없이 보고·main 통합까지 진행한다.')

class Publisher:
    def __init__(self):self.mapping=[]
    def copy(self,src,dst):
        dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.open('xb') as f:f.write(src.read_bytes())
        self.record(src,dst,'EXACT_BYTES_REUSED')
    def record(self,src,dst,mode):
        self.mapping.append(dict(source=ref(src),destination=ref(dst),mode=mode))
    def csv_view(self,src,dst,transform):
        before=readcsv(src);after=transform(before);table(dst,after)
        self.record(src,dst,'ROW_VIEW_LABEL_ONLY_NUMERIC_VALUES_UNCHANGED')
    def cake(self):
        CAKE.mkdir(parents=True,exist_ok=False)
        for p in sorted(CAKE_OLD.glob('*.csv')):
            if p.name=='family-final-comparison.csv':
                def family(rr):
                    return [dict(r,source_arm=r['arm'],arm=label(r['arm']),arm_display_label=label(r['arm'])) for r in rr]
                self.csv_view(p,CAKE/p.name,family)
            elif p.name=='baseline-paired.csv':
                self.csv_view(p,CAKE/p.name,lambda rr:[dict(r,source_comparison=r['comparison'],comparison=label_text(r['comparison'])) for r in rr])
            else:self.copy(p,CAKE/p.name)
        self.copy(CAKE_OLD/'compatibility.json',CAKE/'compatibility.json')
        for name in ('cake-fullseen.png','cake-cohort-retention.png'):
            self.copy(CAKE_OLD/'figures'/name,CAKE/'figures'/name)
        configs=readcsv(BASELINE/'source-config-compatibility.csv')
        mappings=[]
        for r in configs:
            h=json.loads(r['hparams']);name=label(r['arm'])
            if '_BLUE(' in name:assert h['blue'] is True
            mappings.append(dict(raw_arm_id=r['arm'],display_label=name,physical_layers=json.dumps(h['layers']),
                blue_style=h['blue'],job=r['job'],source_head=r['source_head'],model_revision=r['model_revision'],
                sample_root=r['sample_root'],seed=r['seed'],hparam_sha256=r['config_sha256']))
        mappings += [dict(raw_arm_id='CAKE_NATIVE',display_label='CAKE_NATIVE',physical_layers='[4, 5, 6, 7, 8]',blue_style='NATIVE_CAKE',job='48101'),
                     dict(raw_arm_id='PRE_EDIT_W0',display_label='W0',physical_layers='[]',blue_style='NOT_EDITED',job='42673')]
        table(CAKE/'arm-label-provenance.csv',mappings)
        self.record(BASELINE/'source-config-compatibility.csv',CAKE/'arm-label-provenance.csv','LABEL_MAPPING_WITH_RAW_IDS_NO_METRIC_CHANGE')
        allowed={r['raw_arm_id'] for r in mappings}
        def baseline_view(rr):
            return [dict(r,source_arm=r['arm'],arm=label(r['arm']),arm_display_label=label(r['arm'])) for r in rr if r['arm'] in allowed]
        for old,new in [('cumulative-metrics.csv','baseline-cumulative.csv'),('final-distributions.csv','baseline-final-distributions.csv'),
                        ('source-config-compatibility.csv','baseline-compatibility.csv')]:
            self.csv_view(BASELINE/old,CAKE/new,baseline_view)
        self.csv_view(BASELINE/'w0-distributions.csv',CAKE/'w0-reference.csv',
            lambda rr:[r for r in rr if r['prefix_requests']=='10000'])
        self.copy(BASELINE/'preedit-compute.csv',CAKE/'w0-prior-cost.csv')
        cumulative=readcsv(CAKE/'baseline-cumulative.csv')
        b10=[r for r in cumulative if r['batch']=='10']
        assert all(int(r['denominator'])=={'RS':1000,'PS':2000,'NS':10000}[r['metric']] for r in b10)
        table(CAKE/'baseline-B10-first1000.csv',b10)
        self.record(BASELINE/'cumulative-metrics.csv',CAKE/'baseline-B10-first1000.csv','EXACT_BATCH10_POPULATION_VIEW')
        text=(CAKE_OLD/'diagnostic-report-ko.md').read_text()
        text=text.replace('# CAKE_NATIVE — fixed10k B100×100 완료 사실 리뷰','# CAKE + AlphaEdit/MEMIT baseline — fixed10k 독립 정본 v2')
        text=label_text(text).replace('CAKE에는 EP accepted-only ledger가 없다.','CAKE에는 accepted-only ledger가 없다.')
        text=text.replace('Legacy ORIGINAL은 BLUE를 뜻하므로 그 단독표기를 쓰지 않는다.','BLUE-style singleton은 physical L4/L5/L6/L7/L8를 표시한다. raw arm ID는 arm-label-provenance.csv로 분리했다.')
        preface=(f'보고 재구성 instruction `{TASK}`. 이 문서는 CAKE와 동일 fixed10k/order의 W0·AlphaEdit/MEMIT native·BLUE pair·BLUE-style singleton만 다룬다. '
            '새 수치 산출·재평가·scheduler 조회는 없으며 아래 검산과 실행 사실은 완료리뷰 v1에서 확인한 근거를 재사용한다. 원 수치·분모·metric·검증 수준은 바꾸지 않았다.\n\n')
        text=text.replace('\n\nInstruction ', '\n\n'+preface+'기존 완료리뷰 Instruction ',1)
        w0=readcsv(CAKE/'w0-reference.csv')
        w0_table='### 편집 전 W0 — 같은 full10000 입력\n\n'+mdtable(['metric','n/d','검증 출처'],[[r['metric'],r['numerator']+'/'+r['denominator'],'job42673 sealed aggregate reuse'] for r in w0])
        w0_table+='\n\nW0는 pre-edit 상태이고 편집된 W100과 구별한다. W0 사전 평가비용4440GPU-sec는 한 번 수행된 과거 비용이며 각 family에서 중복 청구하지 않는다.\n\n'
        text=text.replace('### AlphaEdit family + CAKE reference',w0_table+'### AlphaEdit family + CAKE reference',1)
        text=text.replace('## 2. 동일10k family별 baseline 비교','## 2. 동일10k family별 baseline 비교\n\nCAKE를 두 family에 반복 표시하는 것은 같은 job48101의 참고 표시다. CAKE 실행·분모·32194GPU-sec를 두 번 합산하지 않는다.')
        text=text.replace('## 8. 저장 CSV로 생성한 그림','## 8. CAKE 및 baseline 전용 그림과 중간 상태')
        text=text.replace('그림의 AE/ME는 AlphaEdit/MEMIT 약칭이다.','모든 legend는 native/BLUE-style 및 실제 physical layer를 구분한다.')
        text+='\n\n![Family별 actual full-seen curves](figures/family-cumulative.png)\n\n'
        text+='baseline-cumulative.csv와 baseline-B10-first1000.csv는 기존 actual checkpoint의 전체 seen-prefix만 선택한 view다. B10 first1000는 W10이며 W100 first1000과 다른 state다. CAKE B10/1k는 CAKE-B10-first1000.csv에 별도 보존했고 final10k 표에 합산하지 않는다.\n\n'
        text+='## 9. v1 → v2 보존·재현\n\nsource-to-v2.json은 각 표·그림의 원경로/SHA와 복사 또는 label-only view를 결속한다. v1 패키지는 변경하지 않았다. arm-label-provenance.csv에서 raw ID와 표시 이름을 대응시킨다. 새 두 family 그림만 코드로 재생성했고 CAKE 자체 추이2개는 동일 bytes를 재사용했다. 재현 명령과 판독 한계는 README.md 및 analysis-manifest.json을 참조한다.\n'
        write(CAKE/'diagnostic-report-ko.md',with_toc(text))
        self.record(CAKE_OLD/'diagnostic-report-ko.md',CAKE/'diagnostic-report-ko.md','RESTRUCTURED_CAKE_BASELINE_ONLY_NUMBERS_UNCHANGED')
    def cap(self):
        CAP.mkdir(parents=True,exist_ok=False)
        for p in sorted(CAP_OLD.glob('*.csv')):
            if p.name=='CAKE-B10-first1000.csv':continue
            self.copy(p,CAP/p.name)
        for p in sorted((CAP_OLD/'figures').glob('*.png')):self.copy(p,CAP/'figures'/p.name)
        text=(CAP_OLD/'diagnostic-report-ko.md').read_text()
        text=text.replace('# EP-TW-1 alpha cap sweep — SH4 완료 사실 보고','# EP-TW-1 alpha-cap sweep — W0→first1000 독립 정본 v2')
        text=text.replace('\n\nInstruction ','\n\n보고 재구성 지시와 source는 analysis-manifest.json에 결속했다. 본 정본은 네 alpha-cap 정책의 완료 결과만 재구성한다. 현재 scheduler·raw tensor·모델 재평가 없이 완료리뷰 v1의 표와 검증 근거를 재사용한다. 수치·분모·선택 arm·검증 수준은 불변이다.\n\n기존 과학 실행 Instruction ',1)
        text=text.replace('CAKE의 no-weight-storage는 sweep에 적용하지 않았다. Sweep monitoring은 이후 사용자 지시로 중지했고, 이번 completed-review recall에서만 지정 네 job의 terminal 확인과 CPU 분석을 재개했다. CAKE는 별도 10k 보고서로 리뷰하며 실행 변경0이다.',
            '신규3arm의 당시10CP/arm 및 W10 보존 계약을 유지한다. 기존 monitoring 중지 이력과 완료리뷰 때의 한정 terminal 확인을 재사용하며, 이번 보고 재구성에서는 scheduler를 조회하지 않았다.')
        text=text.replace('이번 review-only recall에서 원문·pause를 다시 결속했다. 2026-09-15 사용자 pause를 보존했고 이번에는 정확 네 job의 scheduler만 한 번 확인했다. CAKE도 별도 완료 리뷰했지만 10k 최종값과 아래1k를 직접 합치지 않는다.',
            '아래는 기존 완료리뷰의 source-backed 동작 감사다. 당시 pause와 실행 lock을 결속한 근거를 그대로 재사용하며, 이번 v2에서는 신규 source/상태/미분 검사를 수행하지 않았다.')
        text='\n'.join(line for line in text.splitlines() if not line.startswith('CAKE의 actual B10 first1000 참고값'))
        text=text.replace('## 9. 이번 recall의 설계-실행 상세 보충','## 9. 저장 근거로 확인한 설계-실행 상세')
        text=text.replace('`sweep-scheduler-adapted.json`','`scheduler-summary.csv` (기존 terminal 기록의 표)')
        # Existing two image references are replaced by an explicit complete six-figure section.
        text='\n'.join(line for line in text.splitlines() if not line.startswith('!['))
        text+='\n\n## 10. 독립 sweep 그림과 재현\n\n'
        for name,caption in [('final-first1000.png','Actual W10 원분모'),('candidate-versus-selected-action.png','최대 후보와 실제 선택 action'),
                             ('current-curves.png','매 batch Current100'),('candidate-finite-screen.png','모든 후보 finite/quality screen'),
                             ('atwrite-terminal-transitions.png','At-write→W10 문항 전이'),('generic-controller-observer.png','S64 controller와 Dev128 observer')]:
            text+=f'![{caption}](figures/{name})\n\n'
        text+='그림6개와 sweep 수치 CSV는 v1 bytes를 그대로 재사용했다. 필요한 동일 first1000 N4/native/BLUE/W0 비교는 baseline-first1000.csv 및 별도 비교절에만 유지한다. source-to-v2.json과 analysis-manifest.json은 원출처·검증 재사용·새 보고 source를 구분한다.\n'
        assert 'CAKE' not in text
        write(CAP/'diagnostic-report-ko.md',with_toc(text))
        self.record(CAP_OLD/'diagnostic-report-ko.md',CAP/'diagnostic-report-ko.md','RESTRUCTURED_ALPHA_ONLY_NO_CAKE_CONTENT_NUMBERS_UNCHANGED')
    def indexes(self):
        INDEX.mkdir(parents=True,exist_ok=False)
        ca='../'+str(CAKE.relative_to(BASE))+'/diagnostic-report-ko.md'
        cp='../'+str(CAP.relative_to(BASE))+'/diagnostic-report-ko.md'
        write(INDEX/'README.md','# Server4 독립 완료 보고 목록\n\n'
             f'- [CAKE + AlphaEdit/MEMIT baseline — fixed10k]({ca}): 동일 order의 native/BLUE-style pair·singleton baseline과 CAKE 전용 정본.\n'
             f'- [EP-TW-1 alpha-cap sweep — first1000]({cp}): 네 cap 정책과 해당 범위 참고 baseline의 독립 정본.\n\n'
             '현재 기본 정본은 위 두 v2다. 이 목록은 결과를 합쳐 비교하거나 비용을 합산하지 않는다.\n\n'
             '[기존 종합 v1](../completed-experiments-review-2026-09-16-v1/diagnostic-report-ko.md)은 역사적 합본으로 bytes·manifest·receipt를 보존한다.')
        readme=BASE/'README.md'
        default='# Server4 보고 목록\n\n## 현재 독립 정본\n\n'
        default+=f'- [CAKE + AlphaEdit/MEMIT baseline — fixed10k]({CAKE.relative_to(BASE)}/diagnostic-report-ko.md)\n'
        default+=f'- [EP-TW-1 alpha-cap sweep — first1000]({CAP.relative_to(BASE)}/diagnostic-report-ko.md)\n'
        default+='- [범위별 보고 목록](completed-experiments-review-2026-09-16-v2/README.md)\n\n'
        if readme.exists():
            raise RuntimeError('New shared README appeared; preserve it and integrate its existing entries explicitly')
        others=[]
        excluded={CAKE_PARENT.name,CAP_PARENT.name,INDEX.name}
        for d in sorted(BASE.iterdir()):
            if not d.is_dir() or d.name in excluded:continue
            tag=' — 역사적 합본, 현재 기본 정본 아님' if d.name=='completed-experiments-review-2026-09-16-v1' else ''
            others.append(f'- [{d.name}]({d.name}/){tag}')
        write(readme,default+'## 기존 보고 및 보존 자료\n\n'+'\n'.join(others))
    def finish(self):
        for root in (CAKE,CAP):
            mapping=[x for x in self.mapping if str(root.relative_to(WT))+'/' in x['destination']['path']]
            save(root/'source-to-v2.json',dict(instruction_id=TASK,source_commit=EVIDENCE_COMMIT,entries=mapping))
            write(root/'README.md',f'# {"CAKE + baseline" if root==CAKE else "EP alpha-cap"} 독립 정본 v2\n\n'
              '[상세 보고](diagnostic-report-ko.md) · [구판→신판 mapping](source-to-v2.json)\n\n'
              '이 패키지는 완료된 aggregate publication만 재구성한다. 원raw/model/evaluator를 읽거나 실행하지 않는다. 기존 v1 검증 수준을 유지한다.\n\n'
              '```bash\npython -B -m project.run_scripts.server4_completed_review.report_separation.build build\n'
              'python -B -m project.run_scripts.server4_completed_review.report_separation.plots --output NEW_FIGURES\n'
              '/usr/bin/python3 -B -m project.run_scripts.server4_completed_review.report_separation.checks --output NEW_CHECKS\n```\n\n'
              'build는 v2가 없는 clean worktree에서 create-once 실행한다. plots는 새 출력에 쓰고 checks는 기존 CSV·GFM HTML·hash만 확인한다. 실행/분석/현재 publication commit은 manifest에서 구분한다. 새 numerical PASS를 부여하지 않는다.')
        save(AUDIT/'publication-mapping.json',dict(instruction_id=TASK,entries=self.mapping))

def build():
    p=Publisher();p.cake();p.cap();p.indexes();p.finish()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','build']);args=p.parse_args()
    prepare() if args.action=='prepare' else build()
