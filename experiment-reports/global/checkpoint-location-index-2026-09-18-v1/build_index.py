"""Build a metadata-only location report. Never alters remote artifacts."""
from __future__ import annotations
import collections
import concurrent.futures
import csv
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LOCAL = REPO / 'local/checkpoint-location-index/20260918-v1'
SCANS = LOCAL / 'scans'
OLD_MAP = 'transfers/verifications/2026-09-11-checkpoint-migration-server2-destination/migration-map.csv'
TRANSFER_ROOT = '/data/janghj/ODE-edit/local/transfers/20260918-official-lifelong-server4-to-server2-r1/'


def sha(b):
    return hashlib.sha256(b).hexdigest()


def write_json(p, data):
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def csv_write(name, rows, columns=None):
    if columns is None:
        columns = list(dict.fromkeys(k for r in rows for k in r))
    with (HERE / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def remote(server, source):
    cmd = ['python3', '-']
    if server != 'server1':
        cmd = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'rke-' + server, 'python3 -']
    p = subprocess.run(cmd, input=source, text=True, capture_output=True, timeout=180)
    if p.returncode:
        raise RuntimeError(server + ': ' + p.stderr[-1200:])
    return json.loads(p.stdout)


def collect_metadata():
    target = LOCAL / 'transfer-metadata.json'
    if target.exists():
        return json.loads(target.read_text())
    old = subprocess.check_output(['git', 'show', 'origin/main:' + OLD_MAP], cwd=REPO)
    main = subprocess.check_output(['git', 'rev-parse', 'origin/main'], cwd=REPO, text=True).strip()
    source = '''import json,hashlib,os
root=ROOT
out={}
for name in ['source-manifest.json','transfer-command.json']:
 p=root+name
 b=open(p,'rb').read()
 out[name]={'path':p,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'data':json.loads(b)}
print(json.dumps(out))
'''.replace('ROOT', repr(TRANSFER_ROOT))
    data = {'old_map_path': OLD_MAP, 'old_map_git_commit': main, 'old_map_sha256': sha(old),
            'old_map': list(csv.DictReader(io.StringIO(old.decode()))), 'current': remote('server4', source)}
    write_json(target, data)
    return data


def migration_rows(metadata):
    rows = []
    for r in metadata['old_map']:
        rows.append({'migration': '2026-09-11', 'arm': r['arm'], 'batch': r['batch'],
                     'source': r['source'], 'destination': r['destination'], 'incoming': '',
                     'expected_bytes': int(r['bytes']), 'recorded_sha256': r['sha256']})
    curr = metadata['current']['source-manifest.json']['data']
    command = metadata['current']['transfer-command.json']['data']['command']
    source = command[-2].rstrip('/')
    incoming = command[-1].split(':', 1)[1].rstrip('/')
    for r in curr['files']:
        if not r['path'].endswith(('.pt', '.pth', '.ckpt', '.safetensors', '.bin')):
            continue
        rows.append({'migration': '2026-09-18', 'arm': '/'.join(r['path'].split('/')[:2]),
                     'batch': Path(r['path']).stem, 'source': source + '/' + r['path'],
                     'destination': curr['destination'].rstrip('/') + '/' + r['path'],
                     'incoming': incoming + '/' + r['path'], 'expected_bytes': r['bytes'],
                     'recorded_sha256': r['sha256']})
    return rows


def role(row):
    p = row['path'].lower()
    n = Path(p).name
    k = set(row.get('schema_keys', '').split('|'))
    if n.endswith(('.json', '.csv', '.sha256', '.md')):
        return 'SIDECAR_METADATA'
    if n.startswith('.') or any(v in n for v in ('.partial', '.incomplete', '.tmp', '.part.')):
        return 'PARTIAL_OR_TEMP'
    if any(v in p for v in ('/cpu-fixtures/', '/synthetic-guard-fixture', '/tests/e2e/', '/test-fixtures/')):
        return 'TEST_FIXTURE'
    if '/.cache/huggingface/' in p or n.startswith(('pytorch_model', 'model-000')):
        return 'PRETRAINED_MODEL'
    if n in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'training_args.bin'):
        return 'TRAINING_CHECKPOINT_SUPPORT'
    if n == 'adapter_model.safetensors':
        return 'TRAINED_ADAPTER'
    if any(v in n for v in ('teacher', 'gradient', 'jacobian', 'null_space', 'mom2', 'factors', 'direction', 'geometry', 'p-star', 'calibration')) or any(v in p for v in ('/stats/', '/teacher', '/demo_vector/')):
        return 'AUXILIARY_ASSET'
    if 'history' in n:
        return 'HISTORY_COMPONENT'
    if any(v in n for v in ('target', 'readout', 'keys', 'kstate')) or '/snapshots/z' in p:
        return 'TARGET_KEY_OR_Z_ARTIFACT'
    if any(v in n for v in ('increment', 'delta', 'native-evidence')):
        return 'DELTA_OR_RECONSTRUCTION_COMPONENT'
    if n.startswith(('prepared', 'native-capsule', 'own-n4')) or n == 'entry.pt':
        return 'PREPARED_OR_NATIVE_CAPSULE'
    if k & {'weights', 'selected_weights', 'state_dict', 'model_state_dict', 'W', 'W4', 'W8', 'WN', 'weight', 'weight_after', 'weights_after'}:
        return 'WEIGHT_STATE_INDICATORS'
    if re.search(r'endpoint|snapshot|checkpoint|^node[-_0-9]|^state[._-]|^w\d*[-_]m|best_model', n) or '/checkpoints/' in p:
        return 'NAME_ONLY_CHECKPOINT_CANDIDATE'
    if n.endswith('.safetensors'):
        return 'MODEL_OR_TEST_WEIGHT'
    if k & {'delta', 'delta_W'}:
        return 'DELTA_OR_RECONSTRUCTION_COMPONENT'
    if row['category'].startswith('AUXILIARY'):
        return 'AUXILIARY_ASSET'
    return 'UNCLASSIFIED_TENSOR'


INDEX_ROLES = {'TRAINED_ADAPTER', 'WEIGHT_STATE_INDICATORS', 'NAME_ONLY_CHECKPOINT_CANDIDATE',
               'PREPARED_OR_NATIVE_CAPSULE', 'MODEL_OR_TEST_WEIGHT'}


def family(path):
    p = path
    if '/local/' in p:
        marker = next((m for m in ('/payload/local/', '/closure/local/') if m in p), '/local/')
        tail = p.split(marker, 1)[1].split('/')
        if tail[:3] == ['results', 'raw', 'experiments'] and len(tail) > 3:
            return 'reflection-based-ke/' + tail[3]
        if tail[0] in {'results', 'state', 'imports'} and len(tail) > 1:
            return '/'.join(tail[:2])
        return tail[0]
    if '/logs/' in p:
        return 'training-logs/' + p.split('/logs/', 1)[1].split('/')[0]
    if '/00.KE/' in p:
        return '00.KE/' + p.split('/00.KE/', 1)[1].split('/')[0]
    base = p.split('/janghj/', 1)[-1].split('/')[0]
    return base


def collect_recheck(scans, migrations):
    outpath = LOCAL / 'final-stat-recheck.json'
    if outpath.exists():
        return json.loads(outpath.read_text())
    wanted = {s['server']: {r['path'] for r in s['files']} for s in scans}
    for r in migrations:
        wanted['server4'].add(r['source'])
        wanted['server2'].add(r['destination'])
        if r['incoming']:
            wanted['server2'].add(r['incoming'])
    def run(item):
        server, paths = item
        code = '''import os,json,datetime
paths=PATHS
rows=[]
for p in paths:
 r={'path':p}
 try:
  s=os.stat(p)
  r.update(status='PRESENT',bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev,nlink=s.st_nlink)
 except FileNotFoundError:r['status']='ABSENT'
 except OSError as e:r.update(status='ERROR',error=str(e))
 rows.append(r)
print(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':rows}))
'''.replace('PATHS', repr(sorted(paths)))
        return server, remote(server, code)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        result = dict(pool.map(run, wanted.items()))
    write_json(outpath, result)
    return result


def main():
    scans = [json.loads(p.read_text()) for p in sorted(SCANS.glob('server*-scan.json'))]
    assert len(scans) == 4 and all('files' in s for s in scans)
    metadata = collect_metadata()
    migrations = migration_rows(metadata)
    checks = collect_recheck(scans, migrations)
    lookup = {s: {r['path']: r for r in data['rows']} for s, data in checks.items()}
    bindings = {}
    for m in migrations:
        for server, field in [('server4', 'source'), ('server2', 'destination'), ('server2', 'incoming')]:
            if m[field]:
                bindings[(server, m[field])] = m
    flat = []
    for s in scans:
        for raw in s['files']:
            r = dict(raw)
            r['role'] = role(r)
            r['experiment_family'] = family(r['path'])
            r['lifelong_or_sequential_path_hint'] = bool(re.search('lifelong|sequential|seq1000|seq10|single-layer-zflow|blue-checkpoint-downstream|alpha-jv-migration', r['path']))
            c = lookup[s['server']][r['path']]
            r['last_stat_at'] = checks[s['server']]['at']
            r['last_stat_status'] = c['status']
            r['last_stat_bytes'] = c.get('bytes', '')
            r['last_stat_device'] = c.get('device', '')
            r['last_stat_inode'] = c.get('inode', '')
            r['changed_since_scan'] = c['status'] != 'PRESENT' or any(c.get(k) != r.get(k) for k in ('bytes', 'mtime_ns', 'inode', 'device'))
            r['payload_sha256_status'] = 'NOT_RECOMPUTED'
            r['restore_validation'] = 'NOT_PERFORMED'
            flat.append(r)
    known = {(r['server'], r['path']) for r in flat}
    for m in migrations:
        for server, field in [('server4', 'source'), ('server2', 'destination'), ('server2', 'incoming')]:
            path = m[field]
            if not path:
                continue
            st = lookup[server][path]
            m[field + '_status'] = st['status']
            m[field + '_bytes_now'] = st.get('bytes', '')
            m[field + '_size_matches_record'] = st.get('bytes') == m['expected_bytes'] if st['status'] == 'PRESENT' else False
            if st['status'] == 'PRESENT' and (server, path) not in known:
                flat.append({'server': server, 'path': path, 'bytes': st['bytes'], 'device': st['device'], 'inode': st['inode'],
                             'role': 'NAME_ONLY_CHECKPOINT_CANDIDATE', 'experiment_family': family(path),
                             'availability': 'PRESENT_AT_FINAL_STAT_ONLY', 'observed_at': checks[server]['at'],
                             'last_stat_at': checks[server]['at'], 'last_stat_status': st['status'], 'last_stat_bytes': st['bytes'],
                             'schema_status': 'NOT_INSPECTED_NEW_TRANSFER_PATH', 'lifelong_or_sequential_path_hint': True,
                             'payload_sha256_status': 'MANIFEST_RECORDED_NOT_RECOMPUTED', 'restore_validation': 'NOT_PERFORMED'})
                known.add((server, path))
        m['verification_scope'] = 'CURRENT_STAT_SIZE_ONLY; SHA_IS_HISTORICAL_MANIFEST_NOT_CURRENT_REHASH'
    for r in flat:
        p = r['path']
        r['storage_location_kind'] = ('INCOMING_NOT_SEALED' if '.incoming-' in p else
            'ARCHIVE_OR_IMPORTED_COPY' if any(x in p for x in ('/checkpoint-archives/', '/imports/')) else
            'WORKTREE_LOCAL' if '/worktrees/' in p else 'RESEARCH_DIRECTORY')
        m = bindings.get((r['server'], p))
        r['recorded_arm'] = m['arm'] if m else ''
        r['recorded_batch'] = m['batch'] if m else ''
        r['recorded_payload_sha256'] = m['recorded_sha256'] if m else ''
        r['recorded_hash_provenance'] = 'migration-manifest-' + m['migration'] if m else ''
        r['batch_step_path_hint'] = '|'.join(re.findall(r'(?:^|/)((?:B\d+|checkpoint-\d+|state-\d+|snapshot-\d+|W\d+-M\d+)(?:\.pt)?)(?=/|$)', p))
    index = sorted([r for r in flat if r['role'] in INDEX_ROLES], key=lambda r: (r['server'], r['experiment_family'], r['path']))
    components = [r for r in flat if r['role'] in {'DELTA_OR_RECONSTRUCTION_COMPONENT', 'HISTORY_COMPONENT', 'TRAINING_CHECKPOINT_SUPPORT'}]
    csv_write('checkpoint-index.csv', index)
    csv_write('lifelong-sequential-index.csv', [r for r in index if r.get('lifelong_or_sequential_path_hint')])
    csv_write('checkpoint-components.csv', components)
    csv_write('all-tensor-artifacts.csv', flat)
    csv_write('migration-status.csv', migrations)
    groups = collections.defaultdict(list)
    for r in index:
        groups[(r['server'], r['experiment_family'])].append(r)
    summary = []
    for (server, fam), rows in sorted(groups.items()):
        present = [r for r in rows if r['last_stat_status'] == 'PRESENT']
        unique = {(r.get('last_stat_device', r.get('device')), r.get('last_stat_inode', r.get('inode'))): r for r in present}
        summary.append({'server': server, 'experiment_family': fam, 'indexed_paths': len(rows),
                        'present_paths': len(present), 'unique_inodes': len(unique),
                        'unique_logical_bytes': sum(int(r.get('last_stat_bytes') or 0) for r in unique.values()),
                        'roles': '|'.join(sorted({r['role'] for r in rows})), 'example_path': rows[0]['path']})
    csv_write('experiment-summary.csv', summary)
    matrix = []
    for fam in sorted({r['experiment_family'] for r in summary}):
        matrix.append({'experiment_family': fam, **{f'server{i}': sum(r['present_paths'] for r in summary if r['server'] == f'server{i}' and r['experiment_family'] == fam) for i in range(1, 5)}})
    csv_write('experiment-server-matrix.csv', matrix)
    servers = []
    for s in scans:
        rows = [r for r in index if r['server'] == s['server']]
        present = [r for r in rows if r['last_stat_status'] == 'PRESENT']
        unique = {(r.get('last_stat_device', r.get('device')), r.get('last_stat_inode', r.get('inode'))): r for r in present}
        servers.append({'server': s['server'], 'hostname': s['hostname'], 'indexed_paths': len(rows),
                        'present_paths': len(present), 'unique_inodes': len(unique),
                        'unique_logical_GiB': round(sum(int(r.get('last_stat_bytes') or 0) for r in unique.values()) / 2**30, 3),
                        'tensor_candidates_scan': len(s['files']), 'scan_errors': len(s['errors']),
                        'started_at': s['started_at'], 'finished_at': s['finished_at'], 'final_stat_at': checks[s['server']]['at']})
    csv_write('server-summary.csv', servers)
    coverage = [{k: v for k, v in s.items() if k not in ('files', 'metadata_candidates', 'directory_aliases')} for s in scans]
    write_json(HERE / 'scan-coverage.json', coverage)
    csv_write('directory-aliases.csv', [{'server': s['server'], **r} for s in scans for r in s['directory_aliases']])
    totals = collections.Counter(r['role'] for r in flat)
    write_json(HERE / 'classification-counts.json', dict(sorted(totals.items())))
    migration_summary = []
    for period in sorted({r['migration'] for r in migrations}):
        rs = [r for r in migrations if r['migration'] == period]
        migration_summary.append({'period': period, 'members': len(rs), 'expected_GiB': round(sum(r['expected_bytes'] for r in rs)/2**30, 3),
                                  **{f: sum(r.get(f + '_status') == 'PRESENT' for r in rs) for f in ('source', 'destination', 'incoming')},
                                  **{f + '_size_match': sum(r.get(f + '_size_matches_record', False) for r in rs) for f in ('source', 'destination', 'incoming')}})
    def table(header, rows):
        return '\n'.join(['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(['---']*len(header)) + ' |'] + ['| ' + ' | '.join(str(v).replace('|', '/') for v in r) + ' |' for r in rows])
    t_server = table(['서버', '목록 경로', '최종 존재', '고유 inode', '고유 논리 GiB', 'scan 오류'],
                     [[r[k] for k in ('server', 'indexed_paths', 'present_paths', 'unique_inodes', 'unique_logical_GiB', 'scan_errors')] for r in servers])
    t_migration = table(['이관 목록', 'CP 수', 'GiB', 'S4 원본 존재', 'S2 최종 존재/크기일치', 'S2 incoming 존재/크기일치'],
                        [[r['period'], r['members'], r['expected_GiB'], r['source'], f"{r['destination']}/{r['destination_size_match']}", f"{r['incoming']}/{r['incoming_size_match']}"] for r in migration_summary])
    ode = [r for r in summary if not r['experiment_family'].startswith(('training-logs/', '00.KE/', 'course_work'))]
    t_exp = table(['서버', '실험/보관 계열', '존재 경로', '고유 GiB'],
                  [[r['server'], r['experiment_family'], r['present_paths'], round(r['unique_logical_bytes']/2**30, 3)] for r in ode])
    first = min(s['started_at'] for s in scans)
    last = max(d['at'] for d in checks.values())
    text = f'''# 전체 서버 checkpoint 위치 인덱스 — 2026-09-18

## 결론과 사용법

GH가 server1·2·3·4의 사용자 연구 저장소를 직접 읽기 전용 조사했다. **이 보고서는 위치 인덱스이며 복원 성공 인증 또는 삭제 승인서가 아니다.** 원본/이관/실험/작업 상태를 변경하지 않았다.

조사 시작 {first}, 최종 경로 재확인 {last} (UTC; 한국시간은 +9시간). 이관·새 결과 생성 중의 비원자적 snapshot이므로 이후 경로 변화는 반영하지 않는다.

- [checkpoint 전체 위치 CSV](checkpoint-index.csv): 모델 weight 지표·endpoint/snapshot 이름·native 준비 capsule·학습 adapter를 포괄하는 **후보 인덱스**. 실제 서버와 절대 경로로 조회한다.
- [lifelong/sequential 위치 CSV](lifelong-sequential-index.csv): 경로 이름으로 추린 부분집합이며 전체 chain 성공을 뜻하지 않는다.
- [실험 × 서버 위치표](experiment-server-matrix.csv), [실험별 위치 요약](experiment-summary.csv), [서버 요약](server-summary.csv).
- [delta/history/optimizer 구성요소](checkpoint-components.csv): 단독 완전 checkpoint와 분리.
- [모든 tensor 파일 후보](all-tensor-artifacts.csv): target·teacher·pretrained·미분류·부분 파일도 누락 없이 남긴 확장 목록.
- [이관 출발지→최종/임시 목적지](migration-status.csv): 기록된 SHA와 **이번 stat/size 관측**을 구분.

## 서버별 결과

{t_server}

`목록 경로`는 완전 checkpoint 개수가 아니다. 명시적 weight schema 지표, 이름만 있는 미확인 snapshot, 준비 capsule, LoRA adapter를 포함한다. 파일명만 같은 사본을 합치지 않았다. 고유 inode는 같은 서버 내 hardlink/동일 파일 경로 중복만 제외한다. GiB는 파일 논리 크기이며 실제 회수 가능한 디스크 공간이 아니다. 역할별 분류는 [classification-counts.json](classification-counts.json)에 있다.

server3도 SSH/디렉터리 조사에 성공했다. 지정 범위에서 실험 checkpoint 후보는 발견되지 않았고 pretrained·projector·demo vector만 확인했다. 이를 서버 전체 디스크에 checkpoint가 전혀 없다는 주장으로 확대하지 않는다.

## 진행 중인 server4 → server2 이관

{t_migration}

2026-09-18 대상은 `official-layer-realization-debt-lifelong-b100-v1`이다. source manifest의 tensor checkpoint 행만 집계했다. server4 원본은 `/data/janghj/ODE-edit/local/results/official-layer-realization-debt-lifelong-b100-v1/`, server2 최종 예정지는 `/mnt/raid5/janghj/ODE-edit/local/results/official-layer-realization-debt-lifelong-b100-v1/`, 임시 수신지는 같은 경로에 `.incoming-server4-20260918-r1`을 붙인 디렉터리다.

**incoming 파일이 존재하거나 크기가 일치해도 이관 완료/검증 완료가 아니다.** 전송 담당자의 full SHA·closure·seal receipt 확인은 이 조사에서 대체하지 않았다. `.state-*.pt.<suffix>` 같은 임시 파일은 checkpoint 본수에서 제외하고 확장 목록에 보존했다. 현재 이관에는 명령·삭제·이동·대기·재시도를 추가하지 않았다.

2026-09-11 기존 183개 migration-map도 현재 출발/목적지 stat에 대조했다. 과거 map의 SHA는 과거 기록이지 이번 새 재해시가 아니다. 경로가 사라졌다면 다른 위치 이동/삭제/재보관 원인은 이 인덱스만으로 판단하지 않는다. 서로 다른 서버의 실제 byte equality나 마지막 유일 사본 여부를 보장하지 않는다.

## 연구 계열별 위치

아래는 ODE 관련 계열 중심 요약이다. 전체 학습 adapter·별도 연구 디렉터리는 experiment-summary.csv에 함께 있다. archive/payload 안의 원 실험명은 논리적 계열로 묶되 **현재 위치는 CSV의 server/path가 정본**이다.

{t_exp}

## 분류와 복원 한계

- `WEIGHT_STATE_INDICATORS`: 제한된 ZIP data.pkl opcode에서 W/weights/state_dict 등의 문자열 지표가 발견됨. top-level schema 인증이나 tensor finite/shape 검증은 아니다.
- `NAME_ONLY_CHECKPOINT_CANDIDATE`: endpoint/state/node/snapshot 이름 근거이며 내용 확인 부족. target-only일 수 있다.
- `PREPARED_OR_NATIVE_CAPSULE`: 준비/원 native endpoint 보조 묶음. commit 완료 또는 정확 sequential resume 가능 상태라는 뜻이 아니다.
- `TRAINED_ADAPTER`: adapter 파일. base model/config/tokenizer 등 companion이 추가 필요하다.
- delta/history/optimizer/target/key/teacher/statistics는 별도 목록으로 남겼다. `actual-increments`, `selected-delta`에 weight 지표가 있더라도 재구성 component로 보수적으로 분류했다.
- checkpoint 개수는 실험 수·독립 endpoint 수와 다르다. 실패/취소/technical/copy/backup 역시 제외하지 않고 경로 그대로 표시했다. 성능에 따른 선택은 하지 않았다.
- 명시적 CPU/synthetic/test fixture는 checkpoint 본수에서 제외하고 확장 목록의 `TEST_FIXTURE`로 보존했다. `recorded_arm/batch/payload_sha256`은 기존 migration manifest로 결속되는 파일에만 채웠으며, 나머지 `batch_step_path_hint`는 경로 힌트일 뿐이다. 실제 shape나 완료 여부를 추정하지 않았다.
- schema 읽기는 torch/pickle load 없이 pickle opcode만 사용했다. 대형 payload SHA·tensor 로딩·GPU·scheduler·모델 평가·resume 검증은 0회다. 일부 schema의 `NOT_INSPECTED`/오류는 CSV에 그대로 있다.

## 조사 범위와 사각지대

server1/2의 `/mnt/raid5/janghj`, server3/4의 `/data/janghj` 아래 비숨김 연구 디렉터리 전체, `.codex/worktrees`, `.cache/huggingface`를 조사했다. 반대 root 존재 여부도 확인했다. 접근 오류·제외 규칙·시작/종료 시각은 [scan-coverage.json](scan-coverage.json), symlink 대체 경로는 [directory-aliases.csv](directory-aliases.csv)에 기록했다.

`.git`, 설치 환경·package/cache 디렉터리, 별도 숨김 디렉터리, 사용자 root 밖 symlink, 다른 사용자/별도 미등록 mount는 포함하지 않는다. `.pt/.pth/.ckpt/.safetensors/.bin/.npz/.npy` 및 일부 backup/partial 확장자를 대상으로 했으며, 임의 확장자·확장자 없는 모델·다른 포맷·메타데이터만 있는 checkpoint는 완전 탐지하지 못한다. 따라서 **확인된 네 서버의 명시적 연구 범위 전수 파일 탐색**이며 모든 저장장치 무제한 전수조사 주장은 아니다.

수집 중 파일 변화는 `stable_during_inspection`, `changed_since_scan`, `last_stat_status`로 구분한다. manifest를 통해 최종 재확인에서 새로 발견한 수신 파일은 `PRESENT_AT_FINAL_STAT_ONLY`/schema 미검사로 표시한다. 새로 생성됐지만 첫 scan과 manifest 양쪽에 없던 파일은 다음 갱신 대상이다. 본 보고 이후 상태를 자동 추적하지 않는다.

## 재현과 증거

수집기 [checkpoint_inventory.py](checkpoint_inventory.py), 보고 빌더 [build_index.py](build_index.py). 최초 read-only scan은 아래 명령으로 새 디렉터리에 수행할 수 있다. 수집기는 create-once이며 기존 scan 덮어쓰기를 거부한다.

```bash
python3 checkpoint_inventory.py --collect /absolute/new-scan-directory
```

이번 raw metadata(모델/프롬프트 payload 아님)는 `{LOCAL}/`에 보존했다. transfer manifest 두 개만 추가 읽었으며 Git의 기존 migration CSV를 재사용했다. 최종 재확인은 명시적 경로 stat만 수행했다. 빌더는 이 날짜의 고정 입력을 사용하므로 새 조사 시 SCANS/LOCAL과 날짜를 바꿔 별도 version으로 생성한다. 보고 파일/수집 입력 SHA는 [report-manifest.json](report-manifest.json)에 있다.

기존 실험 결과·worktree·이관 작업·Slurm·raw·checkpoint를 수정/삭제하지 않았으며 main push는 수행하지 않았다.
'''
    (HERE / 'checkpoint-location-report-ko.md').write_text(text)
    # Invariants are bookkeeping tests, not checkpoint validity tests.
    assert len({(r['server'], r['path']) for r in flat}) == len(flat)
    assert sum(r['indexed_paths'] for r in servers) == len(index)
    assert len([r for r in migrations if r['migration'] == '2026-09-11']) == 183
    assert all(r['role'] not in {'PARTIAL_OR_TEMP', 'SIDECAR_METADATA'} for r in index)
    write_json(HERE / 'validation.json', {'unique_server_path': True, 'server_totals_reconcile': True,
               'historical_map_rows_183': True, 'partials_excluded_from_checkpoint_index': True,
               'checkpoint_index_rows': len(index), 'all_artifact_rows': len(flat),
               'migration_summary': migration_summary, 'scientific_validation': 'NOT_PERFORMED'})
    inputs = list(SCANS.glob('*.json')) + [LOCAL/'transfer-metadata.json', LOCAL/'final-stat-recheck.json']
    files = [p for p in HERE.iterdir() if p.is_file() and p.name != 'report-manifest.json']
    write_json(HERE/'report-manifest.json', {'created_at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'inputs': [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p.read_bytes())} for p in inputs],
        'members': [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p.read_bytes())} for p in sorted(files)],
        'checkpoint_payload_rehash': False, 'transfer_evidence': {k:v for k,v in metadata.items() if k not in ('old_map','current')},
        'current_transfer_metadata': [{k:v for k,v in x.items() if k!='data'} for x in metadata['current'].values()]})
    print(json.dumps({'servers': servers, 'migration': migration_summary, 'roles': totals}, ensure_ascii=False))


if __name__ == '__main__':
    main()
