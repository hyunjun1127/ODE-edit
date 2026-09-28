# 53996 실패 RCA / repair-r1

최신 nonce ODEEDIT-GH-SH3-MEMIT-HISTORY-REPAIR-INITIAL-GATE-20260928-R1, main2ddb30d5 FULL_READ.
실제 FAILED1:0, janghj/ubuntu,451 allocated GPU-sec(0.125278GPUh), batchMaxRSS16326420KiB.
B1 native100z/5solve/5history append 및 Current/full RPN 저장 뒤 provenance 수집에서
FileNotFoundError: blue-upstream/_classes.py. committed batch0, rollback W/H exact 및 RNG restore,
cleanup_error=null. OOM/수치발산/정상 fallback으로 분류하지 않는다.

PyTorch torch.classes는 __file__ 문자열 _classes.py를 가진 가상 module이며 해당 상대경로를
CWD에서 resolve한 기존 수집식이 실제 source로 오인했다. GPU없이 같은 오류 재현.
inspect.getattr_static으로 __file__를 읽고 absolute spec origin 또는 실제 absolute file만
scope hash에 포함한다. relative virtual module은 사유를 별도 기록한다.
실제 scope file 누락은 계속 실패한다. Native BLUE 및 수식·hparams 변경0.
최신 gate를 위해 W/H에 더해 context/RNG/ledger hash를 observer 전후 및 B1commit/B2entry에 결속.
CPU4checks PASS. 중간 CPU fixture의 class-level 속성 누락을 검출·수정한 이력도 보존.
noCP로 freshW0/H0 새chain; old edited state 재사용0. old source/raw/비용 불변.
새제출 전이며 초기 gate는 아직 NOT_OBSERVED. rerun 후 B1commit/observer→B2entry까지만 관찰한다.
