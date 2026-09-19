"""Compact requirement/source/evidence mapping for exact completed B1 review.

Metadata only: no model, runtime, scheduler, optimization or scientific writes.
"""
import argparse
from pathlib import Path
from .review_completed_b1 import csv_write, member

PREFIX = 'project/run_scripts/single_layer_mechanism_first/'

def run(frozen, package):
    frozen, package = Path(frozen), Path(package)
    rows = []
    def add(requirement, file, marker, evidence, status, level, limitation):
        path = frozen / (PREFIX + file)
        lines = path.read_text().splitlines()
        matches = [i+1 for i,line in enumerate(lines) if marker in line]
        if not matches:
            raise ValueError('SOURCE_MARKER_NOT_FOUND:' + file + ':' + marker)
        rows.append(dict(requirement=requirement, execution_commit='5f79085629b10b2bb8bdee88d017e18a46bb4c74',
            file=PREFIX+file, function_or_marker=marker, line=matches[0], source_sha256=member(path)['sha256'],
            evidence=evidence, status=status, verification_level=level, limitation=limitation))
    add('B1 only and no sequential', 'science.py', 'def batch(', 'terminal.json; execution.lock; B1-to-S3.json',
        'PASS', 'SOURCE_AND_TERMINAL', 'No S3/S10 or B2 observation')
    add('Pinned model FP32 eager TF32off W0 zeroM', 'model.py', 'class Runtime(', 'execution.lock; runtime manifest; batch-entry; native-binding',
        'PASS_BINDING', 'FROZEN_CONFIG_RUNTIME_RECEIPT', 'No independent model reload or whole-model parity')
    add('Fresh shared native100; correction extra z0', 'science.py', 'native=rt.native(', 'native-binding.json; batch-compute.json; selections',
        'PASS', 'SOURCE_COUNTER_REQUEST_ORDER', 'Same B1 native shared; no separate matched timing run')
    add('L4 only physical weight path', 'native.py', 'class CapturedHookedFitter(', 'native-binding; commit W identities',
        'PASS_BINDING', 'SOURCE_AND_RUNTIME_GUARD', 'Full selected weight not persisted')
    add('Prefix cache invalidation per native z call', 'z_hook.py', 'def capture_prefix(', 'native-binding z_hook batches100',
        'PASS_ACTUAL_BATCH1', 'SOURCE_AND_COUNTER', 'Batch16 not executed')
    add('Target full-vocab head and separate final KL hidden', 'z_hook.py', 'def native_losses(', 'prior zhook receipts; native-binding',
        'WARN_NUMERICAL', 'SOURCE_AND_PRIOR_ACTUAL_LIMITED_PANEL', 'Hook gradient 1.341096e-4 exceeded original1e-4; user warning preserved')
    add('Native Adam clamp stop and frozen converged rows', 'z_hook.py', 'def freeze_and_clamp(', '2500 loss;2400 Adam/clamp;100 capped requests',
        'PASS_OBSERVED_CAP_BRANCH', 'SOURCE_COUNTER', 'Early stop and batch16 frozen-row behavior not exercised in B1')
    add('All valid old/new Current K_E and Q intersection', 'science.py', 'def allowed_space(', 'geometry.json; protected keys4596; acceptance-arithmetic.json',
        'PASS_STORED_SCALARS', 'CPU_TENSOR_SCHEMA_AND_THRESHOLD_ARITHMETIC', 'No complete D or Q reconstruction; anchor per-sequence rows not retained')
    add('All R512 generated positions and full vocabulary', 'decision.py', 'class DecisionOracle:', 'reference-native.json; factors manifest; capsule-summary.json',
        'PASS_NATIVE_DERIVATIVE', 'CPU_ID_CARDINALITY_AND_RUNTIME_COUNTER', 'Not evidence of every selected policy choice scan')
    add('GPU FP64 dense-gradient accumulation', 'decision.py', 'class FactorArchive:', 'reference-native work; factor archive',
        'PASS_DECISION_COUNTER', 'SOURCE_AND_ACTUAL_BYTES', 'EN gradient final-transfer counter not stored; emptyPast zero-gradient transfer also occurred')
    add('One center and factor contraction reuse', 'science.py', 'def batch(', 'basis.json; jacobians.pt; factor inventory',
        'PASS_BINDING', 'CPU_FACTOR_SCHEMA_AND_SOURCE', 'No fresh AD/FD in this review')
    add('Functional rank4 plus covariance rank1', 'basis.py', 'def append_covariance_direction(', 'basis residuals; selection details; J schemas',
        'PASS', 'STORED_GRAM_SPAN_AND_SHAPE', 'B1 STEP=CUM alias; no cumulative advantage measured')
    add('Fixed local two-phase QCQP', 'solver.py', 'def solve_coefficients(', 'selection details solver; local-solver.json',
        'PASS_LOCAL_ARITHMETIC', 'INDEPENDENT_SAVED_COEFFICIENT_DUAL_ARITHMETIC', 'Local fixed linearized problem only; no global nonlinear optimum')
    add('EN KL actual Armijo first accepted of four', 'controller.py', 'def optimize_kl(', 'candidate-arithmetic.csv; acceptance-arithmetic.json',
        'PASS_STORED_ARITHMETIC', 'INDEPENDENT_LOSS_ARMIJO_SCALARS', 'G/H/dense trial tensors not persisted')
    add('DEC full512 acceptance after actual movement', 'controller.py', 'def optimize_decision(', 'eight FP32_NO_MOVE trials; zero candidate scans',
        'NOT_EXERCISED', 'SOURCE_AND_COUNTER', 'full512 flag refers native coverage; no candidate acceptance PASS')
    add('EN selected R512 decision choice observer', 'observers.py', 'def evaluate(', 'reference-coverage.json; postseal KL only',
        'NOT_MEASURED', 'ACTUAL_MISSING_OBSERVATION', 'Do not infer choice flips or Phi from KL')
    add('Current individual NLL and strict/pair IDs', 'current.py', 'class CurrentGuard:', 'EN trial4 guard and invariant; canonical reducer',
        'PASS_STORED_SCALARS_AND_IDS', 'THRESHOLD_RECHECK_PLUS_INDEPENDENT_FINAL_NLL', 'Full anchor rows absent for independent per-sequence subtraction')
    add('B1 active-history panel', 'history.py', 'def active_history(', 'history-entry/native empty; gradient backward0',
        'NOT_APPLICABLE_EMPTY_B1', 'SOURCE_AND_RECEIPT', 'Nonempty history/B2 protection not tested')
    add('History once selected endpoint and branch restoration', 'transaction.py', 'def commit(', 'four commit receipts; before/after M; W/RNG identities',
        'PASS_RECEIPT', 'CPU_LINK_AND_COUNTER', 'No disk W/M; observer separate M pre/post hashes not recorded')
    add('No disk checkpoint or disguised full state', 'science.py', 'def _tensor_summary(', 'terminal noCP;522 pt schema inventory',
        'PASS_SCOPE_INVENTORY', 'CPU_WEIGHTS_ONLY_MMAP', 'Exact resume unavailable; no independent selected tensor reconstruction')
    add('Official P/N and Dev postseal only', 'observers.py', 'def evaluate(', 'selection seals; observer complete; controller observer_accesses0',
        'PASS_SOURCE_RECEIPT', 'SOURCE_AND_LEDGER', 'Review did not rerun model or prove unrecorded branches')
    add('B1 mechanical expansion gate', 'gates.py', 'def b1_gate(', 'stage-gate.json',
        'FAIL_QUALITY_GATE', 'INDEPENDENT_METRIC_AND_STORED_SELECTION_ARITHMETIC', 'Nonzero and choice-or-margin fail; no expansion authority anyway')
    add('Writer exact source order and low-rank diagnostics', 'mechanism.py', 'def analyze_writer(', 'writer/modes/control; tensor inventory',
        'PASS_BOUNDED_ALGEBRA', 'CPU_REDUCTION_AND_SAVED_RUNTIME_REPLAY', 'Ideal factor delta differs from actual FP32 by relative6.073846e-6')
    add('Writer total timer', 'mechanism.py', 'for start in range(0,m,128):', 'timing-exclusions.json',
        'FAIL_INSTRUMENTATION', 'SOURCE_BACKED_RCA', 'Loop overwrites monotonic start; invalid7403588.902935sec excluded')
    add('Postselection component panel', 'postselection.py', 'def mechanism_report(', 'postselection-components.csv',
        'PASS_PANEL_REDUCTION', 'INDEPENDENT_RAW_SEQUENCE_NLL_AND_CHOICE_ROWS', '32reference+32N development panel; not full policy endpoint or unbiased test')
    add('T0 FD original threshold', 'reuse_t0_user_waiver.py', 'def reuse(', 'completed-T0-reuse; prior failed T0; waiver lock',
        'WAIVED_USER_DIRECTED', 'PRIOR_ACTUAL_EVIDENCE_REUSE', '3/4 adjacent FD criterion not established; no full numerical PASS')
    csv_write(package/'source-conformance.csv', rows)
    coverage = [
        ('H1','OBSERVED_BOUNDED','writer-modes.csv; mechanism-algebra.json; native-capture tensor schema',
         '100 ideal-metric modes; target loading; raw-map realization; fixed-K residual permutation',
         'Actual operator replay is prior runtime evidence; no causal efficacy or full SVD claim'),
        ('H2','OBSERVED_BOUNDED','mechanism-algebra.json; postselection-components.csv',
         '512 action rows; signed response;32 reference/32N interventions',
         'Energy is not locality; selected panel not independent broad test'),
        ('H3','PARTIAL','acceptance-arithmetic.json; KL.csv; Dev128-transitions.json',
         'EN actual protected response and current guard pass; train KL decreases; official success IDs unchanged',
         'Selected R512 decision Phi missing; Dev has2newflip/2recovery; FD unresolved'),
        ('H4','OBSERVED_NO_REALIZED_DEC_CORRECTION','local-solver.json; candidate-arithmetic.csv',
         'LINE rank1 versus CUM rank5; both local solutions materialize to zero',
         'No nonzero endpoint contrast; finite local result does not establish space-wide impossibility'),
        ('H5','NOT_APPLICABLE_B1','stage-gate.json; terminal B1 only',
         'Entry displacement0 and cross term0; STEP=CUM alias',
         'No B2 or same-entry STEP/CUM probe; no cumulative or lifelong inference'),
    ]
    csv_write(package/'mechanism-coverage.csv',
        [dict(hypothesis=h,status=s,evidence=e,observed=o,limitation=l) for h,s,e,o,l in coverage])
    return dict(conformance_rows=len(rows),hypotheses=len(coverage))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--frozen',required=True);p.add_argument('--package',required=True)
    a=p.parse_args();print(run(a.frozen,a.package))
