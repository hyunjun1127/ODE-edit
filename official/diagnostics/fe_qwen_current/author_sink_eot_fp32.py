"""FP32 / author numeric settings / current-entry z, with the Qwen EOT document prefix extended from the basic rule
(subject-last lookup at position 0) to the attention-sink region.

Qwen2.5 has no BOS token, so an unprefixed prompt puts the attention-sink massive activation on its opening token(s)
(W0 FP32 L4 block norm ~1.4e4 against a median of 41).  The basic rule (official.ours.pos0) prefixes <|endoftext|>
only when the subject-last lookup is 0.  Subjects that open the prompt can also put the sink on position 1: the W0
scan of the CF main stream found three such pairs the basic rule leaves on the sink (The Experiment / The Possible /
The Assault).  FE wrote the case-303 sink key ('{} was originally aired on', lookup 1, L4 init 13710) into all five
layers in CF B8, in both the BF16 (121242) and the FP32 (63809) author runs.

eot_policy (bindings, fixed per run; both are tokenizer-only and need no sink list):
  sink_region    prefix when the template opens with the subject and its last token falls on position 0 or 1
  subject_start  prefix whenever the template opens with the subject
  ours_w0_sink   (2026-10-13) exactly the rule of the paper's Ours runs (SUBJECT_LAST_NEVER_POSITION_0_OR_W0_SINK):
                 prefix when the subject-last lookup is 0 or the (text before '{}', subject) pair is in the W0 sink-scan
                 list bound by sha256 (CF f39f342e..., 15 pairs incl. The Experiment/Possible/Assault; zsRE 4f53cda1...,
                 empty), so FE edits exactly the rows Ours edits.
All contain the basic rule.  Only the prefix rule of the existing edit-side adapter changes (compute_z prompts,
replay supplier, writer key/output extraction): the adapter source, evaluator, author hparams (L2=0, clamp 1, ...),
main streams, z reference and writer are unchanged.  Reuses author_basic_eot_fp32 by patching its prefix rule,
schema, tracking and identity for the duration of one call.
"""
import argparse
import hashlib
import json
from contextlib import ExitStack
from unittest.mock import patch
from official.diagnostics.fe_qwen_current import author_basic_eot_fp32 as base
from official.diagnostics.fe_qwen_current import failure_eot_probe as eot
from official.ours.pos0 import PREFIX, PREFIX_IDS
from official.runners.server1.common import read, verify, member, require

SCHEMA = 'author-sink-eot-fp32-v1'
PROFILE = 'FE_AUTHOR_HPARAMS_CURRENT_Z_SINK_EOT_FP32'
POLICIES = {'sink_region': 'OURS_EOT_SUBJECT_IN_SINK_REGION_POS0_POS1', 'subject_start': 'OURS_EOT_SUBJECT_STARTS_PROMPT',
            'ours_w0_sink': 'OURS_SUBJECT_LAST_NEVER_POSITION_0_OR_W0_SINK'}
OURS_SINK_SHA = {'cf': 'f39f342ea750e410f50c1a118d5b520e82e3e4e3abb9a00202fcaf3179e8ed50',
                 'zsre': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'}


def pairs_sha(pairs):
    """As project.run_scripts.gptj_price_tier1k_sink.sink.pairs_sha (the Ours preparation's sink_pairs_sha256)."""
    return hashlib.sha256(json.dumps(sorted([list(p) for p in pairs]), separators=(',', ':')).encode()).hexdigest()


def prefix_rule(policy, pairs=None):
    """Drop-in for official.ours.pos0.repair_prompt with the same identity checks and a wider trigger."""
    require(policy in POLICIES and (pairs is not None) == (policy == 'ours_w0_sink'), 'EOT_POLICY')
    sinks = frozenset((str(a), str(b)) for a, b in (pairs or ()))

    def repair_prompt(tokenizer, prompt, subject, model, lookup):
        if model is None:
            return prompt
        if policy == 'ours_w0_sink':
            old = lookup(tokenizer, prompt, subject)
            if not (old == 0 or (prompt.split('{}', 1)[0], subject) in sinks):
                return prompt
        else:
            if not prompt.startswith('{}'):
                return prompt
            old = lookup(tokenizer, prompt, subject)
            if policy == 'sink_region' and old > 1:
                return prompt
        token = PREFIX_IDS[model]
        require(tokenizer.encode(PREFIX, add_special_tokens=False) == [token], 'POS0_PREFIX_TOKEN_IDENTITY')
        new = PREFIX + prompt
        require(tokenizer.encode(new.format(subject)) == [token] + tokenizer.encode(prompt.format(subject)),
                'POS0_PREFIX_RETOKENIZATION')
        require(lookup(tokenizer, new, subject) == old + 1, 'POS0_SUBJECT_SHIFT')
        # The valid prefix may equal PAD. Never derive masks from IDs.
        return new
    return repair_prompt


def _policy(b):
    policy = b['eot_policy']
    require(policy in POLICIES, 'EOT_POLICY')
    return policy


def _pairs(b, policy):
    if policy != 'ours_w0_sink':
        require('sink_pairs' not in b, 'SINK_PAIRS_ONLY_FOR_OURS_RULE');return None
    pairs = b['sink_pairs']
    require(isinstance(pairs, list) and all(isinstance(p, list) and len(p) == 2 for p in pairs)
            and pairs_sha(pairs) == b['sink_pairs_sha256'] == OURS_SINK_SHA[b['dataset']], 'OURS_SINK_PAIRS_IDENTITY')
    return pairs


def _patches(b):
    policy = _policy(b);pairs = _pairs(b, policy)
    original_tracking, original_identity = base.tracking, base.identity

    def tracking(bb, env=None):
        values = original_tracking(bb, env)
        values.update(task_id='fe-qwen-author-sink-eot-fp32-20261013',
                      arm=f'AUTHOR-{bb["dataset"]}-FP32-L2-0-clamp1-{policy}EOT',
                      baseline='FE-author-hparams-current-z-sinkEOT-FP32', evaluation_profile=PROFILE + '-' + policy)
        return values

    def identity(bb, resolved):
        value = original_identity(bb, resolved)
        value.update(stage=PROFILE, input_policy=POLICIES[policy], eot_policy=policy, variant=member(__file__))
        if pairs is not None:value.update(sink_pairs_sha256=pairs_sha(pairs), sink_pairs=len(pairs))
        return value

    return [patch.object(eot, 'repair_prompt', prefix_rule(policy, pairs)), patch.object(base, 'SCHEMA', SCHEMA),
            patch.object(base, 'tracking', tracking), patch.object(base, 'identity', identity),
            patch.object(base, 'PROFILE', PROFILE)]


def _with(b, fn, *args):
    with ExitStack() as stack:
        for p in _patches(b):
            stack.enter_context(p)
        return fn(*args)


def validate(bindings):
    b = read(bindings)
    return _with(b, base.validate, b)


def run(bindings, output):
    return _with(read(bindings), base.run, bindings, output)


def collect(result, output):
    proof = read(result)
    b = read(verify(proof['bindings']))
    return _with(b, base.collect, result, output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('validate'); r.add_argument('bindings')
    r = sub.add_parser('run'); r.add_argument('bindings'); r.add_argument('output')
    r = sub.add_parser('collect'); r.add_argument('result'); r.add_argument('output')
    a = p.parse_args()
    if a.command == 'validate':
        cfg, resolved, rows = validate(a.bindings)
        print(json.dumps(dict(status='VALID', rows=len(rows), eot_policy=read(a.bindings)['eot_policy'])))
    elif a.command == 'run':
        print(run(a.bindings, a.output))
    else:
        print(collect(a.result, a.output))
