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
Both contain the basic rule.  Only the prefix rule of the existing edit-side adapter changes (compute_z prompts,
replay supplier, writer key/output extraction): the adapter source, evaluator, author hparams (L2=0, clamp 1, ...),
main streams, z reference and writer are unchanged.  Reuses author_basic_eot_fp32 by patching its prefix rule,
schema, tracking and identity for the duration of one call.
"""
import argparse
import json
from contextlib import ExitStack
from unittest.mock import patch
from official.diagnostics.fe_qwen_current import author_basic_eot_fp32 as base
from official.diagnostics.fe_qwen_current import failure_eot_probe as eot
from official.ours.pos0 import PREFIX, PREFIX_IDS
from official.runners.server1.common import read, verify, member, require

SCHEMA = 'author-sink-eot-fp32-v1'
PROFILE = 'FE_AUTHOR_HPARAMS_CURRENT_Z_SINK_EOT_FP32'
POLICIES = {'sink_region': 'OURS_EOT_SUBJECT_IN_SINK_REGION_POS0_POS1', 'subject_start': 'OURS_EOT_SUBJECT_STARTS_PROMPT'}


def prefix_rule(policy):
    """Drop-in for official.ours.pos0.repair_prompt with the same identity checks and a wider trigger."""
    require(policy in POLICIES, 'EOT_POLICY')

    def repair_prompt(tokenizer, prompt, subject, model, lookup):
        if model is None or not prompt.startswith('{}'):
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


def _patches(b):
    policy = _policy(b)
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
        return value

    return [patch.object(eot, 'repair_prompt', prefix_rule(policy)), patch.object(base, 'SCHEMA', SCHEMA),
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
