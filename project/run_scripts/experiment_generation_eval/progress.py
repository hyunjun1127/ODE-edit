"""Boundary-only scalar progress; no model calls, timer, polling or RNG use."""
import math
import time

from .common import require

PHASES = ('W0_generation', 'generation_evaluation')
FIELDS = ('completed_cases', 'total_cases', 'completed_prompts', 'total_prompts',
          'generated_tokens', 'new_cases', 'reused_cases', 'elapsed_sec',
          'cases_per_sec', 'prompts_per_sec', 'tokens_per_sec',
          'physical_forward_calls', 'prefill_query_tokens', 'decode_query_tokens', 'step')


class GenerationProgress:
    def __init__(self, total_cases, total_prompts, callback=None, *, phase='W0_generation',
                 clock=time.monotonic, interval_seconds=15, prompt_interval=64, first_step=0):
        require(phase in PHASES, 'GENERATION_PROGRESS_PHASE')
        require(type(total_cases) is int and total_cases >= 0
                and type(total_prompts) is int and total_prompts >= 0,
                'GENERATION_PROGRESS_TOTALS')
        self.callback, self.phase, self.clock = callback, phase, clock
        self.interval_seconds, self.prompt_interval = interval_seconds, prompt_interval
        self.started = self.last_time = clock()
        self.last_prompts = 0
        self.step = first_step
        self.counts = dict(completed_cases=0, total_cases=total_cases,
            completed_prompts=0, total_prompts=total_prompts, generated_tokens=0,
            new_cases=0, reused_cases=0, physical_forward_calls=0,
            prefill_query_tokens=0, decode_query_tokens=0)

    def emit(self, event, *, force=False):
        require(event in ('start', 'boundary', 'final', 'error'), 'GENERATION_PROGRESS_EVENT')
        now = self.clock()
        if not force and now-self.last_time < self.interval_seconds \
                and self.counts['completed_prompts']-self.last_prompts < self.prompt_interval:
            return None
        elapsed = max(0.0, now-self.started)
        c = dict(self.counts, elapsed_sec=elapsed,
            cases_per_sec=self.counts['completed_cases']/elapsed if elapsed else 0.0,
            prompts_per_sec=self.counts['completed_prompts']/elapsed if elapsed else 0.0,
            tokens_per_sec=self.counts['generated_tokens']/elapsed if elapsed else 0.0,
            step=self.step)
        require(set(c) == set(FIELDS) and all(type(x) in (int, float) and math.isfinite(x)
            and x >= 0 for x in c.values()), 'GENERATION_PROGRESS_SCALAR_PRIVACY')
        payload = {'generation_progress/'+k: v for k,v in c.items()}
        payload.update(phase=self.phase)
        # Only the caller's already-approved scalar transport receives this data.
        # Exception is not swallowed: callers must preserve a typed logging failure.
        if self.callback is not None:
            self.callback(payload)
        self.step += 1
        self.last_time, self.last_prompts = now, self.counts['completed_prompts']
        return payload

    def complete_case(self, raw, *, reused=False):
        self.counts['completed_cases'] += 1
        self.counts['reused_cases' if reused else 'new_cases'] += 1
        observations = raw['observations']
        self.counts['completed_prompts'] += len(observations)
        self.counts['generated_tokens'] += sum(r['continuation_token_count'] for r in observations)
        if not reused:
            self.counts['physical_forward_calls'] += sum(r.get('physical_forward_calls', r['model_forwards']) for r in observations)
            self.counts['prefill_query_tokens'] += sum(r.get('prefill_query_tokens', 0) for r in observations)
            self.counts['decode_query_tokens'] += sum(r.get('decode_query_tokens', r['full_prefix_token_work']) for r in observations)
        require(self.counts['completed_cases'] <= self.counts['total_cases']
                and self.counts['completed_prompts'] <= self.counts['total_prompts'],
                'GENERATION_PROGRESS_COUNT_OVERFLOW')
        return self.emit('boundary')

    def finish(self):
        require(self.counts['completed_cases'] == self.counts['total_cases']
                and self.counts['completed_prompts'] == self.counts['total_prompts'],
                'GENERATION_PROGRESS_INCOMPLETE')
        return self.emit('final', force=True)
