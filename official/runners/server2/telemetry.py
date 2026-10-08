"""Scalar-only tee for numeric values native code already prints.

No torch/SDK/model imports, tensor conversion, new forward or synchronization.
MEMIT/FE/Alpha/SPHERE/BLUE print rounded-three-decimal latent loss/NLL/KL;
FT prints its unrounded existing Batch loss (not an inferred NLL or KL).
Unknown text, including target/prompt/traceback, is only teed to local stdout.
The source exception and original stream are preserved on every exit path.
"""
import math
import re
import sys
import time

SCHEMA = 'official-server2-existing-native-print-telemetry-v1'
METHODS = ('FT','MEMIT','ALPHAEDIT','ALPHAEDIT_BLUE','MEMIT_FE','SPHERE')
NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
# Greedy bracket content accommodates native targets containing ']' while
# deliberately capturing none of that text. No raw match/group is retained.
LATENT = re.compile(r'^loss ('+NUMBER+r') = ('+NUMBER+r') \+ ('+NUMBER+r') \+ ('+NUMBER+
    r') avg prob of \[[^\r\n]*\] ('+NUMBER+r')$')
FT = re.compile(r'^Batch loss ('+NUMBER+r')$')
PAYLOAD_KEYS = {'batch','candidate','fit/global_candidate','fit/loss','fit/nll','fit/kl','phase_id'}
COUNTERS = ('global_candidate','parsed_loss_lines','fit_phase_starts','logged_points',
            'logging_errors','logging_rejections','oversize_lines','parser_errors')


def require(value,code):
    if not value:
        raise ValueError(code)


def parse(line,method):
    """Return only finite builtin scalars from exact known native formats."""
    if method == 'FT':
        match = FT.fullmatch(line)
        if not match:
            return None
        value = float(match[1])
        return {'fit/loss':value} if math.isfinite(value) else None
    match = LATENT.fullmatch(line)
    if not match:
        return None
    # Decay and average probability are validated as finite known numeric
    # tokens, but not uploaded: decay isn't allowed by the shared schema.
    values = [float(match[index]) for index in range(1,6)]
    if not all(math.isfinite(value) for value in values):
        return None
    return dict(zip(('fit/loss','fit/nll','fit/kl'),values[:3]))


class NativeTelemetry:
    """One native apply scope, same cursor reused across batches/resume.

    ``log_callback`` calls the common official.tracking transport; it is never
    another SDK implementation. ``cursor`` is a mutable scalar-only dict;
    persist ``snapshot()`` in the caller's checkpoint evaluation cursor.
    Candidate counts observed native loss evaluations, NOT Adam updates.
    """
    def __init__(self,log_callback,batch,*,cursor=None,method,interval_seconds=20,
                 stdout=None,clock=None,max_line_chars=8192):
        require(callable(log_callback) and method in METHODS,'TELEMETRY_CALLBACK_METHOD')
        require(type(batch) is int and 1 <= batch <= 20,'TELEMETRY_NATIVE_BATCH')
        require(type(interval_seconds) in (int,float) and 10 <= interval_seconds <= 30,
                'TELEMETRY_FIXED_THROTTLE_10_TO_30_SECONDS')
        require(type(max_line_chars) is int and 256 <= max_line_chars <= 65536,'TELEMETRY_LINE_RESERVE')
        self.callback,self.batch,self.method = log_callback,batch,method
        self.interval,self.clock = interval_seconds,clock or time.monotonic
        self.original = stdout
        self.cursor = {} if cursor is None else cursor
        require(type(self.cursor) is dict,'TELEMETRY_SCALAR_CURSOR')
        if self.cursor:
            require(self.cursor.get('schema') == SCHEMA and self.cursor.get('method') == method
                and set(self.cursor) == {'schema','method','batch','candidate',*COUNTERS},
                'TELEMETRY_CURSOR_IDENTITY_OR_UNKNOWN_DATA')
            require(all(type(self.cursor[key]) is int and self.cursor[key] >= 0
                for key in ('batch','candidate',*COUNTERS)),'TELEMETRY_CURSOR_INTEGER_COUNTS')
        else:
            self.cursor.update(schema=SCHEMA,method=method,batch=0,candidate=0,
                               **{key:0 for key in COUNTERS})
        if self.cursor['batch'] != batch:
            self.cursor['candidate'] = 0
        self.cursor['batch'] = batch
        self.maximum = max_line_chars
        self.line = ''
        self.dropping = self.sending = self.active = False
        self.pending = None
        self.last_sent = -math.inf
        self.last_global_sent = None
        self.receipt = None

    def snapshot(self):
        return dict(self.cursor)

    def _send(self,payload):
        require(set(payload) <= PAYLOAD_KEYS and all(type(value) in (int,float) and math.isfinite(value)
                for value in payload.values()),'TELEMETRY_PRIVATE_PAYLOAD_REJECTED')
        self.sending = True
        try:
            result = self.callback(dict(payload))
            if result is False:
                self.cursor['logging_rejections'] += 1
                return False
            self.cursor['logged_points'] += 1
            return True
        except Exception:
            # The native source and result save must not fail because an SDK
            # diagnostic/auth/network transport failed. No raw exception text.
            self.cursor['logging_errors'] += 1
            return False
        finally:
            self.sending = False

    def _emit(self,force=False):
        if self.pending is None:
            return
        now = self.clock()
        if force or now-self.last_sent >= self.interval:
            self._send(self.pending)
            self.last_sent = now
            self.last_global_sent = self.pending['fit/global_candidate']
            self.pending = None

    def _complete_line(self,line):
        if line == 'Computing right vector (v)' and self.method != 'FT':
            self._emit(force=True)
            self.cursor['fit_phase_starts'] += 1
            return
        values = parse(line,self.method)
        if values is None:
            return
        self.cursor['global_candidate'] += 1
        self.cursor['candidate'] += 1
        self.cursor['parsed_loss_lines'] += 1
        self.pending = dict(values,batch=self.batch,candidate=self.cursor['candidate'],
                            **{'fit/global_candidate':self.cursor['global_candidate']})
        self._emit()

    def _ingest(self,text):
        for part in text.splitlines(keepends=True):
            ended = part.endswith(('\n','\r'))
            if not self.dropping:
                self.line += part.rstrip('\r\n') if ended else part
                if len(self.line) > self.maximum:
                    self.cursor['oversize_lines'] += 1
                    self.line = ''
                    self.dropping = True
                elif ended:
                    self._complete_line(self.line)
                    self.line = ''
            if ended and self.dropping:
                self.dropping = False

    def write(self,text):
        result = self.original.write(text)  # local bytes/text behaviour first
        if not self.sending:
            try:
                self._ingest(text)
            except Exception:
                self.cursor['parser_errors'] += 1
                self.line = ''
                self.dropping = True
        return result

    def flush(self):
        return self.original.flush()

    def __getattr__(self,name):
        # Keep fileno/encoding/errors/isatty/buffer compatible with stdout.
        return getattr(self.original,name)

    def __enter__(self):
        require(not self.active,'TELEMETRY_ALREADY_ACTIVE')
        self.previous = sys.stdout
        self.original = self.previous if self.original is None else self.original
        require(self.original is not self,'TELEMETRY_RECURSIVE_STDOUT')
        self.active = True
        sys.stdout = self
        try:
            self._send(dict(batch=self.batch,phase_id=10))
        except BaseException:
            sys.stdout = self.previous
            self.active = False
            raise
        return self

    def __exit__(self,error_type,error,traceback):
        try:
            try:
                if self.line and not self.dropping:
                    self._complete_line(self.line)
                self._emit(force=True)
                self._send(dict(batch=self.batch,phase_id=12 if error is not None else 11))
            except Exception:
                self.cursor['parser_errors'] += 1
            try:
                self.original.flush()
            except Exception:
                if error is None:
                    raise
                if hasattr(error,'add_note'):
                    error.add_note('LOCAL_STDOUT_FLUSH_FAILED; original native error preserved')
        finally:
            sys.stdout = self.previous
            self.active = False
            self.line = ''  # drop private partial text even on errors
            self.pending = None
            self.receipt = dict(schema=SCHEMA,cursor=self.snapshot(),
                native_values='FT_UNROUNDED_BATCH_LOSS' if self.method == 'FT' else 'NATIVE_PRINT_ROUNDED_3_DECIMAL_LATENT_LOSS_NLL_KL',
                candidate_semantics='OBSERVED_NATIVE_LOSS_EVALUATIONS_NOT_ADAM_UPDATES',
                SDK_acceptance_is_not_remote_delivery=True,
                raw_prompt_target_stdout_upload=False,new_GPU_syncs=0,new_forward_calls=0,
                scientific_method_mutation=False,automatic_retry=False)
        return False
