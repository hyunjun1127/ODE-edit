"""Unchanged legacy current math, with external weight-upload accounting."""
import time
import torch
from project.run_scripts.single_layer_edit_preserving_correction.alltoken import FullWeightLlamaOracle


class MeasuredCurrentOracle(FullWeightLlamaOracle):
    def __init__(self,*args,**kwargs):
        self.weight_work=dict(calls=0,finite_scanned_bytes=0,H2D_calls=0,H2D_bytes=0,
                              wall_seconds=0.,cuda_validation_upload_ms=0.)
        self._weight_events=[]
        super().__init__(*args,**kwargs)

    def _weight(self,weight):
        started=time.perf_counter()
        events=None
        if self.device.type=='cuda':
            events=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
            events[0].record()
        value=super()._weight(weight)
        if events:
            events[1].record();self._weight_events.append(events)
        self.weight_work['calls']+=1
        self.weight_work['finite_scanned_bytes']+=weight.numel()*weight.element_size()
        if weight.device!=self.device:
            self.weight_work['H2D_calls']+=1
            self.weight_work['H2D_bytes']+=weight.numel()*weight.element_size()
        self.weight_work['wall_seconds']+=time.perf_counter()-started
        return value

    def collect_weight_timing(self):
        # One boundary sync, not an added synchronization per document.
        if self._weight_events:
            self._sync()
            self.weight_work['cuda_validation_upload_ms']+=sum(a.elapsed_time(b) for a,b in self._weight_events)
            self._weight_events=[]
        return dict(self.weight_work)
