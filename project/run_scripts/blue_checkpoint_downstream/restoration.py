"""Overwrite selected tensors from verified W0, not checkpoint chaining."""
import importlib.util
from pathlib import Path


def load_reference(path):
    spec = importlib.util.spec_from_file_location('sealed_checkpoint_loader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class W0Transaction:
    def __init__(self, model, reference, entries):
        import torch
        self.model = model
        self.reference = reference
        self.params = dict(model.named_parameters())
        assert self.params and all(p.dtype == torch.float32 for p in self.params.values())
        self.keys = sorted({k for entry in entries for k in entry['weights']})
        self.w0 = {k:self.params[k].detach().cpu().clone() for k in self.keys}
        self.pointer = self.pointer_identity()
        self.w0_hashes = self.hashes()
        for entry in entries:
            for key, member in entry['base_selected_weights'].items():
                assert self.w0_hashes[key] == member['sha256'], 'W0_BYTES_MISMATCH'
        self.expected = dict(self.w0_hashes)
        self.versions = self.version_identity()

    def pointer_identity(self):
        return {k:(id(p),p.data_ptr(),str(p.dtype),list(p.shape)) for k,p in self.params.items()}

    def version_identity(self):
        return {k:p._version for k,p in self.params.items()}

    def hashes(self):
        # Full parameter bytes, streamed tensor by tensor through the pinned
        # digest helper. No second full-model checkpoint is materialized.
        return {k:self.reference.tensor_sha(p) for k,p in self.params.items()}

    def verify_endpoint(self):
        assert self.pointer_identity() == self.pointer
        assert self.version_identity() == self.versions, 'EVALUATION_PARAMETER_MUTATION'
        current = self.hashes()
        assert current == self.expected, 'EVALUATION_WEIGHT_BYTES_MUTATION'
        return current

    def restore_w0(self):
        import torch
        # Check current endpoint before overwriting so corruption is not hidden.
        self.verify_endpoint()
        with torch.no_grad():
            for key,value in self.w0.items():
                self.params[key].copy_(value.to(self.params[key].device))
        self.expected = dict(self.w0_hashes)
        self.versions = self.version_identity()
        return self.verify_endpoint()

    def apply(self, checkpoint, entry):
        assert self.expected == self.w0_hashes, 'CHECKPOINT_MUST_START_FROM_W0'
        self.verify_endpoint()
        before = self.version_identity()
        assert set(checkpoint['weights']) == set(entry['weights'])
        self.reference.apply_selected_to_verified_w0(self.model, checkpoint, entry)
        self.expected = dict(self.w0_hashes)
        self.expected.update({k:v['sha256'] for k,v in entry['weights'].items()})
        for key,p in self.params.items():
            if key not in entry['weights']:
                assert p._version == before[key]
        self.versions = self.version_identity()
        return self.verify_endpoint()
