"""Physical-only batching around the unchanged native representation reader."""
import torch

def bounded_reader(original, ledger, size=2):
    def read(model, tok, contexts, idxs, layer, module_template, track='in'):
        assert len(contexts)==len(idxs) and contexts
        outputs=[]
        for start in range(0,len(contexts),size):
            outputs.append(original(model,tok,contexts[start:start+size],idxs[start:start+size],
                                    layer,module_template,track))
            ledger.add('native_representation_physical_batches')
        if isinstance(outputs[0],tuple):
            return tuple(torch.cat([out[j] for out in outputs]) for j in range(len(outputs[0])))
        return torch.cat(outputs)
    return read
