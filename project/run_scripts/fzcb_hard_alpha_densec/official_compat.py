"""Runtime-only compatibility adapters for the sealed Official source.

The pinned EasyEdit revision registers a ``with_kwargs=True`` forward hook
with the legacy positional signature.  Current PyTorch calls such hooks as
``(module, args, kwargs, output)``.  This module repairs only that calling
convention without editing the sealed checkout or changing traced tensors.
"""

from __future__ import annotations

from typing import Any


def install_official_trace_kwargs_adapter() -> dict[str, Any]:
    from easyeditor.util import nethook

    if getattr(nethook.Trace, "_odeedit_kwargs_adapter", False):
        return {"installed": True, "install_count": 0, "with_kwargs_signature": "module,args,kwargs,output"}

    original_trace = nethook.Trace

    class Trace(original_trace):
        _odeedit_kwargs_adapter = True

        def __init__(
            self,
            module,
            layer=None,
            retain_output=True,
            retain_input=False,
            clone=False,
            detach=False,
            retain_grad=False,
            edit_output=None,
            stop=False,
        ):
            retainer = self
            self.layer = layer
            if layer is not None:
                module = nethook.get_module(module, layer)

            def retain_hook(_module, inputs, kwargs, output):
                if retain_input:
                    if inputs:
                        retainer.input = nethook.recursive_copy(
                            inputs[0] if len(inputs) == 1 else inputs,
                            clone=clone,
                            detach=detach,
                            retain_grad=False,
                        )
                    elif "hidden_states" in kwargs:
                        retainer.input = nethook.recursive_copy(
                            kwargs["hidden_states"], clone=clone, detach=detach, retain_grad=False
                        )
                    else:
                        retainer.input = None
                if edit_output:
                    output = nethook.invoke_with_optional_args(edit_output, output=output, layer=self.layer)
                if retain_output:
                    retainer.output = nethook.recursive_copy(
                        output, clone=clone, detach=detach, retain_grad=retain_grad
                    )
                    if retain_grad:
                        output = nethook.recursive_copy(retainer.output, clone=True, detach=False)
                if stop:
                    raise nethook.StopForward()
                return output

            self.registered_hook = module.register_forward_hook(retain_hook, with_kwargs=True)
            self.stop = stop

    nethook.Trace = Trace
    return {"installed": True, "install_count": 1, "with_kwargs_signature": "module,args,kwargs,output"}
