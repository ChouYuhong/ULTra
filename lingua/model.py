import torch

def reinit_weights(model):
    for module in model.modules():
        if hasattr(module, '_is_hf_initialized'):
            module._is_hf_initialized = False
    model.init_weights()