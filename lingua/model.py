import torch

def reinit_weights(model):
    for module in model.modules():
        if hasattr(module, '_is_hf_initialized'):
            module._is_hf_initialized = False
    model.init_weights()

def load_model_from_config(model_name, config_file):
    if model_name == "transformer":
        from fla.models import TransformerConfig, TransformerForCausalLM
        model = TransformerForCausalLM(TransformerConfig.from_pretrained(config_file))
    elif model_name == "gla":
        from fla.models import GLAForCausalLM, GLAConfig
        model = GLAForCausalLM(GLAConfig.from_pretrained(config_file))
    elif model_name == "hgrn2":
        from fla.models import HGRN2ForCausalLM, HGRN2Config
        model = HGRN2ForCausalLM(HGRN2Config.from_pretrained(config_file))
    elif model_name == "mamba":
        from fla.models import MambaForCausalLM, MambaConfig
        model = MambaForCausalLM(MambaConfig.from_pretrained(config_file))
    elif model_name == "mamba2":
        from fla.models import Mamba2ForCausalLM, Mamba2Config
        model = Mamba2ForCausalLM(Mamba2Config.from_pretrained(config_file))
    elif model_name == "gdn":
        from fla.models import GatedDeltaNetForCausalLM, GatedDeltaNetConfig
        model = GatedDeltaNetForCausalLM(GatedDeltaNetConfig.from_pretrained(config_file))
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model