from typing import Optional, Tuple, Literal
from dataclasses import dataclass

from importlib import import_module

MODEL_REGISTRY = {
    "transformer": ("fla.models", "TransformerConfig", "TransformerForCausalLM"),
    "gla": ("fla.models", "GLAConfig", "GLAForCausalLM"),
    "hgrn2": ("fla.models", "HGRN2Config", "HGRN2ForCausalLM"),
    "mamba": ("fla.models", "MambaConfig", "MambaForCausalLM"),
    "mamba2": ("fla.models", "Mamba2Config", "Mamba2ForCausalLM"),
    "gdn": ("fla.models", "GatedDeltaNetConfig", "GatedDeltaNetForCausalLM"),
    "dela": ("easysp.models.dela", "DELAConfig", "DELAForCausalLM"),
    "mistral": ("ultra.modeling.mistral", "MistralConfig", "MistralForCausalLM"),
    "neox": ("transformers", None, "AutoModelForCausalLM"),
}


@dataclass
class BaseTransformerArgs:

    name_type: str = "model" # ["backbone, "model"]
    model_name: str = "transformer"
    seed: int = 42
    config_path: str = ""
    dim: int = 512
    n_layers: int = 24
    max_seqlen: int = 2048 # will be set by data.seqlen
    vocab_size: int = 32000
    meta_init: bool = True

# Optional and only used for fully shard options (fsdp) is choose. Highly recommanded for large models
def build_fsdp_grouping_plan(model_args: BaseTransformerArgs):
    group_plan: Tuple[int, bool] = []

    if model_args.name_type == "model":
        # Grouping and output seperately
        group_plan.append(("model.embeddings", True))

        # Grouping by layers
        for i in range(model_args.n_layers):
            group_plan.append((f"model.layers.{i}", True))
    elif model_args.name_type == "backbone":
        # Grouping and output seperately
        group_plan.append(("backbone.embeddings", True))

        # Grouping by layers
        for i in range(model_args.n_layers):
            group_plan.append((f"backbone.layers.{i}", False))

    # NOTE here is a dangerous that the lm_head's forward function is not called
    # So I choose no fsdp for the output linear layer, at most cost about 2GB for 32k vocabulary
    group_plan.append(("lm_head", True))

    return group_plan


def get_num_flop_per_token(
    num_non_embed_params: int, n_layers: int, dim: int, seq_len: int
) -> int:
    return 6 * num_non_embed_params + attention_flops_per_token(
        n_layers, seq_len, dim, True
    )

def attention_flops_per_token(n_layers, seq_len, dim, causal):
    # Formula from https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py#L27-L30
    return 3.5 * (4 * n_layers * seq_len * dim // (2 if causal else 1))

def reinit_weights(model):
    for module in model.modules():
        if hasattr(module, '_is_hf_initialized'):
            module._is_hf_initialized = False
    model.init_weights()

def reset_rope_cache(model) -> None:
    """
    Reset parameters for all modules named 'RotaryEmbedding'.
    
    Args:
        model: PyTorch model to traverse
    """
    from fla.modules.rotary import RotaryEmbedding
    # from ultra.modeling.mistral.modeling_mistral import MistralRotaryEmbedding
    for name, module in model.named_modules():
        if isinstance(module, RotaryEmbedding):
            module.reset_parameters()
        # elif isinstance(module, MistralRotaryEmbedding):
        #     module.reset_parameters()

def load_model_from_config(model_name, config_file):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}")
    
    module_name, config_class_name, model_class_name = MODEL_REGISTRY[model_name]
    module = import_module(module_name)

    if model_name == "neox":
        model_class = getattr(module, model_class_name)
        return model_class.from_pretrained(
            "EleutherAI/pythia-70m-deduped",
            revision="step3000",
            cache_dir="./pythia-70m-deduped/step3000"
        )
    else:
        config_class = getattr(module, config_class_name)
        model_class = getattr(module, model_class_name)
        config = config_class.from_pretrained(config_file)
        return model_class(config)


def load_model_from_path(model_name, path):
    if model_name not in MODEL_REGISTRY or model_name == "neox":
        raise ValueError(f"Unsupported or unknown model: {model_name}")
    
    module_name, _, model_class_name = MODEL_REGISTRY[model_name]
    module = import_module(module_name)
    model_class = getattr(module, model_class_name)
    return model_class.from_pretrained(path)


def load_config_from_path(model_name, path):
    if model_name not in MODEL_REGISTRY or MODEL_REGISTRY[model_name][1] is None:
        raise ValueError(f"Unsupported or unknown model: {model_name}")
    
    module_name, config_class_name, _ = MODEL_REGISTRY[model_name]
    module = import_module(module_name)
    config_class = getattr(module, config_class_name)
    return config_class.from_pretrained(path)


def get_config(model_name):
    if model_name not in MODEL_REGISTRY or MODEL_REGISTRY[model_name][1] is None:
        raise ValueError(f"Unsupported or unknown model: {model_name}")
    
    module_name, config_class_name, _ = MODEL_REGISTRY[model_name]
    module = import_module(module_name)
    config_class = getattr(module, config_class_name)
    return config_class()
