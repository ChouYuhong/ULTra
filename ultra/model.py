from typing import Optional, Tuple, Literal
from dataclasses import dataclass

@dataclass
class BaseTransformerArgs:

    name_type: str = "model" # ["backbone, "model"]
    model_name: str = "transformer"
    seed: int = 42
    config_path: str = ""
    dim: int = 512
    n_layers: int = 8
    max_seqlen: int = 2048 # will be set by data.seqlen
    vocab_size: int = 32000

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
            group_plan.append((f"backbone.layers.{i}", True))

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
    for name, module in model.named_modules():
        if isinstance(module, RotaryEmbedding):
            module.reset_parameters()

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
    elif model_name == "neox":
        from transformers import AutoModelForCausalLM
        from transformers.models.gpt_neox import GPTNeoXForCausalLM
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m-deduped",
            revision="step3000",
            cache_dir="./pythia-70m-deduped/step3000",
            )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model

def load_model_from_path(model_name, path):
    if model_name == "transformer":
        from fla.models import TransformerForCausalLM
        model = TransformerForCausalLM.from_pretrained(path)
    elif model_name == "gla":
        from fla.models import GLAForCausalLM
        model = GLAForCausalLM.from_pretrained(path)
    elif model_name == "hgrn2":
        from fla.models import HGRN2ForCausalLM
        model = HGRN2ForCausalLM.from_pretrained(path)
    elif model_name == "mamba":
        from fla.models import MambaForCausalLM
        model = MambaForCausalLM.from_pretrained(path)
    elif model_name == "mamba2":
        from fla.models import Mamba2ForCausalLM
        model = Mamba2ForCausalLM.from_pretrained(path)
    elif model_name == "gdn":
        from fla.models import GatedDeltaNetForCausalLM
        model = GatedDeltaNetForCausalLM.from_pretrained(path)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model


def load_config_from_path(model_name, path):
    if model_name == "transformer":
        from fla.models import TransformerConfig
        config = TransformerConfig.from_pretrained(path)
    elif model_name == "gla":
        from fla.models import GLAConfig
        config = GLAConfig.from_pretrained(path)
    elif model_name == "hgrn2":
        from fla.models import HGRN2Config
        config = HGRN2Config.from_pretrained(path)
    elif model_name == "mamba":
        from fla.models import MambaConfig
        config = MambaConfig.from_pretrained(path)
    elif model_name == "mamba2":
        from fla.models import Mamba2Config
        config = Mamba2Config.from_pretrained(path)
    elif model_name == "gdn":
        from fla.models import GatedDeltaNetConfig
        config = GatedDeltaNetConfig.from_pretrained(path)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return config

def get_config(model_name):
    if model_name == "transformer":
        from fla.models import TransformerConfig
        config = TransformerConfig()
    elif model_name == "gla":
        from fla.models import GLAConfig
        config = GLAConfig()
    elif model_name == "hgrn2":
        from fla.models import HGRN2Config
        config = HGRN2Config()
    elif model_name == "mamba":
        from fla.models import MambaConfig
        config = MambaConfig()
    elif model_name == "mamba2":
        from fla.models import Mamba2Config
        config = Mamba2Config()
    elif model_name == "gdn":
        from fla.models import GatedDeltaNetConfig
        config = GatedDeltaNetConfig()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return config