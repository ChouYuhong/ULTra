from typing import Optional, Union, Tuple
from dataclasses import dataclass


@dataclass
class BaseTransformerArgs:
    dim: int = 512
    n_layers: int = 8
    head_dim: Optional[int] = None
    n_heads: Optional[int] = None
    n_kv_heads: Optional[int] = None

    ffn_dim_multiplier: Optional[float] = None

    multiple_of: int = 256

    norm_eps: float = 1e-5

    rope_theta: float = 10000.0

    init_base_std: Optional[float] = None
    init_std_factor: str = "disabled"

    max_seqlen: int = 1024

@dataclass
class LMTransformerArgs(BaseTransformerArgs):

    seed: int = 42

    vocab_size: int = -1
    weight_tying: bool = False

    sliding_window: Optional[int] = None

# Optional and only used for fully shard options (fsdp) is choose. Highly recommanded for large models
def build_fsdp_grouping_plan(model_args: LMTransformerArgs):
    group_plan: Tuple[int, bool] = []

    # Grouping and output seperately
    group_plan.append(("tok_embeddings", False))

    # Grouping by layers
    for i in range(model_args.n_layers):
        group_plan.append((f"layers.{i}", False))

    group_plan.append(("output", True))

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