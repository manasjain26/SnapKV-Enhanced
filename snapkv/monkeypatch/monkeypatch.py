from importlib.metadata import version as get_version
from packaging.version import Version
import warnings
import transformers

def check_version():
    try:
        transformers_version = get_version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
        transformers_version = "0.0.0"
    return transformers_version

def _is_modern_transformers():
    """Check if transformers >= 4.43 (unified attention classes)."""
    v = check_version()
    try:
        return Version(v) >= Version("4.43.0")
    except Exception:
        # Try simple string check
        return not any(old in v for old in ['4.37', '4.38', '4.39', '4.40', '4.41', '4.42'])

def replace_llama():
    transformers_version = check_version()

    if _is_modern_transformers():
        # Modern transformers (>= 4.43): LlamaAttention is unified
        from snapkv.monkeypatch.llama_hijack_modern import (
            llama_attention_forward_modern,
            prepare_inputs_for_generation_llama_modern,
        )
        print(f"[SnapKV] Using modern hijack for transformers {transformers_version}")
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attention_forward_modern
        transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama_modern
    else:
        # Legacy transformers (4.37): LlamaFlashAttention2 exists
        from snapkv.monkeypatch.llama_hijack_4_37 import (
            llama_flash_attn2_forward as llama_flash_attn2_forward_4_37,
            prepare_inputs_for_generation_llama as prepare_inputs_for_generation_llama_4_37,
        )
        version_list = ['4.37']
        if not any(v in transformers_version for v in version_list):
            warnings.warn(
                f"Transformers version {transformers_version} might not be compatible with SnapKV. "
                f"SnapKV is tested with Transformers version {version_list}."
            )
        transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama_4_37
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_4_37

def replace_mistral():
    transformers_version = check_version()

    if _is_modern_transformers():
        # For modern transformers, Mistral also uses unified attention
        # For now, print a warning — the Llama hijack pattern can be adapted
        print(f"[SnapKV] Mistral support for transformers {transformers_version}: "
              f"Use replace_llama() — modern Mistral uses similar unified attention.")
        warnings.warn(
            "Mistral monkeypatch for modern transformers not yet implemented. "
            "If your model is Mistral-based, the Llama hijack may work if the model "
            "uses LlamaAttention internally (many Mistral models do in modern transformers)."
        )
    else:
        from snapkv.monkeypatch.mistral_hijack_4_37 import (
            mistral_flash_attn2_forward as mistral_flash_attn2_forward_4_37,
            prepare_inputs_for_generation_mistral as prepare_inputs_for_generation_mistral_4_37,
        )
        version_list = ['4.37']
        if not any(v in transformers_version for v in version_list):
            warnings.warn(
                f"Transformers version {transformers_version} might not be compatible with SnapKV. "
                f"SnapKV is tested with Transformers version {version_list}."
            )
        transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_mistral_4_37
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_4_37

def replace_mixtral():
    transformers_version = check_version()

    if _is_modern_transformers():
        print(f"[SnapKV] Mixtral support for transformers {transformers_version}: "
              f"Modern Mixtral uses similar unified attention — use replace_llama() if applicable.")
        warnings.warn(
            "Mixtral monkeypatch for modern transformers not yet implemented."
        )
    else:
        from snapkv.monkeypatch.mixtral_hijack_4_37 import (
            mixtral_flash_attn2_forward as mixtral_flash_attn2_forward_4_37,
            prepare_inputs_for_generation_mixtral as prepare_inputs_for_generation_mixtral_4_37,
        )
        version_list = ['4.37']
        if not any(v in transformers_version for v in version_list):
            warnings.warn(
                f"Transformers version {transformers_version} might not be compatible with SnapKV. "
                f"SnapKV is tested with Transformers version {version_list}."
            )
        transformers.models.mixtral.modeling_mixtral.MixtralForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_mixtral_4_37
        transformers.models.mixtral.modeling_mixtral.MixtralFlashAttention2.forward = mixtral_flash_attn2_forward_4_37