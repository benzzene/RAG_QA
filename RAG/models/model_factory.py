from typing import Mapping, Any
from transformers.utils import logging
from .qwen import QwenModel, QwenInitConfig, QwenGenConfig
logging.set_verbosity_error()
def make_model(cfg: Mapping[str, Any]) -> QwenModel:
    """Creates a QwenModel instance from a configuration dictionary.

    This function builds both the initialization and generation
    configurations for the Qwen model, using values from the provided
    dictionary. Any missing keys will be filled with defaults from
    `QwenInitConfig` and `QwenGenConfig`.

    Args:
        cfg (Mapping[str, Any]): Configuration dictionary with optional keys:
            - "model_id" (str): Hugging Face model identifier.
            - "device" (str): Device to run the model on ("cuda" or "cpu").
            - "use_4bit" (bool): Whether to enable 4-bit quantization.
            - "bnb_4bit_quant_type" (str): Quantization type (e.g., "nf4").
            - "bnb_4bit_use_double_quant" (bool): Enable double quantization.
            - "max_context_chars" (int): Maximum characters in the context block.
            - "system_prompt" (str): System prompt for the assistant.
            - "max_new_tokens" (int): Maximum number of tokens to generate.
            - "temperature" (float): Sampling temperature.
            - "top_p" (float): Nucleus sampling probability threshold.
            - "do_sample" (bool): Whether to sample (vs. greedy decoding).
            - "repetition_penalty" (float): Penalty for repeated tokens.

    Returns:
        QwenModel: A fully initialized Qwen model with tokenizer and configs.
    """
    # 1) initialization
    init_cfg = QwenInitConfig(
        model_id=cfg.get("model_id", QwenInitConfig.model_id),
        device=cfg.get("device", "cuda"),
        use_4bit=cfg.get("use_4bit", True),
        bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
        max_context_chars=cfg.get("max_context_chars", QwenInitConfig.max_context_chars),
        system_prompt=cfg.get("system_prompt", QwenInitConfig.system_prompt)
    )

    # 2) generation parameters
    gen_cfg = QwenGenConfig(
        max_new_tokens=cfg.get("max_new_tokens", QwenGenConfig.max_new_tokens),
        temperature=cfg.get("temperature", QwenGenConfig.temperature),
        top_p=cfg.get("top_p", QwenGenConfig.top_p),
        do_sample=cfg.get("do_sample", QwenGenConfig.do_sample),
        repetition_penalty=cfg.get("repetition_penalty", QwenGenConfig.repetition_penalty),
    )

    return QwenModel(init_cfg=init_cfg, gen_cfg=gen_cfg)
