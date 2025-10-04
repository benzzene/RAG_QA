from typing import Mapping, Any
from transformers.utils import logging
from .qwen import QwenModel, QwenInitConfig, QwenGenConfig
logging.set_verbosity_error()
def make_model(cfg: Mapping[str, Any]) -> QwenModel:
    """
    Tworzy instancję QwenModel na podstawie słownika konfiguracyjnego.
    """
    # 1) inicjalizacja
    init_cfg = QwenInitConfig(
        model_id=cfg.get("model_id", QwenInitConfig.model_id),
        device=cfg.get("device", "cuda"),
        use_4bit=cfg.get("use_4bit", True),
        bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
        max_context_chars=cfg.get("max_context_chars", QwenInitConfig.max_context_chars),
        system_prompt=cfg.get("system_prompt", QwenInitConfig.system_prompt)
    )

    # 2) parametry generacji
    gen_cfg = QwenGenConfig(
        max_new_tokens=cfg.get("max_new_tokens", QwenGenConfig.max_new_tokens),
        temperature=cfg.get("temperature", QwenGenConfig.temperature),
        top_p=cfg.get("top_p", QwenGenConfig.top_p),
        do_sample=cfg.get("do_sample", QwenGenConfig.do_sample),
        repetition_penalty=cfg.get("repetition_penalty", QwenGenConfig.repetition_penalty),
    )

    return QwenModel(init_cfg=init_cfg, gen_cfg=gen_cfg)
