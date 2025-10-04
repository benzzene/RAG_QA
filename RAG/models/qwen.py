from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)


@dataclass
class QwenInitConfig:
    """Initialization configuration for the Qwen model.

    Attributes:
        model_id (str): Model identifier from Hugging Face Hub.
        device (Optional[str]): Device to run the model on ("cuda", "cpu", or None for auto-detect).
        use_4bit (bool): Whether to enable 4-bit quantization.
        bnb_4bit_quant_type (str): Quantization type (e.g., "nf4").
        bnb_4bit_use_double_quant (bool): Whether to use double quantization.
        bnb_compute_dtype (torch.dtype): Compute dtype for quantization.
        torch_dtype (torch.dtype): Torch dtype for model weights.
        device_map (str | dict): Device map for model placement.
        system_prompt (str): Default system prompt for the assistant.
        max_context_chars (int): Maximum number of characters allowed in the context block.
    """
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    device: Optional[str] = None  # "cuda" | "cpu" | None
    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_compute_dtype: torch.dtype = torch.float16
    torch_dtype: torch.dtype = torch.float16
    device_map: str | dict = "auto"

    system_prompt: str = (
        "Jesteś asystentem QA. Odpowiadasz WYŁĄCZNIE na podstawie przekazanego kontekstu. "
        "Twoim zadaniem jest wydobyć informację z kontekstu, nawet jeżeli są rozproszone. "
        "Cytuj źródła (nazwy plików/ścieżki) w nawiasach, gdy to możliwe."
    )

    # Domyślny user_prompt – budowany w build_messages()
    max_context_chars: int = 12000


@dataclass
class QwenGenConfig:
    """Generation configuration for the Qwen model.

    Attributes:
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling probability threshold.
        do_sample (bool): Whether to sample from the distribution (vs. greedy decoding).
        repetition_penalty (float): Penalty for repeated tokens.
    """
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.1


class QwenModel:
    """Wrapper around the Qwen 2.5 model for QA with context-based answering.

    This class handles initialization, context formatting, message building,
    and response generation with optional quantization and generation settings.
    """

    def __init__(self, init_cfg: QwenInitConfig, gen_cfg: Optional[QwenGenConfig] = None):
        """
        Args:
            init_cfg (QwenInitConfig): Initialization configuration.
            gen_cfg (Optional[QwenGenConfig]): Generation configuration. If None, a default is created.
        """
        self.init_cfg = init_cfg
        self.gen_cfg = gen_cfg or QwenGenConfig()

        if self.init_cfg.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.init_cfg.device

        # BitsAndBytes
        self._bnb_config = None
        if self.init_cfg.use_4bit:
            self._bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.init_cfg.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.init_cfg.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=self.init_cfg.bnb_compute_dtype,
            )

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.init_cfg.model_id, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.init_cfg.model_id,
            quantization_config=self._bnb_config,
            torch_dtype=self.init_cfg.torch_dtype,
            device_map=self.init_cfg.device_map,
        )
        self.model.eval()

        # Base generation config (linked to tokenizer)
        self._base_gen_cfg = GenerationConfig(
            max_new_tokens=self.gen_cfg.max_new_tokens,
            temperature=self.gen_cfg.temperature,
            top_p=self.gen_cfg.top_p,
            do_sample=self.gen_cfg.do_sample,
            repetition_penalty=self.gen_cfg.repetition_penalty,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )


    def format_context(self, ctxs: List[Dict[str, Any]], max_chars: Optional[int] = None) -> str:
        """Formats retrieval contexts into a compact text block with source headers.

        Args:
            ctxs (List[Dict[str, Any]]): List of dictionaries containing
                'text' and 'meta' fields (metadata should include 'source' or 'source_path',
                and optionally 'chunk_id').
            max_chars (Optional[int]): Maximum number of characters allowed.
                Defaults to `QwenInitConfig.max_context_chars`.

        Returns:
            str: Formatted context block.
        """
        limit = max_chars if max_chars is not None else self.init_cfg.max_context_chars
        blocks: List[str] = []
        total = 0

        for c in ctxs:
            meta = c.get("meta", {}) or {}
            source = meta.get("source") or meta.get("source_path") or "unknown_source"
            chunk_id = meta.get("chunk_id")
            header = f"### Source: {source} | chunk_id: {chunk_id}\n"
            body = (c.get("text") or "").strip()
            piece = header + body
            if total + len(piece) > limit:
                break
            blocks.append(piece)
            total += len(piece)

        return "\n\n".join(blocks)

    def build_messages(self, query: str, ctxs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Builds system and user messages for Qwen in chat format.

        Args:
            query (str): User query.
            ctxs (List[Dict[str, Any]]): List of context dictionaries with text and metadata.

        Returns:
            List[Dict[str, str]]: Chat messages in the format expected by Qwen.
        """
        context_block = self.format_context(ctxs)
        user_prompt = (
            f"CONTEXT:\n{context_block}\n\n"
            f"Zadanie: Odpowiedz na pytanie wyłącznie na podstawie powyższego kontekstu.\n"
            f"QUESTION: {query}\n"
            f"Odpowiedź po polsku. Jeśli brak danych w kontekście, powiedz to wprost."
        )
        return [
            {"role": "system", "content": self.init_cfg.system_prompt},
            {"role": "user", "content": user_prompt},
        ]


    def _encode_messages(self, messages: List[Dict[str, str]]):
        """Encodes chat messages into model input tensors.

        Args:
            messages (List[Dict[str, str]]): Chat messages (system and user).

        Returns:
            torch.Tensor: Tokenized input IDs ready for the model.
        """
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)
        return input_ids

    @torch.inference_mode()
    def generate_from_messages(
        self,
        messages: List[Dict[str, str]],
        gen_cfg: Optional[GenerationConfig] = None,
    ) -> str:
        """Generates a model response given pre-built chat messages.

        Args:
            messages (List[Dict[str, str]]): Chat messages (system and user).
            gen_cfg (Optional[GenerationConfig]): Custom generation config.
                Defaults to the base generation config.

        Returns:
            str: Generated answer string.
        """
        input_ids = self._encode_messages(messages)
        cfg = gen_cfg or self._base_gen_cfg

        outputs = self.model.generate(
            input_ids=input_ids,
            **cfg.to_dict(),
        )
        gen_ids = outputs[0, input_ids.shape[-1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    def answer_from_contexts(
        self,
        query: str,
        ctxs: List[Dict[str, Any]],
        gen_cfg: Optional[GenerationConfig] = None,
    ) -> Dict[str, Any]:
        """Answers a query using the provided contexts.

        This method builds chat messages, generates an answer, and returns
        both the answer and metadata.

        Args:
            query (str): User query string.
            ctxs (List[Dict[str, Any]]): List of context dictionaries with text and metadata.
            gen_cfg (Optional[GenerationConfig]): Custom generation config. 
                Defaults to the base generation config.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - "answer" (str): The generated answer.
                - "contexts" (List[Dict[str, Any]]): The input contexts.
                - "query" (str): The original query.
                - "gen_config" (dict): The generation configuration used.
                - "model_id" (str): Model identifier.
        """
        messages = self.build_messages(query, ctxs)
        answer = self.generate_from_messages(messages, gen_cfg=gen_cfg)

        return {
            "answer": answer,
            "contexts": ctxs,
            "query": query,
            "gen_config": (gen_cfg or self._base_gen_cfg).to_dict(),
            "model_id": self.init_cfg.model_id,
        }
