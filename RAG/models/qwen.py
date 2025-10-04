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
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.1


class QwenModel:
    """
    Pojedyncza implementacja modelu Qwen:
    - inicjalizacja (tokenizer + model, ewentualnie 4-bit),
    - formatowanie kontekstu,
    - budowa wiadomości chatowych,
    - generacja odpowiedzi z przekazanego kontekstu.
    """

    def __init__(self, init_cfg: QwenInitConfig, gen_cfg: Optional[QwenGenConfig] = None):
        self.init_cfg = init_cfg
        self.gen_cfg = gen_cfg or QwenGenConfig()

        if self.init_cfg.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.init_cfg.device

        # BitsAndBytes (opcjonalny 4-bit)
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

        # Przygotuj bazowy GenerationConfig (zależny od tokenizer’a)
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
        """
        ctxs: lista słowników z polami: 'text', 'meta' (np. {'source'/'source_path', 'chunk_id'}).
        Składa zwarty blok kontekstu z nagłówkami źródeł.
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
        """
        Konstruuje wiadomości w formacie chat dla Qwen 2.5 (system + user).
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
        """
        Stosuje wpudowany chat template i zwraca tensory wyjściowe na wcześniej podanym device
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
        """
        Generuje odpowiedź z gotowych messages (system+user).
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
        """
        Generuje odpowiedz na bazie przekazanego kontekstu.
        Zwraca: answer + użyty kontekst + echo ustawień.
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
