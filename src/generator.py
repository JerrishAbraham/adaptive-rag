"""
Stage 4 — Response Generation.

Thin, swappable generator abstraction. The rest of the pipeline calls
generator.generate(query=..., context=...) and gets back a structured
GenerationResult — no Groq-specific types or calls leak outside this
module. Provider/model/temperature/etc. come entirely from config, so
swapping providers later means adding a new branch here, not touching
any other module.

The generator NEVER retrieves, re-ranks, or decides K. It receives
already-optimized context and only produces an answer from it.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

from src.prompt_builder import build_prompt

logger = logging.getLogger(__name__)

load_dotenv()  # populates os.environ from a local .env if present


@dataclass
class GenerationResult:
    answer: str
    model: str
    provider: str
    latency_seconds: float
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    success: bool
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class Generator:
    """Provider-agnostic generator interface, currently backed by Groq."""

    def __init__(self, config: dict):
        """
        Args:
            config: the 'generator' section of config.yaml
                (provider, model, temperature, max_tokens,
                 reasoning_effort, timeout).
        """
        self.config = config
        self.provider = config["provider"]

        if self.provider != "groq":
            raise ValueError(
                f"Unsupported provider '{self.provider}'. Only 'groq' is "
                f"implemented in this stage."
            )

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Set it in your shell or in a "
                "local .env file (never hardcode it)."
            )

        from groq import Groq
        self._client = Groq(api_key=api_key)

    def generate(self, query: str, context: list[dict]) -> GenerationResult:
        """
        Generate an answer for `query` grounded in `context`.

        Args:
            query: the user's question (str).
            context: list of RetrievalResult-shaped dicts (post diversity
                filtering), each with at least 'title' and 'text'.

        Returns:
            GenerationResult with the answer and generation metadata.
            On failure, success=False and error is populated; answer is
            an empty string rather than raising, so a single failed
            generation doesn't crash a batch evaluation run later.
        """
        prompt = build_prompt(query, context)
        model = self.config["model"]

        start = time.time()
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]},
                ],
                temperature=self.config.get("temperature", 0.2),
                max_tokens=self.config.get("max_tokens", 512),
                reasoning_effort=self.config.get("reasoning_effort", "low"),
                timeout=self.config.get("timeout", 60),
            )
            latency = time.time() - start

            answer = response.choices[0].message.content.strip()
            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
            output_tokens = getattr(usage, "completion_tokens", None) if usage else None

            result = GenerationResult(
                answer=answer,
                model=model,
                provider=self.provider,
                latency_seconds=round(latency, 3),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                success=True,
            )
            logger.info(
                "Generation OK: model=%s latency=%.2fs in_tok=%s out_tok=%s",
                model, latency, input_tokens, output_tokens,
            )
            return result

        except Exception as exc:  # noqa: BLE001 - deliberately broad: any
            # provider/network failure must degrade to a structured error,
            # not crash the caller.
            latency = time.time() - start
            logger.error("Generation FAILED: model=%s error=%s", model, exc)
            return GenerationResult(
                answer="",
                model=model,
                provider=self.provider,
                latency_seconds=round(latency, 3),
                input_tokens=None,
                output_tokens=None,
                success=False,
                error=str(exc),
            )


def load_generator_config(config_path: str = "configs/config.yaml") -> dict:
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["generator"]


def build_generator(config_path: str = "configs/config.yaml") -> Generator:
    """Convenience factory: load config and construct a Generator."""
    config = load_generator_config(config_path)
    return Generator(config)