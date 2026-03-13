from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import mlflow.pyfunc
import pandas as pd
from transformers import pipeline

from .rag_retriever import TfidfRagRetriever

try:
    import torch
except Exception:  # pragma: no cover - optional at static analysis time
    torch = None


@dataclass
class GenerationConfig:
    max_new_tokens: int = 180
    temperature: float = 0.0


@dataclass
class RetrievalConfig:
    enabled: bool = True
    top_k: int = 3
    min_score: float = 0.02
    max_chunk_chars: int = 380


class PromptSpecializedHFModel(mlflow.pyfunc.PythonModel):
    """MLflow model that applies DSPy-optimized prompts with optional local RAG."""

    def __init__(
        self,
        task: str = "text2text-generation",
        generation: GenerationConfig | None = None,
        retrieval: RetrievalConfig | None = None,
    ) -> None:
        self.task = task
        self.generation = generation or GenerationConfig()
        self.retrieval = retrieval or RetrievalConfig()
        self._generator = None
        self._prompt = None
        self._retriever: TfidfRagRetriever | None = None

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        artifacts = context.artifacts
        prompt_file = artifacts["optimized_prompt"]
        model_dir = artifacts["hf_model_dir"]

        with open(prompt_file, "r", encoding="utf-8") as f:
            self._prompt = json.load(f)

        rag_chunks_file = artifacts.get("rag_chunks")
        if rag_chunks_file and self.retrieval.enabled:
            self._retriever = TfidfRagRetriever.from_jsonl(rag_chunks_file)

        device = -1
        if torch is not None and torch.cuda.is_available():
            device = 0

        self._generator = pipeline(
            self.task,
            model=model_dir,
            tokenizer=model_dir,
            device=device,
        )

    def _build_prompt(
        self,
        question: str,
        context: str | None = None,
        retrieved_chunks: list[dict[str, str | float]] | None = None,
    ) -> str:
        if self._prompt is None:
            raise RuntimeError("Prompt artifact was not loaded")

        instruction = self._prompt.get("instructions", "Answer accurately in English.")
        default_context = self._prompt.get("default_context", "")
        demos = self._prompt.get("few_shot_demos", [])

        lines = [instruction]
        if default_context:
            lines.append(f"Domain context: {default_context}")
        if context:
            lines.append(f"Request context: {context}")

        if demos:
            lines.append("Examples:")
            for idx, demo in enumerate(demos[:4], start=1):
                lines.append(f"{idx}. Q: {demo.get('question', '').strip()}")
                lines.append(f"   A: {demo.get('answer', '').strip()}")

        if retrieved_chunks:
            lines.append("Knowledge snippets:")
            for idx, chunk in enumerate(retrieved_chunks, start=1):
                source = str(chunk.get("source", "kb"))
                text = str(chunk.get("text", "")).strip()
                if not text:
                    continue
                if len(text) > self.retrieval.max_chunk_chars:
                    text = text[: self.retrieval.max_chunk_chars].rstrip() + "..."
                lines.append(f"{idx}. [{source}] {text}")

        lines.append(f"User question: {question}")
        lines.append("Answer:")
        return "\n".join(lines)

    def _generate(self, prompt: str) -> str:
        if self._generator is None:
            raise RuntimeError("Generator was not initialized")

        result = self._generator(
            prompt,
            max_new_tokens=self.generation.max_new_tokens,
            temperature=self.generation.temperature,
            do_sample=self.generation.temperature > 0,
        )

        if isinstance(result, list) and result:
            candidate = result[0]
            if isinstance(candidate, dict):
                for key in ("generated_text", "summary_text", "text"):
                    if key in candidate:
                        return str(candidate[key]).strip()
            return str(candidate).strip()

        return str(result).strip()

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: Any) -> pd.DataFrame:
        del context

        if isinstance(model_input, pd.DataFrame):
            df = model_input.copy()
        else:
            df = pd.DataFrame(model_input)

        if "question" not in df.columns:
            raise ValueError("Input must include a 'question' column")

        if "context" not in df.columns:
            df["context"] = ""

        answers: list[str] = []
        for _, row in df.iterrows():
            question = str(row["question"])
            request_context = str(row.get("context", ""))

            retrieved_chunks: list[dict[str, str | float]] = []
            if self._retriever is not None and self.retrieval.enabled:
                retrieval_query = question if not request_context else f"{question}\n{request_context}"
                retrieved_chunks = self._retriever.retrieve(
                    query=retrieval_query,
                    top_k=self.retrieval.top_k,
                    min_score=self.retrieval.min_score,
                )

            prompt = self._build_prompt(
                question=question,
                context=request_context,
                retrieved_chunks=retrieved_chunks,
            )
            answers.append(self._generate(prompt))

        return pd.DataFrame({"answer": answers})
