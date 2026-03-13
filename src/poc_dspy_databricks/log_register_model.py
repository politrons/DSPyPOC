from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

try:
    import torch
except Exception:  # pragma: no cover - optional at static analysis time
    torch = None

from .pyfunc_model import GenerationConfig, PromptSpecializedHFModel, RetrievalConfig
from .rag_retriever import TfidfRagRetriever

DEFAULT_BASELINE_INSTRUCTION = (
    "You are a specialized support assistant. Respond in English with concise actionable guidance."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log and register a DSPy-specialized Hugging Face model")
    parser.add_argument("--optimized-prompt", default="artifacts/dspy_optimized/optimized_prompt.json")
    parser.add_argument("--test-dataset", default="data/saas_support_test_qa.jsonl")
    parser.add_argument("--hf-model-id", default="google/flan-t5-base")
    parser.add_argument("--hf-task", default="text2text-generation")
    parser.add_argument("--uc-model-name", required=True, help="Unity Catalog name: <catalog>.<schema>.<model>")
    parser.add_argument("--run-name", default="dspy-context-specialization")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-test-samples", type=int, default=200)
    parser.add_argument("--min-test-f1", type=float, default=0.25)
    parser.add_argument("--min-improvement", type=float, default=0.0)
    parser.add_argument("--skip-quality-gate", action="store_true")

    parser.add_argument("--rag-chunks", default="data/knowledge_base_chunks.jsonl")
    parser.add_argument("--rag-top-k", type=int, default=3)
    parser.add_argument("--rag-min-score", type=float, default=0.02)
    parser.add_argument("--rag-max-chunk-chars", type=int, default=380)
    parser.add_argument("--disable-rag", action="store_true")
    return parser.parse_args()


def _download_hf_snapshot(model_id: str, dst_dir: Path) -> Path:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(dst_dir)

    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_id)

    model.save_pretrained(dst_dir)
    return dst_dir


def _normalize(text: str) -> list[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text)
    return [token for token in cleaned.split() if token]


def _token_f1(gold: str, pred: str) -> float:
    gold_tokens = _normalize(gold)
    pred_tokens = _normalize(pred)

    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0

    gold_counts: dict[str, int] = {}
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1

    overlap = 0
    for token in pred_tokens:
        if gold_counts.get(token, 0) > 0:
            overlap += 1
            gold_counts[token] -= 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _load_test_rows(path: str | Path, max_samples: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        context = str(item.get("context", "")).strip()
        if question and answer:
            rows.append({"question": question, "answer": answer, "context": context})

    if not rows:
        raise ValueError(f"No valid question/answer rows were found in {path}")

    if max_samples > 0:
        return rows[:max_samples]
    return rows


def _build_prompt(
    prompt_artifact: dict[str, Any],
    question: str,
    context: str = "",
    retrieved_chunks: list[dict[str, str | float]] | None = None,
    rag_max_chunk_chars: int = 380,
) -> str:
    instruction = str(prompt_artifact.get("instructions", "Answer accurately in English.")).strip()
    default_context = str(prompt_artifact.get("default_context", "")).strip()
    demos = prompt_artifact.get("few_shot_demos", [])

    lines = [instruction]
    if default_context:
        lines.append(f"Domain context: {default_context}")
    if context:
        lines.append(f"Request context: {context}")

    if isinstance(demos, list) and demos:
        lines.append("Examples:")
        for idx, demo in enumerate(demos[:4], start=1):
            q = str((demo or {}).get("question", "")).strip()
            a = str((demo or {}).get("answer", "")).strip()
            if q and a:
                lines.append(f"{idx}. Q: {q}")
                lines.append(f"   A: {a}")

    if retrieved_chunks:
        lines.append("Knowledge snippets:")
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            source = str(chunk.get("source", "kb"))
            text = str(chunk.get("text", "")).strip()
            if not text:
                continue
            if len(text) > rag_max_chunk_chars:
                text = text[:rag_max_chunk_chars].rstrip() + "..."
            lines.append(f"{idx}. [{source}] {text}")

    lines.append(f"User question: {question}")
    lines.append("Answer:")
    return "\n".join(lines)


def _generate_answer(
    generator: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    result = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
    )

    if isinstance(result, list) and result:
        candidate = result[0]
        if isinstance(candidate, dict):
            for key in ("generated_text", "summary_text", "text"):
                if key in candidate:
                    return str(candidate[key]).strip()
        return str(candidate).strip()

    return str(result).strip()


def _evaluate_prompt_artifact(
    generator: Any,
    test_rows: list[dict[str, str]],
    prompt_artifact: dict[str, Any],
    max_new_tokens: int,
    temperature: float,
    retriever: TfidfRagRetriever | None,
    rag_top_k: int,
    rag_min_score: float,
    rag_max_chunk_chars: int,
) -> float:
    scores: list[float] = []
    for row in test_rows:
        retrieved_chunks: list[dict[str, str | float]] = []
        if retriever is not None:
            retrieval_query = row["question"] if not row.get("context") else f"{row['question']}\n{row['context']}"
            retrieved_chunks = retriever.retrieve(
                query=retrieval_query,
                top_k=rag_top_k,
                min_score=rag_min_score,
            )

        prompt = _build_prompt(
            prompt_artifact,
            question=row["question"],
            context=row.get("context", ""),
            retrieved_chunks=retrieved_chunks,
            rag_max_chunk_chars=rag_max_chunk_chars,
        )
        predicted_answer = _generate_answer(
            generator,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        scores.append(_token_f1(row["answer"], predicted_answer))

    return sum(scores) / len(scores) if scores else 0.0


def _build_baseline_prompt(optimized_prompt: dict[str, Any]) -> dict[str, Any]:
    return {
        "instructions": DEFAULT_BASELINE_INSTRUCTION,
        "default_context": str(optimized_prompt.get("default_context", "")).strip(),
        "few_shot_demos": [],
    }


def _assert_quality_gate(
    optimized_score: float,
    baseline_score: float,
    min_test_f1: float,
    min_improvement: float,
) -> None:
    improvement = optimized_score - baseline_score
    if optimized_score < min_test_f1:
        raise RuntimeError(
            "Quality gate failed: optimized_test_f1 "
            f"{optimized_score:.4f} is below threshold {min_test_f1:.4f}."
        )
    if improvement < min_improvement:
        raise RuntimeError(
            "Quality gate failed: improvement "
            f"{improvement:.4f} is below required minimum {min_improvement:.4f}."
        )


def main() -> None:
    args = parse_args()

    prompt_artifact_path = Path(args.optimized_prompt)
    if not prompt_artifact_path.exists():
        raise FileNotFoundError(f"Optimized prompt file not found: {prompt_artifact_path}")

    test_dataset_path = Path(args.test_dataset)
    if not test_dataset_path.exists():
        raise FileNotFoundError(f"Test dataset file not found: {test_dataset_path}")

    if not (0.0 <= args.min_test_f1 <= 1.0):
        raise ValueError("--min-test-f1 must be in [0.0, 1.0]")
    if args.rag_top_k < 0:
        raise ValueError("--rag-top-k must be non-negative")
    if args.rag_max_chunk_chars < 80:
        raise ValueError("--rag-max-chunk-chars must be at least 80")

    use_rag = not args.disable_rag
    rag_chunks_path = Path(args.rag_chunks)
    if use_rag and not rag_chunks_path.exists():
        raise FileNotFoundError(
            f"RAG chunk file not found: {rag_chunks_path}. Run build_rag_index first or pass --disable-rag."
        )

    if args.experiment_name:
        mlflow.set_experiment(args.experiment_name)

    mlflow.set_registry_uri("databricks-uc")
    os.environ.setdefault("MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC", "True")

    with mlflow.start_run(run_name=args.run_name) as run:
        with tempfile.TemporaryDirectory(prefix="hf_snapshot_") as tmp:
            model_dir = _download_hf_snapshot(args.hf_model_id, Path(tmp) / "hf_model")

            optimized_prompt = json.loads(prompt_artifact_path.read_text(encoding="utf-8"))
            test_rows = _load_test_rows(test_dataset_path, max_samples=args.max_test_samples)
            retriever = TfidfRagRetriever.from_jsonl(rag_chunks_path) if use_rag else None

            device = -1
            if torch is not None and torch.cuda.is_available():
                device = 0

            generator = pipeline(
                args.hf_task,
                model=str(model_dir),
                tokenizer=str(model_dir),
                device=device,
            )

            baseline_prompt = _build_baseline_prompt(optimized_prompt)
            baseline_test_f1 = _evaluate_prompt_artifact(
                generator,
                test_rows,
                baseline_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                retriever=retriever,
                rag_top_k=args.rag_top_k,
                rag_min_score=args.rag_min_score,
                rag_max_chunk_chars=args.rag_max_chunk_chars,
            )
            optimized_test_f1 = _evaluate_prompt_artifact(
                generator,
                test_rows,
                optimized_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                retriever=retriever,
                rag_top_k=args.rag_top_k,
                rag_min_score=args.rag_min_score,
                rag_max_chunk_chars=args.rag_max_chunk_chars,
            )
            improvement = optimized_test_f1 - baseline_test_f1

            mlflow.log_params(
                {
                    "hf_model_id": args.hf_model_id,
                    "hf_task": args.hf_task,
                    "test_dataset": str(test_dataset_path),
                    "test_rows": len(test_rows),
                    "max_test_samples": args.max_test_samples,
                    "min_test_f1": args.min_test_f1,
                    "min_improvement": args.min_improvement,
                    "quality_gate_enabled": not args.skip_quality_gate,
                    "rag_enabled": use_rag,
                    "rag_chunks": str(rag_chunks_path) if use_rag else "",
                    "rag_top_k": args.rag_top_k,
                    "rag_min_score": args.rag_min_score,
                    "rag_max_chunk_chars": args.rag_max_chunk_chars,
                }
            )
            mlflow.log_metrics(
                {
                    "baseline_test_f1": baseline_test_f1,
                    "optimized_test_f1": optimized_test_f1,
                    "test_f1_improvement": improvement,
                }
            )

            print(
                "quality_metrics="
                + json.dumps(
                    {
                        "baseline_test_f1": baseline_test_f1,
                        "optimized_test_f1": optimized_test_f1,
                        "test_f1_improvement": improvement,
                        "min_test_f1": args.min_test_f1,
                        "min_improvement": args.min_improvement,
                        "quality_gate_enabled": not args.skip_quality_gate,
                        "rag_enabled": use_rag,
                    },
                    indent=2,
                )
            )

            if not args.skip_quality_gate:
                try:
                    _assert_quality_gate(
                        optimized_score=optimized_test_f1,
                        baseline_score=baseline_test_f1,
                        min_test_f1=args.min_test_f1,
                        min_improvement=args.min_improvement,
                    )
                    mlflow.set_tag("quality_gate", "passed")
                except RuntimeError:
                    mlflow.set_tag("quality_gate", "failed")
                    raise
            else:
                mlflow.set_tag("quality_gate", "skipped")

            python_model = PromptSpecializedHFModel(
                task=args.hf_task,
                generation=GenerationConfig(
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                ),
                retrieval=RetrievalConfig(
                    enabled=use_rag,
                    top_k=args.rag_top_k,
                    min_score=args.rag_min_score,
                    max_chunk_chars=args.rag_max_chunk_chars,
                ),
            )

            input_example = pd.DataFrame(
                [
                    {
                        "question": "Why is my invoice still in Processing?",
                        "context": "A customer has been waiting for one hour.",
                    }
                ]
            )
            output_example = pd.DataFrame([{"answer": "Check stamping queue and PAC latency first."}])
            signature = infer_signature(input_example, output_example)

            artifacts: dict[str, str] = {
                "optimized_prompt": str(prompt_artifact_path),
                "hf_model_dir": str(model_dir),
            }
            if use_rag:
                artifacts["rag_chunks"] = str(rag_chunks_path)

            model_info = mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=python_model,
                artifacts=artifacts,
                code_path=["src"],
                signature=signature,
                input_example=input_example,
                pip_requirements=[
                    "mlflow>=2.16.0",
                    "pandas>=2.0.0",
                    "transformers>=4.40.0",
                    "torch>=2.1.0",
                ],
                registered_model_name=args.uc_model_name,
            )

            print(f"run_id={run.info.run_id}")
            print(f"model_uri={model_info.model_uri}")
            print(f"registered_model={args.uc_model_name}")


if __name__ == "__main__":
    main()
