from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import dspy

from .domain_dataset import (
    DEFAULT_DOMAIN_CONTEXT,
    load_domain_examples,
    split_examples,
    to_dspy_examples,
)
from .dspy_program import DomainAssistant


def _normalize(text: str) -> list[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text)
    return [token for token in cleaned.split() if token]


def _read_field(obj: Any, key: str, default: str = "") -> str:
    if hasattr(obj, key):
        value = getattr(obj, key)
        if value is not None:
            return str(value)
    if isinstance(obj, dict):
        value = obj.get(key)
        if value is not None:
            return str(value)
    return default


def answer_token_f1(example: Any, pred: Any, trace: Any = None) -> float:
    del trace
    gold_tokens = _normalize(_read_field(example, "answer"))
    pred_tokens = _normalize(_read_field(pred, "answer"))

    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0

    common = {}
    for token in gold_tokens:
        common[token] = common.get(token, 0) + 1

    overlap = 0
    for token in pred_tokens:
        if common.get(token, 0) > 0:
            overlap += 1
            common[token] -= 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _configure_lm(args: argparse.Namespace) -> None:
    lm_kwargs: dict[str, Any] = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if args.api_key:
        lm_kwargs["api_key"] = args.api_key
    if args.api_base:
        lm_kwargs["api_base"] = args.api_base

    lm = dspy.LM(args.compiler_model, **lm_kwargs)
    if hasattr(dspy, "configure"):
        dspy.configure(lm=lm)
    else:
        dspy.settings.configure(lm=lm)


def _maybe_enable_mlflow_tracing(enabled: bool) -> None:
    if not enabled:
        return
    try:
        import mlflow

        mlflow.dspy.autolog(log_compiles=True, log_traces_from_compile=True)
    except Exception as exc:  # pragma: no cover - optional feature
        print(f"[warn] MLflow DSPy autolog was not enabled: {exc}")


def _evaluate(program: DomainAssistant, devset: list[Any]) -> float:
    scores: list[float] = []
    for ex in devset:
        pred = program(question=_read_field(ex, "question"), context=_read_field(ex, "context"))
        scores.append(answer_token_f1(ex, pred))
    return sum(scores) / len(scores) if scores else 0.0


def _extract_prompt_artifact(
    optimized_program: DomainAssistant,
    context: str,
) -> dict[str, Any]:
    """Build a portable prompt artifact from a compiled DSPy program.

    Why this exists:
    - `optimizer.compile(...)` returns a DSPy program that may contain optimized
      instructions and few-shot demonstrations.
    - In production serving (MLflow pyfunc + HF pipeline), we do not execute
      DSPy compile-time logic again. Instead, we persist only the prompt pieces
      needed at inference time.

    What we extract:
    - `instructions`: optimized instruction text from the predictor signature.
    - `few_shot_demos`: Q/A (and optional context) examples selected by DSPy.
    - `default_context`: domain policy/context loaded from config.

    Fallback behavior:
    - If no optimized instructions are found, we keep a safe default instruction.
    - If demos are missing, the artifact is still valid with an empty demo list.
    """
    predictor = getattr(optimized_program, "respond", None)

    instructions = "You are a specialized support assistant. Respond in English with concise actionable guidance."
    demos: list[dict[str, str]] = []

    if predictor is not None:
        signature = getattr(predictor, "signature", None)
        if signature is not None:
            extracted = str(getattr(signature, "instructions", "")).strip()
            if extracted:
                instructions = extracted

        for demo in getattr(predictor, "demos", []):
            item = {
                "question": _read_field(demo, "question"),
                "context": _read_field(demo, "context"),
                "answer": _read_field(demo, "answer"),
            }
            if item["question"] and item["answer"]:
                demos.append(item)

    return {
        "instructions": instructions,
        "default_context": context,
        "few_shot_demos": demos,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile a DSPy prompt-specialized program")
    parser.add_argument("--dataset", default="data/saas_support_qa.jsonl")
    parser.add_argument("--domain-context-file", default="config/domain_context.txt")
    parser.add_argument("--output-dir", default="artifacts/dspy_optimized")
    parser.add_argument("--compiler-model", default="databricks/databricks-llama-4-maverick")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=350)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-trials", type=int, default=8)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--auto", choices=["light", "medium", "heavy"], default="light")
    parser.add_argument("--enable-mlflow-tracing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    context_text = Path(args.domain_context_file).read_text(encoding="utf-8").strip()
    if not context_text:
        context_text = DEFAULT_DOMAIN_CONTEXT

    examples = load_domain_examples(args.dataset)
    train_rows, dev_rows = split_examples(examples, train_ratio=args.train_ratio, seed=args.seed)
    trainset = to_dspy_examples(train_rows, context=context_text)
    devset = to_dspy_examples(dev_rows, context=context_text)

    _configure_lm(args)
    _maybe_enable_mlflow_tracing(args.enable_mlflow_tracing)

    baseline = DomainAssistant()
    baseline_score = _evaluate(baseline, devset)

    try:
        optimizer = dspy.MIPROv2(metric=answer_token_f1, auto=args.auto, num_threads=args.num_threads)
    except TypeError:  # pragma: no cover - older DSPy compatibility
        optimizer = dspy.MIPROv2(metric=answer_token_f1, num_threads=args.num_threads)

    compile_kwargs: dict[str, Any] = {
        "trainset": trainset,
        "valset": devset,
        "num_trials": args.num_trials,
    }

    try:
        optimized = optimizer.compile(baseline, requires_permission_to_run=False, **compile_kwargs)
    except TypeError:  # pragma: no cover - older DSPy compatibility
        optimized = optimizer.compile(baseline, **compile_kwargs)

    optimized_score = _evaluate(optimized, devset)

    optimized_state_path = output_dir / "optimized_program.json"
    optimized.save(str(optimized_state_path), save_program=False)

    prompt_artifact = _extract_prompt_artifact(optimized, context=context_text)
    prompt_artifact_path = output_dir / "optimized_prompt.json"
    prompt_artifact_path.write_text(json.dumps(prompt_artifact, indent=2), encoding="utf-8")

    summary = {
        "train_size": len(trainset),
        "dev_size": len(devset),
        "baseline_score": baseline_score,
        "optimized_score": optimized_score,
        "compiler_model": args.compiler_model,
        "optimized_state_path": str(optimized_state_path),
        "optimized_prompt_path": str(prompt_artifact_path),
    }
    (output_dir / "compile_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
