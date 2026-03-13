# DSPy + Databricks + Hugging Face Context-Prompt + RAG PoC

This repository contains an end-to-end PoC for:

1. Taking a free Hugging Face model (`google/flan-t5-base` by default).
2. Specializing behavior with DSPy prompt optimization (no weight fine-tuning).
3. Adding local RAG retrieval from a domain knowledge base.
4. Registering the final model in Unity Catalog.
5. Deploying it to Mosaic AI Model Serving.

## Architecture

- `src/poc_dspy_databricks/optimize_prompts.py`
  - Compiles a DSPy program with `MIPROv2` using QA examples.
  - Produces:
    - `artifacts/dspy_optimized/optimized_program.json`
    - `artifacts/dspy_optimized/optimized_prompt.json`
- `src/poc_dspy_databricks/build_rag_index.py`
  - Chunks `.md/.txt` knowledge docs and writes `jsonl` chunks for retrieval.
- `src/poc_dspy_databricks/rag_retriever.py`
  - Lightweight in-process TF-IDF retriever used during evaluation and serving.
- `src/poc_dspy_databricks/log_register_model.py`
  - Downloads HF snapshot.
  - Runs quality gate on held-out test set using the final serving model and optional RAG.
  - Logs MLflow pyfunc and registers to UC only if gate passes.
- `src/poc_dspy_databricks/deploy_mosaic_endpoint.py`
  - Creates/updates Mosaic AI Model Serving endpoint.

## Project Structure

- `config/domain_context.txt`: Domain policy and constraints.
- `data/saas_support_qa.jsonl`: Train/dev examples for DSPy compile.
- `data/saas_support_test_qa.jsonl`: Held-out test set for quality gate.
- `data/knowledge_base.md`: Source knowledge docs for RAG.
- `data/knowledge_base_chunks.jsonl`: Generated RAG chunks.
- `src/poc_dspy_databricks/pyfunc_model.py`: Production inference wrapper (prompt + RAG + generation).

## Prerequisites

- Databricks workspace with:
  - Unity Catalog enabled.
  - Permissions for model registry and serving endpoint management.
- Python 3.10+ environment (Databricks cluster, job, or local dev env).
- PAT token with workspace and serving permissions.

## 1) Install

```bash
pip install -r requirements.txt
```

Set environment variables:

```bash
export DATABRICKS_HOST="https://<your-workspace>.cloud.databricks.com"
export DATABRICKS_TOKEN="<your-databricks-pat>"
export UC_MODEL_NAME="<catalog>.<schema>.dspy_hf_context_poc"
export ENDPOINT_NAME="dspy-hf-context-poc-endpoint"
```

## 2) Optimize Prompt with DSPy

```bash
python3 main.py optimize -- \
  --dataset data/saas_support_qa.jsonl \
  --domain-context-file config/domain_context.txt \
  --output-dir artifacts/dspy_optimized \
  --compiler-model "${DSPY_COMPILER_MODEL:-databricks/databricks-llama-4-maverick}" \
  --num-trials 8 \
  --auto light \
  --enable-mlflow-tracing
```

Artifacts:

- `artifacts/dspy_optimized/compile_summary.json`
- `artifacts/dspy_optimized/optimized_prompt.json`

## 3) Build RAG Chunk Index

```bash
python3 main.py index-kb -- \
  --input-paths data/knowledge_base.md \
  --output-jsonl data/knowledge_base_chunks.jsonl \
  --max-chars 900 \
  --overlap-chars 120
```

## 4) Register Model with Quality Gate

This step evaluates on held-out test data before registration. Registration is blocked if thresholds fail.

```bash
python3 main.py register -- \
  --optimized-prompt artifacts/dspy_optimized/optimized_prompt.json \
  --test-dataset data/saas_support_test_qa.jsonl \
  --hf-model-id google/flan-t5-base \
  --hf-task text2text-generation \
  --rag-chunks data/knowledge_base_chunks.jsonl \
  --rag-top-k 3 \
  --rag-min-score 0.02 \
  --min-test-f1 0.25 \
  --min-improvement 0.00 \
  --uc-model-name "$UC_MODEL_NAME"
```

To register without RAG retrieval, add `--disable-rag`.

## 5) Deploy to Mosaic AI Model Serving

```bash
python3 main.py deploy -- \
  --endpoint-name "$ENDPOINT_NAME" \
  --uc-model-name "$UC_MODEL_NAME" \
  --uc-model-version 1 \
  --workload-size Small \
  --scale-to-zero
```

## 6) Invoke Endpoint

```bash
curl -sS -X POST "$DATABRICKS_HOST/serving-endpoints/$ENDPOINT_NAME/invocations" \
  -H "Authorization: Bearer $DATABRICKS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_records": [
      {
        "question": "Our webhook signature check fails after rotating secrets. What should we validate first?",
        "context": "Production tenant eu-west"
      }
    ]
  }'
```

## Notes

- This PoC combines prompt specialization + local RAG retrieval.
- RAG chunks are static artifacts packaged with the model at registration time.
- For larger scale or frequently changing knowledge, use managed vector search and online retrieval.
