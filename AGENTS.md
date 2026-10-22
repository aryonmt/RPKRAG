# Implementation Brief for Coding Agents

This file applies to every coding agent working in this repository, whether it is Cursor, Claude Code, Codex, GitHub Copilot, Gemini CLI, Windsurf, or another assistant. Follow it as the entry point. If your tool uses its own instruction file, that file should simply point here.

You are the senior software engineer responsible for this project. The running code under `src/document_qa/` is the source of truth. The old `specs/` contracts were removed because the program diverged from them. Do not recreate that directory or treat a deleted spec as a requirement.

## Before writing code

1. Inspect the repository structure and the module you are about to change. Do not delete or rewrite existing files without need.
2. Check the versions of LlamaIndex, `llama-index-vector-stores-qdrant`, the Qdrant client, ONNX Runtime, `tokenizers`, and PyMuPDF against the official documentation and `uv.lock`. Do not assume class names or parameters from old examples.
3. Match the behavior already implemented in `services/`, `retrieval/`, `generation/`, and `ingestion/`. If a change would alter answer, citation, or retrieval behavior, say so before broadening the change.

## How the system works

- Routes handle HTTP only. RAG logic lives in `retrieval/`, `generation/`, and `services/`.
- Ingest is a single worker with a process-local queue. Do not parallelize it.
- Each question is retrieved only from the ready documents of its own conversation (`indexed` or `partial_warning`).
- Dense search uses Qdrant. Lexical search uses SQLite FTS5. Results are fused with RRF, then reranked, then validated.
- The embedding model is a `BaseEmbedding` implementation with ONNX Runtime and `tokenizers`, loaded from `EMBEDDING_MODEL_DIR`. There must be no download at runtime and no online embedding service.
- The language model is an OpenAI-compatible chat client. Endpoints come from `llm_services.json` and the `LLM_*` settings. Do not hard-code a provider or model name in code or tests.
- Do not use the global LlamaIndex `Settings`. Inject models and stores.
- Citation cards are built from SQLite chunk metadata. The model must not determine the page number or filename.
- `generation/validator.py` decides whether a claim survives. Do not disable it, skip numeric or lexical checks, or let the model answer from general knowledge when the documents do not support the claim.
- When validation leaves no supported claim, the fixed abstention text in `domain/policies.py` is what the user sees.

## Prohibited

- Using an online embedding service.
- Committing `.env`, API keys, real PDFs, model weights, sample documents, or other local data.
- Logging full document text in application logs. The local trace file under `data/` is the existing debug sink; do not add another one.
- Reporting success for a test or command that was not run or is red.
- Using the global LlamaIndex `Settings`.
- Changing a chunk body to more than 100 tokens or letting the embedding input exceed 128 tokens.
- Unlimited retries of the same language-model request.

## After a change

Report:

1. What changed and which behavior it affects.
2. Which test, lint, and typecheck commands were actually run and what their output was.
3. Whether a live language-model call was run. Do not run one unless the user asks.

Never declare a result successful that you did not actually run.
