# RPKRAG — Evidence-Grounded Persian Document QA System

**RPKRAG** (`document-qa` v2.0.0) is a full-stack Retrieval-Augmented Generation (RAG) platform engineered for high-precision question answering over Persian organizational, legal, financial, and technical documents (`.pdf`, `.txt`, and `.md`).

Unlike generic chat-over-document wrappers that allow language models to improvise answers from general knowledge or hallucinate citations, RPKRAG enforces **strict evidence grounding** across every architectural layer:

* **100% Local CPU Embeddings**: Subword tokenization and 384-dimensional vector embeddings run locally via **ONNX Runtime** (`Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q`) and Hugging Face `tokenizers`. The system performs zero runtime model downloads and never calls external embedding APIs.

* **Conversation-Scoped Hybrid Retrieval**: Combines dense vector similarity search in **Qdrant** with Persian-normalized **SQLite FTS5** (`BM25`) lexical search, merged via **Reciprocal Rank Fusion (RRF)** and refined by an **LLM Reranker**—strictly isolated to the ready documents of the active conversation.

* **Deterministic Post-Generation Claim & Number Validation**: Every claim produced by the language model is audited by a deterministic validator (`src/document_qa/generation/validator.py`). Claims containing unverified numbers or insufficient lexical overlap with cited evidence chunks are automatically stripped before reaching the user.

* **Metadata-Driven Citations & Fixed Policy Responses**: Citation cards (filename, document title, physical PDF page numbers, hierarchical section breadcrumbs, and verbatim quotes) are constructed directly from SQLite chunk metadata—never generated or guessed by the LLM. When evidence is absent or contradictory, the system returns fixed Persian policy messages (`abstained` or `conflict`).

## Project File Structure & Module Guide

The repository is structured into a modular Python backend (`src/document_qa/`), a Next.js 16 RTL frontend (`web/`), Persian lexical resources (`resources/`), and an automated test suite (`tests/`).

```
RPKRAG/
├── docker-compose.yml                  # Orchestrates Qdrant, FastAPI backend, and Next.js web containers
├── Dockerfile                          # Python 3.12-slim-bookworm + uv production image for the API service
├── .dockerignore                       # Excludes local venv, git, web, data, models, eval, and secrets from API build
├── .gitignore                          # Excludes .env, llm_services.json, data/, models/, eval/, and build artifacts
├── pyproject.toml                      # Python package config, dependencies, Ruff linter rules, and Pytest markers
├── .env.example                        # Environment variable template (pre-configured with Docker container paths)
├── llm_services.example.json           # Template for the OpenAI-compatible LLM service pool
├── AGENTS.md                           # Architectural invariants and engineering rules for coding agents
├── README.md                           # Project documentation
│
├── resources/
│   └── fa_stopwords.txt                # Curated Persian stopwords list used by FTS5 query builder & claim validator
│
├── src/
│   └── document_qa/                    # Core Python backend package (single source of truth)
│       ├── main.py                     # FastAPI app factory, lifespan hooks (SQLite & Qdrant init), CORS, error handlers
│       │
│       ├── core/                       # Configuration, deterministic UUID generation, and logging
│       │   ├── config.py               # Pydantic BaseSettings loading environment variables from .env
│       │   ├── ids.py                  # UUIDv4 entity IDs & deterministic UUIDv5 chunk node_id generator
│       │   └── logging.py              # Console + rotating file logger writing to data/logs/backend.log
│       │
│       ├── domain/                     # Core domain models, lifecycle enums, and fixed policy strings
│       │   ├── models.py               # Enums (DocumentStatus, ConversationStatus, AnswerStatus) & Pydantic entities
│       │   └── policies.py             # Fixed Persian strings for abstention, conflict, and missing sub-questions
│       │
│       ├── api/                        # HTTP transport layer (request schemas, error envelopes, route handlers)
│       │   ├── errors.py               # AppError exception class and stable Persian error message mappings
│       │   ├── schemas.py              # Pydantic request/response schemas for conversations, messages, and citations
│       │   └── routes/
│       │       ├── conversations.py    # CRUD routes for conversations (/api/v1/conversations)
│       │       ├── documents.py        # Multipart file upload, background ingest trigger, list/get/delete documents
│       │       └── messages.py         # Question answering endpoint (?trace=true support) & message history
│       │
│       ├── services/                   # Application orchestration & business logic layer
│       │   ├── document_service.py     # Upload validation, SHA-256 deduplication, and atomic multi-store deletion
│       │   └── qa_service.py           # End-to-end QA pipeline (multi-query, hybrid search, RRF, rerank, math, validate)
│       │
│       ├── ingestion/                  # Document extraction, cleaning, Persian normalization, and chunking
│       │   ├── extractor.py            # PyMuPDF PDF extraction, low-text/garbage page detection, UTF-8-SIG text reader
│       │   ├── cleaner.py              # Repeated header/footer/page-number removal and broken Persian line merging
│       │   ├── normalizer.py           # Persian/Arabic Unicode NFKC normalization, digit folding, ZWNJ & Ezafe handling
│       │   ├── headings.py             # 4-level hierarchical breadcrumb state machine (Chapter > Article > Note > Clause)
│       │   ├── chunker.py              # Local HuggingFace tokenizer wrapper & sentence-packing chunker (<=100 body tokens)
│       │   └── pipeline.py             # Coordinates extract -> clean -> chunk -> ONNX embed -> Qdrant + SQLite save
│       │
│       ├── retrieval/                  # Dense vector, lexical FTS5, and hybrid fusion retrieval
│       │   ├── embedding.py            # LocalOnnxEmbedding (CPU ONNX Runtime + mean pooling) & FakeEmbedding for tests
│       │   ├── qdrant_index.py         # Qdrant collection initialization, keyword payload indexes, and point deletion
│       │   ├── fts_query.py            # Normalizes queries, removes stopwords, and builds safe FTS5 MATCH expressions
│       │   ├── fts_retriever.py        # SQLite FTS5 BM25 retriever filtered by conversation & ready documents
│       │   └── fusion.py               # Reciprocal Rank Fusion (RRF) combining dense and lexical ranked lists
│       │
│       ├── generation/                 # LLM service pool, prompts, chat client, and deterministic validator
│       │   ├── service_pool.py         # Load-balanced LLM endpoint pool with 60s rate-limit cooldowns & env fallback
│       │   ├── prompts.py              # Prompts for query expansion, chunk reranking, and strict JSON answer generation
│       │   ├── answerer.py             # OpenAILike LLM client wrapper with 1x network retry and 429 failover
│       │   └── validator.py            # Deterministic JSON parser, conflict checker, exact number auditor, lexical validator
│       │
│       └── storage/                    # Relational database connection factory and transactional repositories
│           ├── sqlite.py               # SQLite WAL connection setup, foreign keys, and table/FTS5 initialization
│           └── repositories.py         # Transactional chunk + FTS5 persistence with optimistic generation checking
│
├── web/                                # Next.js 16 + React 19 + Tailwind CSS v4 RTL Persian web application
│   ├── Dockerfile                      # Multi-stage Node 22 standalone production image
│   ├── .dockerignore                   # Excludes node_modules, .next, and .git from frontend image build
│   ├── next.config.ts                  # Next.js configuration (output: "standalone")
│   ├── tsconfig.json                   # TypeScript compiler configuration and @/* path alias
│   ├── package.json                    # Frontend dependencies (Next 16.4.2, React 19.3.0, Tailwind 4, Lucide icons)
│   └── src/
│       ├── app/
│       │   ├── layout.tsx              # Root RTL Persian layout (<html lang="fa" dir="rtl">) with Vazirmatn font
│       │   ├── page.tsx                # Main workspace layout coordinating Sidebar, ChatArea, DocumentsPanel, PrivacyModal
│       │   └── globals.css             # Tailwind v4 imports, custom RTL scrollbar styling, and CSS variables
│       ├── components/
│       │   ├── Sidebar.tsx             # Conversation list, 5s background sync, search filter, inline rename, delete
│       │   ├── ChatArea.tsx            # Message stream, expandable citation cards, inline file attachment, error banners
│       │   ├── DocumentsPanel.tsx      # Slide-over drawer, drag-and-drop uploader, 2s processing poll, document delete
│       │   └── PrivacyModal.tsx        # First-visit data privacy acknowledgment modal persisted in localStorage
│       └── lib/
│           └── api.ts                  # Typed Fetch client for all /api/v1 endpoints
│
├── tests/                              # Pytest suite (offline unit tests + optional model/integration/live markers)
│   ├── test_normalizer.py              # Digit folding, Arabic-to-Persian character mapping, and stopword loading tests
│   ├── test_extractor.py               # PDF typo correction and plain text extraction tests
│   ├── test_cleaner.py                 # Repeated header/footer removal and heading breadcrumb state tests
│   ├── test_chunker.py                 # Token cap enforcement (<=100 chunk tokens, <=128 embedding tokens)
│   ├── test_fts_query.py               # FTS5 query quoting and stopword removal tests
│   ├── test_validator.py               # Comprehensive unit tests for claim validation, number checks, and abstention
│   ├── test_embedding.py               # FakeEmbedding unit test + @pytest.mark.model real ONNX cosine similarity test
│   ├── test_qdrant.py                  # @pytest.mark.integration Qdrant insertion and conversation isolation test
│   ├── test_full_ingest.py             # @pytest.mark.integration + @pytest.mark.model end-to-end ingestion & hybrid search
│   ├── test_api_keys.py                # @pytest.mark.live connectivity checks for external LLM providers
│   ├── test_upload_api.py              # @pytest.mark.live upload endpoint and status polling test
│   ├── test_generation.py              # @pytest.mark.live + @pytest.mark.model Qdrant retrieval + LLM generation test
│   └── test_chat_api.py                # @pytest.mark.live full end-to-end conversation, upload, and question answering test
│
├── docs/
│   └── samples/
│       └── large-samples/              # Ten large Persian sample texts and reference answers tracked in Git
│
└── [Untracked Local Runtime Paths]     # Created locally during setup/runtime (git-ignored)
    ├── .env                            # Local environment configuration
    ├── llm_services.json               # Active LLM provider credentials and models
    ├── models/                         # Local ONNX embedding model directory
    │   └── paraphrase-multilingual-MiniLM-L12-v2-onnx-Q/
    │       ├── model_optimized.onnx    # Quantized ONNX model weights
    │       └── tokenizer.json          # Hugging Face fast tokenizer definition
    └── data/                           # Persistent local runtime storage
        ├── app.db                      # SQLite database (conversations, documents, chunks, chunks_fts, messages)
        ├── uploads/                    # Raw uploaded files (<document_id>.pdf/.txt/.md)
        ├── latest_trace.json           # Detailed execution trace of the most recent QA request
        └── logs/
            └── backend.log             # Rotating application log file (10 MB x 5 backups)

```

## System Architecture & Component Breakdown

### High-Level Data Flow

```
Browser Client (Next.js 16 RTL UI on http://localhost:3000)
   │
   │  HTTP JSON & Multipart REST Calls (/api/v1/*)
   ▼
FastAPI Backend Service (http://localhost:8000)
   ├── API Layer (api/routes/*) ────────────► Request validation, HTTP status codes, Persian error envelopes
   ├── Document Service (document_service) ─► SHA-256 deduplication, raw file storage (data/uploads/), multi-store cleanup
   ├── Ingestion Pipeline (ingestion/*) ────► PyMuPDF extract -> Header/line clean -> Heading breadcrumb -> Token chunker
   ├── Local ONNX Embedder (retrieval/*) ───► CPU ONNX Runtime + HuggingFace Tokenizer (384-dim, <=128 tokens, offline)
   ├── Dual Storage Layer
   │      ├── SQLite WAL + FTS5 (data/app.db) ─► Conversations, Documents, Chunks metadata, BM25 lexical index, Messages
   │      └── Qdrant Vector DB (Port 6333) ────► 384-dim Cosine vectors indexed by conversation_id & document_id
   └── QA Service & Validator (qa_service / generation/*)
          ├── 1. Multi-Query Expansion ─────► Rewrites question into 3 Persian sub-queries via LLM Service Pool
          ├── 2. Hybrid Retrieval & RRF ────► Merges Qdrant Dense Top-K + SQLite FTS5 BM25 Top-K via RRF (k=60)
          ├── 3. LLM Relevance Reranker ────► Scores up to 50 candidate chunks (0-10 scale), selects top EVIDENCE_MAX
          ├── 4. Deterministic Math Check ──► Injects exact downtime/availability calculation block when applicable
          ├── 5. Constrained Generation ────► Temperature 0.0 JSON output with per-claim evidence_ids
          └── 6. Post-Generation Validator ─► Verifies exact numbers, lexical overlap (>=0.2), and conflict distinctness

```

### Core Architectural Invariants

1. **Strict Transport vs. Domain Separation**: Route handlers under `src/document_qa/api/routes/` only handle HTTP parsing and status codes. All RAG logic resides in `services/`, `ingestion/`, `retrieval/`, and `generation/`.

2. **No Global LlamaIndex Singleton**: The system never reads or mutates the global LlamaIndex `Settings` object. Embeddings, vector store clients, SQLite connections, and LLM instances are explicitly constructed and injected.

3. **Conversation-Level Data Isolation**: Every Qdrant vector query and SQLite FTS5 query filters strictly by `conversation_id` and restricts results to documents in that conversation whose status is `indexed` or `partial_warning`. Documents from other conversations—or documents currently `processing`, `failed`, or `deleting`—can never leak into retrieval.

4. **Single-Worker Process-Local Ingestion**: File uploads return HTTP `202 Accepted` immediately and run `process_document` via FastAPI `BackgroundTasks` inside the API process. Always run a single Uvicorn process (`--workers 1`) so background ingestion and SQLite WAL transactions remain serialized.

### Deep Dive 1: The Ingestion & Indexing Pipeline (`src/document_qa/ingestion/`)

#### Step 1 — Upload Validation & Deduplication (`services/document_service.py`)

* Confirms the target `conversation_id` exists and has `status = 'active'`.

* Sanitizes the uploaded filename (`safe_filename` strips control characters with `ord(c) < 32` and truncates to 120 characters) and checks that the extension is `.pdf`, `.txt`, or `.md`.

* Streams the uploaded file to `UPLOAD_DIR/<doc_id>.<ext>` in 8,192-byte chunks while enforcing `MAX_FILE_MB` (`80 MB`). If the stream exceeds 80 MB, the partial file is unlinked and HTTP `413` (`file_too_large`) is raised.

* For `.pdf` files, verifies that the first 5 bytes equal `b"%PDF-"` (raising `422 invalid_document` otherwise).

* Verifies that the conversation has fewer than `MAX_DOCUMENTS_PER_CONVERSATION` (`50`) non-deleting documents.

* Computes the **SHA-256** digest (`calculate_sha256`) and queries SQLite for any non-deleting document in the same conversation with an identical `file_hash`. If a match exists, the newly written file is deleted and `{"duplicate": True, "document_id": existing_id}` is returned without re-running ingestion.

* Inserts a new row in `documents` with `status = 'processing'` and `generation = 1`, enqueues `run_ingest_background`, and returns HTTP `202 Accepted`.

#### Step 2 — Text Extraction & Low-Text / Garbage Detection (`ingestion/extractor.py`)

* **PDF Files (`extract_pdf`)**:

  * Opens the PDF via PyMuPDF (`fitz.open`). Rejects encrypted PDFs (`pdf_encrypted`) and PDFs with `page_count > MAX_PDF_PAGES` (`2000`).

  * Extracts each page using `page.get_text("text", sort=True)` to preserve top-to-bottom reading order.

  * Applies targeted Persian ligature/typo normalization (`fix_persian_pdf_text`, correcting common PDF extraction artifacts like `"صالحیت"` $\rightarrow$ `"صلاحیت"`).

  * Flags `is_low_text = True` if the stripped page has fewer than `LOW_TEXT_PAGE_CHARS` (`40`) characters or triggers `is_garbage_page(text)` (zero alphabetic characters, or digits outnumber letters by more than $2\times$ when total letters $< 40$).

* **Text & Markdown Files (`extract_text_file`)**:

  * Reads `.txt` and `.md` files with `encoding="utf-8-sig"` (automatically stripping UTF-8 BOM if present) and wraps the content as a single page object with `page_number = None`.

#### Step 3 — Header/Footer Stripping & Broken Line Merging (`ingestion/cleaner.py`)

* **Repeated Header/Footer Detection**: Collects the first 3 and last 3 non-empty lines of every page across the document. Any top or bottom line appearing on at least $\max(3, \lfloor\text{total\_pages} / 2\rfloor)$ pages is identified as a repeating running header or footer and removed.

* **Standalone Page Number Removal**: Strips top/bottom lines matching `^\s*(صفحه|از)?\s*[\d۰-۹٠-٩]+\s*(از)?\s*[\d۰-۹٠-٩]*\s*$`. Whenever lines are removed, a Persian warning (`"<N> خط تکراری (هدر/فوتر/شماره صفحه) حذف شد."`) is attached to the document, causing its final status to be saved as `partial_warning` rather than `indexed` (both statuses are fully active for retrieval).

* **Persian Broken Line Merging**: Reconstructs sentences split across lines by PDF layout wrapping. If the previous non-empty line does not end with sentence punctuation (`[.\؟!;:\)]$`) and the current line does not start with a heading or bullet marker (`^(#|[-*•]|فصل|بخش|ماده|تبصره|بند)`), the two lines are joined with a space.

#### Step 4 — Dual-Representation Persian Normalization (`ingestion/normalizer.py`)

Every chunk preserves `original_text` (exact original text for citations) alongside `search_text` (normalized via `normalize_for_search`). Normalization applies 7 deterministic transformations:

1. **Unicode NFKC**: Normalizes compatibility and presentation glyphs to canonical forms.

2. **Control & Tatweel Removal**: Removes zero-width space (`U+200B`), LTR/RTL marks (`U+200E`, `U+200F`, `U+202A`–`U+202E`), BOM (`U+FEFF`), and kashida/tatweel (`U+0640`).

3. **Arabic-to-Persian Letter Mapping**: Maps `ي` (`U+064A`), `ى` (`U+0649`), `ے` (`U+06D2`) $\rightarrow$ `ی` (`U+06CC`) and `ك` (`U+0643`), `ڪ` (`U+06AA`) $\rightarrow$ `ک` (`U+06A9`).

4. **Digit Folding (`fold_digits`)**: Converts Persian (`۰`–`۹`) and Arabic-Indic (`٠`–`٩`) digits to ASCII (`0`–`9`), `٫` $\rightarrow$ `.`, and `٪` $\rightarrow$ `%`.

5. **Diacritic Stripping with Ezafe Preservation**: Removes harakat (`U+064B`–`U+0653`, `U+0655`–`U+065F`, `U+0670`) while explicitly preserving `U+0654` (Persian Ezafe marker).

6. **ZWNJ Normalization**: Converts Zero-Width Non-Joiner (`U+200C`, نیم‌فاصله) to a standard space so morphological variants match consistently across dense and lexical search.

7. **Whitespace Collapsing**: Collapses multi-space sequences into a single space and trims edges.

#### Step 5 — Hierarchical Breadcrumbs & Token-Capped Chunking (`ingestion/headings.py`, `ingestion/chunker.py`)

* **4-Level Breadcrumb State Machine (`HeadingState`)**: Tracks structural context across pages for both Persian legal/organizational documents and Markdown files:

  * **Level 1 (`chapter`)**: Markdown `# ` or `^(فصل|بخش)\s+([\d۰-۹٠-٩]+|اول|دوم|سوم|چهارم|پنجم|ششم|هفتم|هشتم|نهم|دهم)`

  * **Level 2 (`article`)**: Markdown `## ` or `^ماده\s*[\(]?\s*([\d۰-۹٠-٩]+)\s*[\)]?`

  * **Level 3 (`note`)**: Markdown `### ` or `^تبصره\s*(\d+|:|$)`

  * **Level 4 (`clause`)**: Markdown `####`–`######` or `^بند\s+([\d۰-۹٠-٩]+|[آ-ی]|الف|ب|ج|د)` When a higher-level heading appears, all subordinate levels reset to `None`. The active hierarchy is formatted as `"فصل اول > ماده ۲ > تبصره ۱"` and truncated at `BREADCRUMB_MAX_TOKENS` (`20` tokens).

* **Token-Capped Sentence Chunker (`chunk_pages`)**:

  * Uses `LocalTokenizer` (wrapping `<EMBEDDING_MODEL_DIR>/tokenizer.json`) to count exact subword tokens.

  * Splits `.md` pages on `\n(?=#{1,3}\s)` and `.pdf`/`.txt` pages on `\n\s*\n`.

  * Checks `state.is_heading(...)` on the first line of each block **before** mutating state: if a block begins with a heading, any accumulated `current_body` is flushed under the previous breadcrumb first, and only then does `state.process_line(...)` advance the breadcrumb for the new chunk.

  * Packs sentences into chunks while enforcing two strict invariants:

    1. `body_tokens <= CHUNK_MAX_TOKENS` (`100` tokens).

    2. `embed_tokens` (`normalize_for_search(breadcrumb + "\n" + body)` + 2 special tokens) `<= EMBEDDING_MAX_TOKENS` (`128` tokens).

  * Carries over up to `CHUNK_OVERLAP_TOKENS` (`16` tokens, from the last 20 words) between split chunks, and falls back to word-by-word splitting (`_split_oversized_text`) if a single sentence exceeds 100 tokens.

  * Computes exact character offsets (`char_start`, `char_end`) within the page and assigns a deterministic **UUIDv5 `node_id`** (`core/ids.py`) derived from `f"{document_id}:{ordinal}:{sha256(original_text)}"`.

#### Step 6 — Local ONNX Embedding & Transactional Storage (`retrieval/embedding.py`, `ingestion/pipeline.py`, `storage/repositories.py`)

* **`LocalOnnxEmbedding`**: Loads `model_optimized.onnx` via `onnxruntime.InferenceSession` (`CPUExecutionProvider`, `intra_op_num_threads = EMBEDDING_THREADS`) and `tokenizer.json` with truncation disabled (`_tokenizer.no_truncation()`). If any input exceeds `EMBEDDING_MAX_TOKENS` (`128`), it raises `EmbeddingInputTooLong` rather than silently truncating text. Performs attention-masked mean pooling (`_mean_pooling`) over the 3D hidden state `(batch, seq_len, 384)` to produce 384-dimensional vectors.

* **Qdrant Upsert & Optimistic SQLite Commit**: Uploads `TextNode` batches of 64 to Qdrant, then calls `save_document_chunks` inside a SQLite transaction. `save_document_chunks` verifies that the document is still in `status = 'processing'` with `generation == expected_generation`. If the user deleted the document while ingestion was running, `save_document_chunks` returns `False` and `pipeline.py` immediately deletes the newly uploaded Qdrant points (`delete_document_points`) to prevent orphaned vectors.

### Deep Dive 2: Hybrid Retrieval, Reranking & Answer Validation (`services/qa_service.py`, `generation/`)

1. **Service Pool & Automatic Failover (`generation/service_pool.py`, `generation/answerer.py`)**:

   * Loads OpenAI-compatible endpoints from `llm_services.json` (or falls back to `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` in `.env` if `llm_services.json` is absent).

   * Requires `LLM_DATA_POLICY_ACK=true` in `.env`; otherwise the question returns HTTP 503 `llm_not_configured`.

   * Load-balances randomly across healthy services. If a service returns HTTP `429`, `"quota"`, or `"rate limit"`, `ServicePool` marks that service on cooldown for **60 seconds** and `LLMService.chat` transparently retries the call on another service in the pool.

   * Performs at most **1 retry** (after a 1s delay) on transient network connection errors or HTTP `500`/`502`/`503`/`504` responses, and never retries on timeouts.

2. **Multi-Query Expansion (`_generate_multi_queries`)**:

   * Rewrites the user's question into 3 short (<15 words) Persian sub-queries (`build_query_expansion_prompt`). If the question contains a number or specific entity, the prompt enforces keeping the exact number/entity in one sub-query and expanding the surrounding concept in another.

3. **Scoped Dense + Lexical Retrieval & RRF Fusion (`retrieval/`)**:

   * Queries Qdrant (`RETRIEVAL_DENSE_TOP_K`) and SQLite FTS5 (`RETRIEVAL_LEXICAL_TOP_K`) across the original question and all expanded sub-queries, keeping the highest score per `node_id` in each modality.

   * Merges both ranked lists using **Reciprocal Rank Fusion** (`rrf_fusion`, $k = 60$, `top_n = 60`), hydrates full chunk records from SQLite, and deduplicates chunks with identical `original_text`.

4. **LLM Relevance Reranking (`_rerank_with_llm`)**:

   * Sends up to 50 fused chunks (`E1`–`E50`) to the LLM relevance judge (`build_rerank_prompt`), which assigns each chunk an integer score from `0` to `10`. Re-sorts chunks by score descending and slices the top `EVIDENCE_MAX` chunks (`25` in `.env.example`, `15` default in `config.py`).

5. **Deterministic Availability / SLA Math Override (`_calculate_math_override`)**:

   * Scans the digit-folded question for both a percentage (`(\d+(?:\.\d+)?)\s*%`) and a day count (`(\d+(?:\.\d+)?)\s*روز`). When present, computes exact total minutes ($\text{days} \times 1440$) and downtime minutes in Python and injects a `[سیستمی - محاسبه دقیق]` block into the prompt. The system prompt (`Rule 13`) mandates quoting these exact figures.

6. **Deterministic Claim & Number Validator (`generation/validator.py`)**:

   * Parses the LLM's raw JSON response (`status`, `claims`, `missing`, `conflict_evidence_ids`).

   * **Conflict Check**: For `status == "conflict"`, verifies at least two valid `conflict_evidence_ids` whose chunks differ in `document_id`, `page_start`, or non-overlapping `char_start`/`char_end` offsets and have distinct `original_text`. Otherwise downgrades to `abstained`.

   * **Evidence Resolution**: Resolves `evidence_ids` for each claim from the claim dict, inline `(E1)`/`[E1]` tags, top-level `evidence_ids`, or fallback lexical matching (`_best_evidence` with $\ge 0.5$ overlap).

   * **Exact Number Verification**: Strips `(E1)`/`[E1]` markers and list prefixes (`1.`, `2)`), folds digits, and extracts all numbers (`\d+(?:\.\d+)?`). Every number in a claim **must** exist in the cited evidence chunks (`original_text` + `breadcrumb`), the user's question, or the `[سیستمی - محاسبه دقیق]` block. If a number is missing from the cited chunk(s), the validator checks if another chunk in the `evidence_chunks` bundle contains that number **and** shares $\ge$ `LEXICAL_SUPPORT_MIN_RATIO` (`0.2`) token overlap with the claim; if so, that chunk is automatically added to the claim's citations. If any number remains unverified, **the claim is rejected**.

   * **Lexical Support Check**: Tokenizes the claim (removing Persian stopwords from `resources/fa_stopwords.txt`, single-character tokens, and numbers) and verifies that at least `LEXICAL_SUPPORT_MIN_RATIO` (`0.2`) of the claim's content tokens appear in the cited evidence chunks. Claims below threshold are rejected.

   * **Abstention Fallback**: If all claims are rejected, the validator returns `status = "abstained"` and the user receives the fixed `ABSTENTION_TEXT` policy string.

## Data Modeling & Database Schemas

### 1. SQLite Relational & FTS5 Schema (`src/document_qa/storage/sqlite.py`)

All SQLite connections enable Write-Ahead Logging (`PRAGMA journal_mode=WAL;`), foreign keys (`PRAGMA foreign_keys=ON;`), and a 5-second busy timeout (`PRAGMA busy_timeout=5000;`).

| Table | Primary Key | Columns & Constraints | 
| ----- | ----- | ----- | 
| `conversations` | `id` (`TEXT` UUIDv4) | `title TEXT NOT NULL`, `status TEXT NOT NULL DEFAULT 'active'`, `created_at TEXT NOT NULL`, `updated_at TEXT NOT NULL`. | 
| `documents` | `id` (`TEXT` UUIDv4) | `conversation_id TEXT NOT NULL` (FK $\rightarrow$ `conversations.id ON DELETE CASCADE`), `filename TEXT NOT NULL`, `title TEXT`, `file_type TEXT NOT NULL`, `file_hash TEXT NOT NULL`, `status TEXT NOT NULL`, `generation INTEGER NOT NULL DEFAULT 1`, `page_count INTEGER`, `chunk_count INTEGER NOT NULL DEFAULT 0`, `warnings TEXT NOT NULL DEFAULT '[]'`, `error_code TEXT`, `error_message TEXT`, `cleanup_pending INTEGER NOT NULL DEFAULT 0`, `raw_file_path TEXT`, `created_at TEXT NOT NULL`, `updated_at TEXT NOT NULL`. Unique partial index: `idx_doc_hash ON documents(conversation_id, file_hash) WHERE status != 'deleting'`. | 
| `chunks` | `node_id` (`TEXT` UUIDv5) | `conversation_id TEXT NOT NULL` (FK cascade), `document_id TEXT NOT NULL` (FK cascade), `ordinal INTEGER NOT NULL`, `filename TEXT NOT NULL`, `title TEXT NOT NULL`, `page_start INTEGER`, `page_end INTEGER`, `breadcrumb TEXT`, `original_text TEXT NOT NULL`, `search_text TEXT NOT NULL`, `char_start INTEGER NOT NULL`, `char_end INTEGER NOT NULL`, `token_count INTEGER NOT NULL`. | 
| `chunks_fts` | Virtual Table (`fts5`) | `node_id UNINDEXED`, `body_search`, `breadcrumb_search`, `title_search`, configured with `tokenize='unicode61'`. | 
| `messages` | `id` (`TEXT` UUIDv4) | `conversation_id TEXT NOT NULL` (FK cascade), `role TEXT NOT NULL` (`user` or `assistant`), `content TEXT NOT NULL`, `standalone_question TEXT`, `status TEXT`, `answer_text TEXT`, `missing TEXT` (JSON list), `citations TEXT` (JSON list), `created_at TEXT NOT NULL`. | 
| `index_metadata` | `key` (`TEXT`) | `value TEXT NOT NULL`. | 

### 2. Domain Status Lifecycles (`src/document_qa/domain/models.py`)

| Enum | Values | Description | 
| ----- | ----- | ----- | 
| `ConversationStatus` | `active`, `deleting` | Controls visibility in conversation listings and blocks uploads/questions during deletion. | 
| `DocumentStatus` | `processing`, `indexed`, `partial_warning`, `failed`, `deleting` | Only `indexed` and `partial_warning` documents are eligible for retrieval. | 
| `AnswerStatus` | `answered`, `abstained`, `conflict` | Determines how the UI renders the message bubble and whether fixed policy text is used. | 

### 3. Fixed Persian Policy Texts (`src/document_qa/domain/policies.py`)

| Constant | Exact Persian Output | When Used | 
| ----- | ----- | ----- | 
| `ABSTENTION_TEXT` | `در اسناد این گفتگو اطلاعات کافی برای پاسخ به این پرسش یافت نشد.` | Returned whenever `status == "abstained"`. | 
| `CONFLICT_TEXT` | `در سندهای این گفتگو دو حکم ناسازگار آمده است.` | Returned whenever `status == "conflict"` passes distinct-chunk validation. | 
| `MISSING_PREFIX` | `این بخش در اسناد این گفتگو پیدا نشد:` | Appended below `answered` claims when validated `missing` sub-questions exist. | 

## Step-by-Step Setup & Installation Guide

You can bring up the project using either **Docker Compose (Full Stack)** or **Local Development Mode** (Qdrant in Docker + FastAPI and Next.js running natively on your host).

Both modes require three untracked local items before starting:

1. The local ONNX embedding model files under `models/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q/`

2. The `.env` configuration file

3. The `llm_services.json` provider configuration file

### Prerequisites

* **Git**

* **Docker Desktop** or **Docker Engine** with **Docker Compose**

* **For Local Development Mode only**:

  * **Python 3.12+**

  * [**`uv`**](https://docs.astral.sh/uv/) (Python environment and package manager)

  * **Node.js 22+** and **npm**

### Step 1: Clone the Repository

```
git clone https://github.com/aryonmt/RPKRAG.git
cd RPKRAG

```

*(Note: `data/`, `models/`, `eval/`, `.env`, and `llm_services.json` are git-ignored and will not exist immediately after cloning.)*

### Step 2: Download and Place the Local ONNX Embedding Model

The backend loads **`Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q`** (384 dimensions, 128 max tokens) directly from disk. It never downloads models at runtime.

1. Create the target directory:

   **Linux / macOS:**

   ```
   mkdir -p models/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q
   
   ```

   **Windows (PowerShell):**

   ```
   New-Item -ItemType Directory -Force -Path models\paraphrase-multilingual-MiniLM-L12-v2-onnx-Q
   
   ```

2. Download **`model_optimized.onnx`** and **`tokenizer.json`** from the `main` branch of [Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q on Hugging Face](https://www.google.com/search?q=https://huggingface.co/Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q/tree/main) into that directory:

   ```
   curl -L https://huggingface.co/Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q/resolve/main/model_optimized.onnx \
     -o models/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q/model_optimized.onnx
   
   curl -L https://huggingface.co/Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q/resolve/main/tokenizer.json \
     -o models/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q/tokenizer.json
   
   ```

   > **Important**: Always download `tokenizer.json` from revision `main` on Hugging Face (older revisions of that repository did not include `tokenizer.json`). Confirm both `model_optimized.onnx` and `tokenizer.json` are present before starting the server.

### Step 3: Create `.env` and `llm_services.json`

1. Copy the template files:

   **Linux / macOS:**

   ```
   cp .env.example .env
   cp llm_services.example.json llm_services.json
   
   ```

   **Windows (PowerShell):**

   ```
   Copy-Item .env.example .env
   Copy-Item llm_services.example.json llm_services.json
   
   ```

2. **Populate `llm_services.json`**: Edit `llm_services.json` to list one or more OpenAI-compatible Chat Completions endpoints. Each entry must have a unique `name`:

   ```
   [
     {
       "name": "primary",
       "base_url": "https://api.openai.com/v1",
       "api_key": "sk-your-real-api-key",
       "model": "gpt-4o-mini"
     }
   ]
   
   ```

   *(You can add multiple entries from OpenAI, OpenRouter, Gemini's OpenAI-compatible API, NVIDIA NIM, or a local vLLM/Ollama server; `ServicePool` will distribute calls across them and automatically cool down any endpoint that returns HTTP `429`.)*

3. **Configure `.env`**:

   * **Enable `LLM_DATA_POLICY_ACK` (Mandatory)**: After verifying your LLM provider's data policy, set:

     ```
     LLM_DATA_POLICY_ACK=true
     
     ```

     Until this is `true`, a question returns HTTP 503 with code `llm_not_configured`.

   * **Check Paths for Your Execution Mode**:

     * **For Docker Compose (Step 4A)**: Keep the container paths in `.env.example` (`/app/data/app.db`, `/app/data/uploads`, `http://qdrant:6333`, `/app/models/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q`). Note that `docker-compose.yml` also overrides these four environment variables inside the `api` container.

     * **For Local Development (Step 4B)**: Change those four variables in `.env` to host-relative paths:

       ```
       DATABASE_PATH=./data/app.db
       UPLOAD_DIR=./data/uploads
       QDRANT_URL=http://localhost:6333
       EMBEDDING_MODEL_DIR=./models/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q
       
       ```

   * **Frontend API URL (`NEXT_PUBLIC_API_BASE_URL`)**: Keep `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/api/v1`. This URL is called by the browser on your host machine (not container-to-container) and is baked into the Next.js production build during `docker compose build`.

### Step 4A: Full Stack Deployment with Docker Compose (Recommended)

From the root of the repository:

```
docker compose up --build

```

> **Mount Check**: Ensure `llm_services.json` exists as a file before running `docker compose up`. `docker-compose.yml` mounts `./llm_services.json:/app/llm_services.json:ro`, which will fail if `llm_services.json` was not created in Step 3.

**Container Volume Mounts (`api` service):**

| Host Path | Container Path | Access Mode | 
| ----- | ----- | ----- | 
| `./data` | `/app/data` | Read / Write | 
| `./models` | `/app/models` | Read / Write | 
| `./llm_services.json` | `/app/llm_services.json` | Read-Only (`:ro`) | 

**Service Endpoints:**

* **Next.js Web UI**: <http://localhost:3000>

* **FastAPI Backend & Swagger UI**: <http://localhost:8000/docs>

* **Health Check**: <http://localhost:8000/api/v1/health> $\rightarrow$ `{"status":"ok","version":"2.0.0"}`

* **Qdrant Vector Database**: `http://127.0.0.1:6333`

**Stopping the Stack:**

```
# Stop containers while keeping Qdrant vectors and ./data
docker compose down

# Stop containers and remove the Qdrant Docker volume
docker compose down -v

```

*(Because `./data` is a host bind-mount, deleting conversations and uploaded files requires removing `./data` on the host.)*

### Step 4B: Local Development Setup (Host API + Host Web + Docker Qdrant)

1. **Start Qdrant in Docker**:

   ```
   docker compose up -d qdrant
   
   ```

2. **Install Python Dependencies & Run FastAPI**: Make sure `.env` has local host paths (`./data/...`, `http://localhost:6333`, `./models/...`) and `LLM_DATA_POLICY_ACK=true`, then run:

   ```
   uv sync
   uv run uvicorn document_qa.main:app --reload --host 127.0.0.1 --port 8000
   
   ```

3. **Install Frontend Dependencies & Run Next.js Dev Server**: In a second terminal:

   ```
   npm ci --prefix web
   npm run dev --prefix web
   
   ```

   Open <http://localhost:3000>. In `npm run dev` mode, Next.js defaults to `http://localhost:8000/api/v1` (via `web/src/lib/api.ts`) unless `NEXT_PUBLIC_API_BASE_URL` is exported in your shell environment.

### Step 5: Using the Application

1. **Privacy Acknowledgment**: Open <http://localhost:3000> and click **«متوجه شدم و موافقم»** on the privacy notice modal.

2. **Create a Conversation**: Click **«گفتگوی تازه»** in the sidebar.

3. **Upload Documents**: Open the **«اسناد»** drawer from the top bar (or use the paperclip button in the chat box) and upload `.pdf`, `.txt`, or `.md` files. Wait a few seconds for the document status badge to change from **«در حال پردازش»** (`processing`) to **«آماده»** (`indexed` or `partial_warning`).

4. **Ask Questions**: Submit your question in Persian. Expand the **«منابع»** cards below the answer to inspect the source filename, physical PDF page number, section breadcrumb, and exact document quote.

## REST API Reference

All routes are served under `/api/v1`.

| HTTP Method | Endpoint | Response Code | Description | 
| ----- | ----- | ----- | ----- | 
| `GET` | `/api/v1/health` | `200 OK` | Process health check (`{"status": "ok", "version": "2.0.0"}`). Does not depend on Qdrant connectivity. | 
| `POST` | `/api/v1/conversations` | `201 Created` | Create a new conversation (`{"title": "Optional title"}`). Defaults to `"گفتگوی جدید"`. | 
| `GET` | `/api/v1/conversations` | `200 OK` | List up to 200 active conversations ordered by `updated_at DESC`. | 
| `GET` | `/api/v1/conversations/{id}` | `200 OK` | Fetch single active conversation metadata including `document_count` and `message_count`. | 
| `PATCH` | `/api/v1/conversations/{id}` | `200 OK` | Rename a conversation (`{"title": "New Title"}`, 1–120 chars). | 
| `DELETE` | `/api/v1/conversations/{id}` | `200 OK` | Delete a conversation along with all Qdrant points, uploaded files, chunks, and messages. | 
| `POST` | `/api/v1/conversations/{id}/documents` | `202 Accepted` | Upload one or more documents via `multipart/form-data` field `files`. Returns `duplicate` flag and `document.id`. | 
| `GET` | `/api/v1/conversations/{id}/documents` | `200 OK` | List all non-deleting documents in the conversation. | 
| `GET` | `/api/v1/conversations/{id}/documents/{document_id}` | `200 OK` | Poll document status (`processing`, `indexed`, `partial_warning`, `failed`), `chunk_count`, and error details. | 
| `DELETE` | `/api/v1/conversations/{id}/documents/{document_id}` | `200 OK` | Delete a single document from Qdrant, disk, `chunks_fts`, `chunks`, and `documents`. | 
| `POST` | `/api/v1/conversations/{id}/messages` | `200 OK` | Ask a question (`{"question": "..."}`, max 1000 chars). Append `?trace=true` to include the full diagnostic trace object. | 
| `GET` | `/api/v1/conversations/{id}/messages` | `200 OK` | List chronological conversation history (`user` and `assistant` messages). | 

### Stable API Error Codes (`src/document_qa/api/errors.py`)

| Error Code | HTTP Status | Persian User Message | Cause | 
| ----- | ----- | ----- | ----- | 
| `conversation_not_found` | `404` | `این گفتگو پیدا نشد.` | Conversation does not exist or is in `deleting` state. | 
| `document_not_found` | `404` | `این سند در این گفتگو پیدا نشد.` | Document ID not found in the conversation. | 
| `ingest_in_progress` | `409` | `هنوز سندی برای جست‌وجو آماده نیست.` | No documents are `indexed`/`partial_warning` yet, and at least one is still `processing`. | 
| `file_too_large` | `413` | `حجم فایل از حد مجاز بیشتر است.` | Uploaded file exceeds `MAX_FILE_MB` (80 MB). | 
| `validation_error` | `422` | `درخواست معتبر نیست.` | FastAPI request schema validation failed. | 
| `title_invalid` | `422` | `عنوان باید بین ۱ تا ۱۲۰ نویسه باشد.` | Conversation rename title is empty or $> 120$ characters. | 
| `unsupported_file_type` | `422` | `فقط فایل PDF، TXT و Markdown پشتیبانی می‌شود.` | File extension is not `.pdf`, `.txt`, or `.md`. | 
| `invalid_document` | `422` | `فایل قابل‌خواندن نیست یا قالب آن معتبر نیست.` | Missing `%PDF-` header, corrupt PDF, or invalid UTF-8 text file. | 
| `too_many_documents` | `422` | `تعداد سندهای این گفتگو از حد مجاز بیشتر است.` | Active documents in conversation reached `MAX_DOCUMENTS_PER_CONVERSATION` (50). | 
| `empty_question` | `422` | `سؤال نمی‌تواند خالی باشد.` | Question string is empty after trimming whitespace. | 
| `question_too_long` | `422` | `سؤال از حد مجاز بلندتر است.` | Question length exceeds `MAX_QUESTION_CHARS` (1000). | 
| `llm_rate_limited` | `429` | `سهمیه یا محدودیت درخواست مدل زبانی پر شده است. بعداً دوباره تلاش کنید.` | Every configured service in `llm_services.json` is currently in rate-limit cooldown. | 
| `llm_unavailable` | `503` | `سرویس مدل زبانی در دسترس نیست.` / `مهلت پاسخ به این سؤال تمام شد.` | LLM network/server failure or `QUESTION_DEADLINE_SECONDS` (150s) exceeded. | 
| `llm_output_invalid` | `503` | `مدل پاسخ قابل‌پردازش برنگرداند؛ دوباره تلاش کنید.` | LLM returned malformed JSON or an invalid `status` value. | 
| `retrieval_unavailable` | `503` | `جست‌وجوی برداری در دسترس نیست.` | Qdrant connection or query failure during dense retrieval. | 
| `cleanup_failed` | `503` | `حذف کامل انجام نشد. دوباره تلاش کنید.` | Exception occurred while deleting vectors, files, or SQLite rows (`cleanup_pending = 1`). | 
| `internal_error` | `500` | `خطای پیش‌بینی‌نشده رخ داد.` | Uncaught exception handled by `global_exception_handler`. | 

## Environment Variables Reference

Values are read from `.env` by `src/document_qa/core/config.py`. Note that `.env.example` configures higher retrieval depth (`RETRIEVAL_DENSE_TOP_K=100`, `RETRIEVAL_LEXICAL_TOP_K=100`, `EVIDENCE_MAX=25`, `LLM_MAX_OUTPUT_TOKENS=2048`) than the fallback defaults in `config.py`. Retaining the `.env.example` retrieval values is recommended.

| Variable | `.env.example` Value | `config.py` Fallback | Purpose | 
| ----- | ----- | ----- | ----- | 
| `DATABASE_PATH` | `/app/data/app.db` | `./data/app.db` | SQLite database file path. | 
| `UPLOAD_DIR` | `/app/data/uploads` | `./data/uploads` | Directory for storing uploaded raw files. | 
| `QDRANT_URL` | `http://qdrant:6333` | `http://localhost:6333` | Qdrant HTTP endpoint. | 
| `QDRANT_COLLECTION` | `document_chunks` | `document_chunks` | Qdrant vector collection name. | 
| `EMBEDDING_MODEL_DIR` | `/app/models/paraphrase-...` | `./models/paraphrase-...` | Path containing `model_optimized.onnx` and `tokenizer.json`. | 
| `EMBEDDING_MODEL_ID` | `Qdrant/paraphrase-...` | `Qdrant/paraphrase-...` | Model identifier string. | 
| `EMBEDDING_DIM` | `384` | `384` | Embedding vector dimension (must stay `384`). | 
| `EMBEDDING_MAX_TOKENS` | `128` | `128` | Maximum tokens per embedding input (must stay `128`). | 
| `EMBEDDING_BATCH_SIZE` | `32` | `32` | Batch size parameter for embeddings. | 
| `EMBEDDING_THREADS` | `2` | `2` | CPU intra-op threads for ONNX Runtime session. | 
| `CHUNK_TARGET_TOKENS` | `90` | `90` | Target token count for chunk body. | 
| `CHUNK_MAX_TOKENS` | `100` | `100` | Hard cap on chunk body tokens (never set above `100`). | 
| `CHUNK_OVERLAP_TOKENS` | `16` | `16` | Overlap tokens carried between consecutive chunks. | 
| `BREADCRUMB_MAX_TOKENS` | `20` | `20` | Maximum tokens in structural section breadcrumbs. | 
| `LOW_TEXT_PAGE_CHARS` | `40` | `40` | Minimum characters required for a page to be indexed. | 
| `RETRIEVAL_DENSE_TOP_K` | `100` | `40` | Dense candidates retrieved from Qdrant per sub-query. | 
| `RETRIEVAL_LEXICAL_TOP_K` | `100` | `40` | Lexical candidates retrieved from FTS5 per sub-query. | 
| `RRF_K` | `60` | `60` | Smoothing constant $k$ for Reciprocal Rank Fusion. | 
| `EVIDENCE_MAX` | `25` | `15` | Top reranked chunks passed to the LLM answer generator. | 
| `LEXICAL_SUPPORT_MIN_RATIO` | `0.2` | `0.2` | Minimum token overlap ratio required for claim validation. | 
| `CLAIM_MAX_CHARS` | `1500` | `1500` | Maximum characters allowed in a single validated claim. | 
| `MISSING_MAX_CHARS` | `160` | `160` | Maximum characters allowed in a missing sub-question entry. | 
| `QUOTE_MAX_CHARS` | `600` | `600` | Maximum characters in a citation card quote before truncation. | 
| `MAX_FILE_MB` | `80` | `80` | Maximum upload size per file in MB. | 
| `MAX_PDF_PAGES` | `2000` | `2000` | Maximum page count per PDF file. | 
| `MAX_DOCUMENTS_PER_CONVERSATION` | `50` | `50` | Maximum non-deleting documents per conversation. | 
| `MAX_CHUNKS_PER_CONVERSATION` | `100000` | `100000` | Maximum chunks cap per conversation. | 
| `MAX_QUESTION_CHARS` | `1000` | `1000` | Maximum length of a user question in characters. | 
| `INGEST_TIMEOUT_SECONDS` | `900` | `900` | Timeout setting for document ingestion. | 
| `QUESTION_DEADLINE_SECONDS` | `150` | `150` | Total wall-clock deadline for answering a question. | 
| `ALLOWED_HOSTS` | `localhost:8000,127.0.0.1:8000` | `localhost:8000,127.0.0.1:8000` | Allowed host headers. | 
| `CORS_ORIGINS` | `http://localhost:3000,...` | `http://localhost:3000,...` | Comma-separated allowed CORS origins. | 
| `LOG_LEVEL` | `INFO` | `INFO` | Console logger verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). | 
| `LLM_DATA_POLICY_ACK` | `false` | `false` | **Must be `true`** to enable LLM calls. | 
| `LLM_TIMEOUT_SECONDS` | `60` | `60` | Timeout in seconds per individual LLM call. | 
| `LLM_MAX_OUTPUT_TOKENS` | `2048` | `1024` | Max completion tokens requested from the LLM. | 
| `LLM_BASE_URL` | `https://api.openai.com/v1` | `None` | Fallback LLM base URL (used only if `llm_services.json` is missing). | 
| `LLM_API_KEY` | `sk-your-api-key-here` | `None` | Fallback LLM API key (used only if `llm_services.json` is missing). | 
| `LLM_MODEL` | `gpt-4o-mini` | `None` | Fallback LLM model name (used only if `llm_services.json` is missing). | 
| `LLM_JSON_MODE` | `auto` | `auto` | JSON mode flag (`auto`, `on`, `off`). | 
| `LLM_EXTRA_BODY` | `{}` | `{}` | Extra body JSON string. | 
| `RUN_LIVE_LLM` | `0` | `0` | Test flag for live LLM runs. | 
| `NEXT_PUBLIC_API_BASE_URL` | `http://localhost:8000/api/v1` | `http://localhost:8000/api/v1` | Browser-facing API base URL for the Next.js frontend. | 

## Testing, Linting & Debugging

### Default Offline Test Suite

By default, `pyproject.toml` configures Pytest with `-m 'not integration and not model and not live'`. This allows running the core unit test suite completely offline in under a second:

```
# Run offline unit tests (normalizer, extractor, cleaner, chunker, FTS query, fake embedding, validator)
uv run pytest

# Check Python code style and formatting with Ruff
uv run ruff check .
uv run ruff format --check .

# Lint the Next.js TypeScript frontend
npm --prefix web run lint

```

### Opt-In Test Markers (`model`, `integration`, `live`)

When the ONNX model files, Qdrant container, or live LLM API keys are available, you can run the hardware/network-dependent tests explicitly:

```
# Requires local ONNX model in models/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q/
uv run pytest -m model

# Requires running Qdrant instance (e.g., docker compose up -d qdrant)
uv run pytest -m integration

# Requires running backend, sample PDF, and live LLM credentials
uv run pytest -m live

```

### Observability & Trace Debugging

* **Application Logs (`data/logs/backend.log`)**: Configured via `RotatingFileHandler` (10 MB max per file, 5 backups). To protect document confidentiality, full document text is never written to application logs.

* **Latest QA Execution Trace (`data/latest_trace.json`)**: Every question execution writes its full diagnostic state—including expanded sub-queries, dense/lexical hit counts, RRF candidates, LLM reranker scores, injected math override blocks, raw LLM JSON output, and per-claim validation/rejection reasons—to `data/latest_trace.json`. You can also inspect this trace over HTTP by passing `?trace=true` to `POST /api/v1/conversations/{id}/messages`.

## Known Limitations & Security Notes

* **Digital Text PDFs Only (No OCR)**: PDFs must contain an extractable text layer. Scanned image PDFs, `.docx` files, and standalone images are not supported.

* **Complex Formatting**: Multi-column layouts, complex tables, handwritten notes, and text inside vector graphics may not extract in strict reading order.

* **Physical Page Numbers**: Citation cards display the 1-indexed physical page number of the PDF (`page_idx + 1`), which can differ from printed page numbers when a document includes unnumbered cover pages or Roman-numeral front matter. `.txt` and `.md` files display `"فایل متنی"` (`page_start = null`).

* **Single-Process API Requirement**: Because ingestion uses in-process background tasks and SQLite WAL, always run a single Uvicorn process (`1` worker). Running multiple workers will cause inconsistent ingestion state.

* **Local Network Binding**: No user authentication layer is built into the app. Docker Compose binds ports `3000`, `8000`, and `6333` to `127.0.0.1` (loopback). Do not expose these ports directly to the public internet.

* **External LLM Data Transmission**: While embeddings, SQLite storage, and Qdrant vector search execute 100% locally on your machine, retrieved evidence chunks (`E1`–`En`) and questions are sent to the external LLM provider configured in `llm_services.json`. Deleting a document or conversation locally removes all local files, database rows, and vectors, but cannot recall prompts previously processed by an external LLM provider.