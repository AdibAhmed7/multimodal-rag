# 🌌 Multimodal RAG System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests: Pytest](https://img.shields.io/badge/tests-pytest-green.svg)](tests/)
[![Architecture: Modular](https://img.shields.io/badge/architecture-modular-orange.svg)](#architecture-overview)

A modular, production-grade **Multimodal Retrieval-Augmented Generation (RAG)** system designed to ingest, parse, semantically chunk, index, and query complex enterprise documents containing dense text, diagrams, figures, and embedded images.

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Architecture Overview](#-architecture-overview)
- [Pipeline Phases](#-pipeline-phases)
  - [Phase 1: PDF Document Ingestion](#phase-1-pdf-document-ingestion)
  - [Phase 2: Multimodal Chunking & Semantic Linking](#phase-2-multimodal-chunking--semantic-linking)
  - [Roadmap & Future Phases](#roadmap--future-phases)
- [Directory Layout](#-directory-layout)
- [Installation & Setup](#-installation--setup)
- [Quick Start Guide](#-quick-start-guide)
- [Data & Output Schemas](#-data--output-schemas)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Configuration](#-configuration)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

Traditional RAG pipelines discard visual context, converting PDFs purely into flat strings and losing diagrams, system flowcharts, formulas, and charts. 

This **Multimodal RAG** platform preserves the rich spatial and semantic connection between textual narrative and visual figures:
- **Zero-loss Extraction**: Extracts text per page alongside embedded raster images with exact metadata and coordinate associations.
- **Context-Preserving Chunking**: Splits text along natural sentence and paragraph boundaries while maintaining bidirectional references between text chunks and co-occurring figures.
- **Modular Extensibility**: Decoupled ingestion, chunking, embedding, vector store, and generation engines ready for local or cloud deployment.

---

## 📐 Architecture Overview

```
                          ┌────────────────────────┐
                          │   Input PDF Document   │
                          └───────────┬────────────┘
                                      │
                                      ▼
                      ┌────────────────────────────────┐
                      │    Phase 1: Ingestion Engine   │
                      │  (PDFLoader & Context Manager) │
                      └───────────────┬────────────────┘
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
        ┌─────────────────────┐               ┌─────────────────────┐
        │    TextExtractor    │               │    ImageExtractor   │
        │ (1-based page text) │               │ (Raster resolution) │
        └──────────┬──────────┘               └──────────┬──────────┘
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      ▼
                      ┌────────────────────────────────┐
                      │    PageParser & MetadataBuilder│
                      │  data/processed/metadata/      │
                      └───────────────┬────────────────┘
                                      │
                                      ▼
                      ┌────────────────────────────────┐
                      │   Phase 2: Semantic Chunking   │
                      └───────────────┬────────────────┘
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
        ┌─────────────────────┐               ┌─────────────────────┐
        │     TextChunker     │               │     ImageChunker    │
        │ (Boundary-aware)    │               │ (Page Context Link) │
        └──────────┬──────────┘               └──────────┬──────────┘
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      ▼
                      ┌────────────────────────────────┐
                      │    Bidirectional Linker        │
                      │  data/processed/chunks/        │
                      │  • document_chunks.json        │
                      │  • text_chunks.json            │
                      │  • image_records.json          │
                      └────────────────────────────────┘
```

---

## 🚀 Pipeline Phases

### Phase 1: PDF Document Ingestion
Extracts text, embedded raster images, page boundaries, and document metadata from raw PDF files.
- [`PDFLoader`](src/ingestion/pdf_loader.py): Safe context management (`with PDFLoader(...) as pdf:`) with boundary validation.
- [`TextExtractor`](src/ingestion/text_extractor.py): Extracts high-fidelity text per page with 1-based indexing.
- [`ImageExtractor`](src/ingestion/image_extractor.py): Extracts raster images from PDF streams, persists them cleanly formatted (`page_XXX_img_YYY.ext`), and captures dimensions.
- [`PageParser`](src/ingestion/page_parser.py): Combines per-page text and images into a single structured page model.
- [`MetadataBuilder`](src/ingestion/metadata_builder.py): Generates structured JSON representations stored in `data/processed/metadata/{doc_id}.json`.

### Phase 2: Multimodal Chunking & Semantic Linking
Transforms raw ingested pages into searchable, linked chunks.
- [`TextChunker`](src/chunking/text_chunker.py): Configurable chunk size (e.g., 1000 chars) and overlap (e.g., 150 chars) that snaps to natural sentence/paragraph boundaries.
- [`ImageChunker`](src/chunking/image_chunker.py): Extracts image descriptors, retains surrounding textual context, and links images to text chunks on the same page.
- [`SemanticChunker`](src/chunking/semantic_chunker.py): Orchestrates unified chunking with bidirectional reference graphs (`chunk.related_image_ids` and `image.related_text_chunk_ids`).
- [`Chunk Models`](src/chunking/chunk_models.py): Strictly typed Pydantic V2 models (`TextChunk`, `ImageRecord`).

### Phase 3: Text & Image Embeddings
Converts chunks and visual figures into dense mathematical vectors for semantic search.
- [`TextEmbeddingModel`](src/embeddings/text_embeddings.py): Encodes text chunks into dense 1024-dimensional normalized vectors using `BAAI/bge-m3`.
- [`ImageEmbeddingModel`](src/embeddings/image_embeddings.py): Encodes figures and text-to-image queries into a shared 768-dimensional multimodal vector space using `google/siglip2-base-patch16-224`.
- **Decoupled Vector Spaces**: BGE-M3 handles text-to-text queries; SigLIP 2 handles image-to-image and text-to-image queries.
- **Persistent Artifacts**: Compact `.npz` storage (`vectors.npz`) paired with comprehensive manifests (`manifest.json`) mapping row indices to chunk/image IDs.

### Phase 4: Vector Database Setup and Indexing (Qdrant)
Stores dense embeddings and complete metadata in local embedded Qdrant collections.
- [`QdrantManager`](src/vectorstore/qdrant_manager.py): Embedded Qdrant client manager that provisions `text_chunks` (1024D, Cosine) and `document_images` (768D, Cosine).
- [`TextIndexer`](src/vectorstore/index_text.py): Upserts text embeddings alongside full chunk contents, source filenames, page numbers, and bidirectional image references using deterministic UUIDv5 identifiers.
- [`ImageIndexer`](src/vectorstore/index_images.py): Upserts image embeddings alongside image disk paths, dimensions, captions, surrounding text, and linked chunk references using deterministic UUIDv5 identifiers.
- **Persistent Storage**: Local on-disk database stored in `data/processed/qdrant/`.

### Phase 5: Multimodal Retrieval Engine
Retrieves text and visual evidence using decoupled vector spaces fused via Reciprocal Rank Fusion.
- [`QueryEncoder`](src/retrieval/query_encoder.py): Dispatches user queries to BGE-M3 (for text chunks) and SigLIP 2 (for image matching and image-to-image similarity).
- [`TextRetriever`](src/retrieval/text_retriever.py): Searches `text_chunks` with optional metadata filtering (`document_id`).
- [`ImageRetriever`](src/retrieval/image_retriever.py): Performs text-to-image and image-to-image searches on `document_images`.
- [`reciprocal_rank_fusion`](src/retrieval/result_fusion.py): Merges heterogeneous rank positions using RRF ($k=60$) without conflating non-calibrated similarity score distributions.
- [`MultimodalRetrievalPipeline`](src/retrieval/retrieval_pipeline.py): Unified retrieval facade coordinating encoders, retrievers, and rank fusion.

### Phase 6: Context Building & Vision-Based Understanding
Converts raw retrieval hits into an integrated, page-by-page multimodal context package.
- [`ContextOrganizer`](src/context/context_organizer.py): Groups text chunks and images by document and page, sorting by chunk index and preserving bidirectional figure references.
- [`ImageDescriber`](src/context/image_describer.py): Generates technical figure descriptions using `Qwen/Qwen2.5-VL-3B-Instruct`, with persistent JSON caching in `data/processed/image_descriptions/cache.json`.
- [`ContextBuilder`](src/context/context_builder.py): Coordinates page aggregation, selective image description generation, and context statistics.
- [`build_context.py`](scripts/build_context.py): CLI generating structured JSON context packages ready for vision-language answering models.

### Phase 7: LLM-Based Multimodal Answer Generation
Generates grounded multimodal answers using Qwen2.5-VL with page citations and original attached visual figures.
- [`QwenModel`](src/llm/model_loader.py): Vision-language model wrapper loading `Qwen/Qwen2.5-VL-3B-Instruct` with offline and resource resilience.
- [`PromptBuilder`](src/llm/prompt_builder.py): Enforces strict evidence grounding, prevents hallucinations, and mandates `[Page N]` citations.
- [`AnswerGenerator`](src/llm/answer_generator.py): Dispatches text evidence, page contexts, and original labeled raster images to Qwen.
- [`ResponseFormatter`](src/llm/response_formatter.py): Structures answers alongside cited source pages, text chunks, and attached display images.
- [`ask_pdf.py`](scripts/ask_pdf.py): End-to-end RAG CLI executing retrieval, context preparation, answer generation, and response persistence.

### Roadmap & Future Phases
- **Phase 8: User Interface & Evaluation** — Interactive Streamlit chat interface with image viewer, citation pill navigation, and RAGAS benchmark metrics.

---

## 📂 Directory Layout

```plaintext
multimodal-rag/
├── configs/                  # YAML configurations (pipeline, models, prompts)
│   ├── config.yaml
│   ├── models.yaml
│   └── prompts.yaml
├── data/
│   ├── raw/pdfs/             # Source input PDF documents
│   ├── processed/            # Processed artifacts (metadata, images, chunks)
│   └── generated/            # Model outputs and synthetic responses
├── docker/                   # Containerization
│   ├── Dockerfile
│   └── Dockerfile.gpu
├── notebooks/                # Exploratory research notebooks
├── scripts/                  # CLI utilities and entrypoints
│   ├── ingest_pdf.py         # Ingestion script
│   ├── prepare_chunks.py     # Chunk preparation script
│   ├── test_loader.py        # PDF loader test script
│   ├── test_chunking.py      # Chunking test script
│   └── ...
├── src/                      # Core modular source code
│   ├── api/                  # FastAPI endpoints & schemas
│   ├── chunking/             # Semantic text & image chunking
│   ├── embeddings/           # Text & multimodal embedding models
│   ├── evaluation/           # Evaluation metrics & benchmarks
│   ├── generation/           # Prompt & response builders
│   ├── ingestion/            # PDF ingestion & asset extraction
│   ├── llm/                  # Vision-Language models interface
│   ├── pipeline/             # End-to-end pipeline orchestration
│   ├── query/                # Intent classifier & query router
│   ├── retrieval/            # Hybrid retrieval & rerankers
│   ├── ui/                   # Streamlit interactive interface
│   ├── utils/                # Logging, file & image utilities
│   ├── vectorstore/          # Qdrant client & collection schemas
│   └── vision/               # OCR, diagram analyzer & captioning
├── tests/                    # Pytest test suite
│   ├── unit/                 # Unit tests for core components
│   └── integration/          # End-to-end pipeline integration tests
├── .env.example              # Environment variables template
├── .gitignore                # Production gitignore rules
├── docker-compose.yml        # Vector store & services orchestration
├── pyproject.toml            # Project packaging & pytest config
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10+ (Recommended: Python 3.11 or 3.12)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/multimodal-rag.git
cd multimodal-rag
```

### 2. Set Up a Virtual Environment
```bash
# On Linux / macOS
python -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
```bash
cp .env.example .env
# Edit .env with your favorite editor
```

---

## 🚦 Quick Start Guide

### 1. Ingest a PDF Document
Ingest any PDF document to extract text per page and rip out embedded figures:
```bash
python scripts/ingest_pdf.py data/raw/pdfs/example.pdf
```
*Output:*
- Extracted images stored in `data/processed/images/example/`
- Full structured JSON document in `data/processed/metadata/example.json`

### 2. Run Semantic Chunking & Linking
Convert the ingested document into semantic text chunks and bidirectionally linked image records:
```bash
python scripts/prepare_chunks.py data/processed/metadata/example.json
```
*Output:*
- `data/processed/chunks/example/text_chunks.json`
- `data/processed/chunks/example/image_records.json`
- `data/processed/chunks/example/document_chunks.json`

### 3. Generate Multimodal Embeddings
Generate BGE-M3 text vectors and SigLIP 2 image vectors:
```bash
python scripts/generate_embeddings.py data/processed/chunks/example
```
*Output:*
- `data/processed/embeddings/example/vectors.npz` (contains `text_vectors` [1024D] and `image_vectors` [768D])
- `data/processed/embeddings/example/manifest.json` (maps array rows to chunk & image metadata)

### 4. Validate Generated Embeddings
Verify shapes, dimensional consistency, and manifest mappings:
```bash
python scripts/test_embeddings.py data/processed/embeddings/example
```

### 5. Index Vectors into Local Qdrant
Create local collections and store embeddings with metadata and deterministic UUIDv5 points:
```bash
python scripts/index_vectors.py data/processed/embeddings/example
```

### 6. Verify Vectorstore Collections
Inspect indexed collections and sample payloads:
```bash
python scripts/test_vectorstore.py
```

### 7. Run Multimodal Retrieval
Search across text chunks, document images, or both simultaneously:
```bash
# Text-to-text retrieval
python scripts/test_text_retrieval.py --query "Explain the Transformer architecture"

# Text-to-image retrieval
python scripts/test_image_retrieval.py --query "Find diagrams of neural networks"

# Multimodal retrieval with Reciprocal Rank Fusion (RRF)
python scripts/test_multimodal_retrieval.py --query "Explain the Transformer and show its architecture diagram"
```

### 8. Build Structured Multimodal Context
Organize retrieval evidence into page-grouped contexts with figure descriptions:
```bash
# Context building without image description model
python scripts/build_context.py --query "Explain the architecture of the neural network and find its related diagram" --skip-image-descriptions

# Context building with Qwen vision descriptions
python scripts/build_context.py --query "Explain the architecture of the neural network and find its related diagram" --max-images-to-describe 3
```

### 9. Ask Questions with Full Multimodal RAG
Run the end-to-end pipeline to answer questions grounded in text and original PDF images:
```bash
python scripts/ask_pdf.py --query "Explain the architecture of the neural network and show the relevant diagram"
```

### 10. Visual Generation & Multimodal Orchestration (Phase 8)
Run the integrated router and visual generator (supporting image generation with Stable Diffusion Turbo alongside PDF retrieval and Qwen answering):

```bash
# 1. Generate a new concept illustration only
python scripts/ask_multimodal.py --query "Generate a professional illustration of a multimodal RAG architecture"

# 2. Retrieve existing PDF diagrams only
python scripts/ask_multimodal.py --query "Show me the architecture diagram from the PDF"

# 3. Hybrid: Generate new diagram AND retrieve original PDF diagram
python scripts/ask_multimodal.py --query "Generate a new neural network architecture illustration and show me the original diagram from the PDF"

# 4. Standard evidence-grounded answer
python scripts/ask_multimodal.py --query "Explain the architecture of the proposed system"
```

---

## 📊 Data & Output Schemas

### `TextChunk` Schema
```json
{
  "chunk_id": "example_p001_c000",
  "document_id": "example",
  "page_number": 1,
  "chunk_index": 0,
  "content": "This document introduces the Multimodal RAG system architecture...",
  "source_file": "example.pdf",
  "related_image_ids": [
    "example_p001_img001"
  ]
}
```

### `ImageRecord` Schema
```json
{
  "image_id": "example_p001_img001",
  "document_id": "example",
  "page_number": 1,
  "image_index": 1,
  "image_path": "data/processed/images/example/page_001_img_001.png",
  "extension": "png",
  "width": 1280,
  "height": 720,
  "caption": null,
  "surrounding_text": "Figure 1 illustrates the overarching system data flow...",
  "related_text_chunk_ids": [
    "example_p001_c000"
  ]
}
```

---

## 🧪 Testing & Quality Assurance

The codebase includes comprehensive unit tests verifying PDF loading, text extraction, image extraction, boundary-aware chunking, and bidirectional linking:

```bash
pytest -v
```

To run a specific test suite:
```bash
pytest tests/unit/test_chunking.py -v
pytest tests/unit/test_pdf_loader.py -v
```

---

## 🐳 Docker Deployment

To run tests or services within a containerized environment:

```bash
# Build the Docker image
docker build -t multimodal-rag -f docker/Dockerfile .

# Start Qdrant vector database service
docker-compose up -d
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature-name`.
3. Commit your changes: `git commit -m "feat: add feature"`.
4. Push to the branch: `git push origin feature/your-feature-name`.
5. Open a Pull Request.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
