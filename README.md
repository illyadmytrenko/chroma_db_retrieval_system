# Hybrid Document & Code Search (ChromaDB + BM25 + Streamlit)

An interactive Streamlit application for indexing and searching `.py`, `.md`, `.txt`, `.docx`, and `.pdf` files using **hybrid search**—combining semantic dense retrieval (**ChromaDB** + `all-mpnet-base-v2`) with lexical sparse retrieval (**BM25** via `bm25s`) using **Reciprocal Rank Fusion (RRF)**, complete with metadata filtering, dual data modes, and structure-aware chunking.

## Key Features

- **Dual Storage Modes (Demo & Private Session):**
  - **Demo Database (Read-Only):** Connects to a preloaded `PersistentClient` (`./demo_chroma_db`) so visitors can immediately test hybrid search and metadata filtering out of the box without modifying shared data.
  - **User Session Database (In-Memory):** Uses `chromadb.EphemeralClient` stored in `st.session_state` to give every user an isolated, private workspace for uploading, searching, and deleting their own documents.
- **Multi-Format Parsing & Structure-Aware Chunking (`smart_chunker`):** Extracts text from `.pdf` (`pypdf`), `.docx` (`python-docx`), and UTF-8 code/text files. Splits `.py` files on top-level `def`, `async def`, and `class` boundaries, `.md` files on Markdown headings (`#`–`###`), normalizes PDF line breaks/hyphenation, and preserves context with `overlap`.
- **Hybrid Search with RRF (`get_rrf_top_n`):** Combines dense vector search distances from ChromaDB with sparse keyword rankings from `bm25s` using Reciprocal Rank Fusion (k = 60 by default), alongside standalone Vector and BM25 search modes.
- **Metadata Filtering:** Filter search results dynamically by metadata fields (`$in` for categorical attributes such as `source`, `directory`, and `extension`) and upload recency (`uploaded_at` via `$gte` timestamp cutoff).
- **Batch Ingestion & Collection Management:** Upload directories or individual files in Streamlit with extension validation, batch `upsert` writes (`BATCH_SIZE = 500`), collection inspection table, selective file deletion by `source`, and collection reset.
- **Resource Caching:** Uses `@st.cache_resource` to share the embedding model, preloaded demo collection, and BM25 retriever instances across runs without redundant reloads.

## Tech Stack

- **UI:** Streamlit
- **Vector Store & Embeddings:** ChromaDB (`PersistentClient` & `EphemeralClient`), `SentenceTransformerEmbeddingFunction` (`all-mpnet-base-v2`)
- **Lexical Search:** `bm25s`
- **Document Parsing:** `pypdf`, `python-docx`
- **Numerical Ranking:** `numpy`

## Project Structure

```text
.
├── app.py              # Main Streamlit application & UI
├── text.py             # File text extraction (.pdf, .docx, text) & smart_chunker
├── db.py               # ChromaDB clients (Demo Persistent & Session Ephemeral) + caching
├── search.py           # BM25S retrieval & NumPy RRF ranking logic
├── demo_chroma_db/     # Preloaded read-only demo database
└── README.md           # Project documentation
```

## Installation

1. Clone the repository and navigate to the project folder:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. Install dependencies:
   ```bash
   pip install streamlit chromadb sentence-transformers bm25s numpy python-docx pypdf
   ```

## Usage

Run the Streamlit app from the project directory:
```bash
streamlit run app.py
```

In the web UI:
1. **Select a Data Source (Sidebar):** Choose between the preloaded **Demo database (read-only)** or **My files (temporary session)** for private uploads.
2. **Add files:** *(Available in User Session mode)* Upload `.py`, `.md`, `.txt`, `.docx`, or `.pdf` files to chunk and upsert into your in-memory collection.
3. **Query collection:** Enter a search query, pick a search mode (`Hybrid RRF`, `Vector`, or `BM25`), set `Max results`, optionally apply metadata or upload-date filters, and inspect ranked cards with expandable full chunk text.
4. **Show files:** View indexed chunks and metadata in a table.
5. **Delete files / Delete whole collection:** *(Available in User Session mode)* Remove chunks by source file or reset your session collection.

## Notes & Limitations

- Only `.py`, `.md`, `.txt`, `.docx`, and `.pdf` (with an extractable text layer; scanned image-only PDFs are skipped) are indexed.
- The preloaded demo database (`./demo_chroma_db`) is locked in read-only mode in the UI to prevent visitors from overwriting shared demo data, while user uploads live strictly in RAM (`EphemeralClient`) for the duration of the browser session.
- This application performs retrieval and ranking over indexed chunks; it does not yet include an LLM answer-generation step.
