# AI-Powered Code Assistant

Ask questions about a PHP/Laravel project (or another supported source tree) using local semantic retrieval and Google Gemini Flash.

## What it does

- Reads supported source and documentation files from a local project folder.
- Splits files into overlapping chunks and embeds them with Sentence Transformers.
- Uses a FAISS inner-product index over normalized embeddings for semantic search.
- Sends the question and retrieved excerpts to Gemini through LangChain, with source-file citations in the answer.
- Caches the index locally in `.code-assistant-index/` and rebuilds it when indexed files or the embedding model change.

The codebase is indexed locally. Only the question and retrieved excerpts are sent to Google's Gemini API when you ask a question. Review your organization's data policies before using a hosted model with proprietary code.

## Setup

Use Python 3.10 or newer. From this folder, create and activate a virtual environment, then install dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env`, then set `GOOGLE_API_KEY` to a Google AI Studio API key. Optionally adjust `GOOGLE_MODEL`, `CODEBASE_PATH`, or `EMBEDDING_MODEL`.

Start the app:

```powershell
streamlit run app.py
```

Enter the path to your PHP/Laravel project in the sidebar and select **Index / refresh codebase**. Once indexing finishes, ask a question in the main panel. The first run downloads the sentence embedding model; subsequent indexing runs reuse the local cache when the source files have not changed.

## Indexed files

The indexer includes common source, markup, configuration, SQL, and Markdown extensions (including PHP and Blade templates). It skips dependency/build directories such as `vendor`, `node_modules`, `storage`, and `.git`, skips `.env` files and lockfiles, and ignores files larger than 1 MB.

## Run tests

```powershell
python -m unittest discover -s tests
```
