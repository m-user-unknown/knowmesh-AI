# knowmesh-AI

A personal knowledge base and RAG app that turns your local documents into searchable, source-cited answers using vector embeddings and a Groq-powered chat model.

## What this project does

- Scans documents from `Data/Docs` for PDFs and Markdown files
- Splits them into chunks and embeds them with `sentence-transformers/all-MiniLM-L6-v2`
- Stores the chunks in a local Chroma vector database at `chroma_db/`
- Lets you ask questions in a Streamlit UI and retrieves the most relevant source passages
- Returns answers grounded in the retrieved context with filenames cited as sources

This is a lightweight "second brain" prototype for personal knowledge management and document Q&A.

## Project structure

- `app.py` — Streamlit app for chat-based retrieval and answer generation
- `ingest.py` — document ingestion pipeline that loads docs, chunks them, and indexes them into Chroma
- `Data/Docs/` — place your PDFs and Markdown files here
- `chroma_db/` — persisted vector database created by the ingestion step
- `requirements.txt` — Python dependencies
- `LICENSE` — MIT license

## Prerequisites

- Python 3.10+
- pip
- A Groq API key

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables in a `.env` file:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Ingest your documents

Place your PDFs and Markdown files in `Data/Docs`, then run:

```bash
python ingest.py
```

This will create or update the local Chroma database under `chroma_db/`.

## Run the app

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit and ask questions about your documents.

## Notes

- The app is optimized for local personal use and stores the vector database on disk.
- Embeddings are generated locally via Hugging Face models, while the LLM calls are performed through Groq.
- Answers are intentionally grounded in retrieved context; when information is not found, the app says it could not find it in your documents.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
