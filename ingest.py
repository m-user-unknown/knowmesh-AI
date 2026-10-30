import os
import pickle
import uuid
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DOCS_DIR = "Data/Docs"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "knowmesh"
PARENTS_PATH = "parent_store.pkl"


def load_documents():
    docs = []

    try:
        pdf_loader = DirectoryLoader(
            DOCS_DIR,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
        )
        docs.extend(pdf_loader.load())
    except Exception as e:
        print(f"PDF load warning: {e}")

    try:
        md_loader = DirectoryLoader(
            DOCS_DIR,
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader,
            show_progress=True,
        )
        docs.extend(md_loader.load())
    except Exception as e:
        print(f"Markdown load warning: {e}")

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        doc.metadata["filename"] = os.path.basename(source)

    print(f"Loaded {len(docs)} documents")
    return docs


def create_parent_child_chunks(documents):
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=200,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    child_chunks = []
    parent_store = {}

    for doc in documents:
        parents = parent_splitter.split_documents([doc])

        for parent in parents:
            parent_id = str(uuid.uuid4())
            parent.metadata["parent_id"] = parent_id
            parent.metadata["chunk_type"] = "parent"
            parent.metadata["filename"] = doc.metadata.get("filename", "unknown")
            parent_store[parent_id] = parent

            children = child_splitter.split_documents([parent])
            for child in children:
                child.metadata["parent_id"] = parent_id
                child.metadata["chunk_type"] = "child"
                child.metadata["filename"] = doc.metadata.get("filename", "unknown")
                child_chunks.append(child)

    print(f"Created {len(parent_store)} parents and {len(child_chunks)} children")
    return child_chunks, parent_store


def main():
    documents = load_documents()
    if not documents:
        print("No documents found in data/docs")
        return

    child_chunks, parent_store = create_parent_child_chunks(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    Chroma.from_documents(
        documents=child_chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )

    with open(PARENTS_PATH, "wb") as f:
        pickle.dump(parent_store, f)

    print(f"Stored {len(child_chunks)} child chunks in Chroma")
    print(f"Saved {len(parent_store)} parents to {PARENTS_PATH}")
    print("Ingestion complete.")


if __name__ == "__main__":
    main()