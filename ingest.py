import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchian_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchian_groq import ChatGroq
from langchain_chroma import Chroma

load_dotenv()

DOCS_DIR = "Data/Docs"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "knowmesh"

def load_document():
  # load pdfs
  pdf_loader = DirectoryLoader(
    DOCS_DIR,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True
  )

  # load markdown
  md_loader = DirectoryLoader(
    DOCS_DIR,
    glob="**/*.md",
    loader_cls=UnstructuredMarkdownLoader,
    show_progress=True
  )

  docs = pdf_loader.load() + md_loader.load()
  print(f"loaded : {len(docs)} Document")
  return docs

def main():
  document = load_document()

  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len,
  )

  chunks = text_splitter.split_documents(document)

  print(f"Split into {len(chunks)} chunks")

  for chunk in chunks:
    # in here add metadata to better citations
    source = chunk.metadata.get("source","unkown")
    chunk.metadata["filename"] = os.path.basename(source)

  # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
  embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
  # store chunks in vectore db
  vectorStore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,
    collection_name=COLLECTION_NAME
  )

  print(f"successfully store {len(chunks)} chunks in chroma at '{CHROMA_DIR}'")

if __name__ == "__main__":
  main()