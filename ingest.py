import os
from dotenv import loar_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

loar_dotenv()

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
  markdown_loader = DirectoryLoader(
    DOCS_DIR,
    glob="**/*.md",
    loader_cls=UnstructuredMarkdownLoader,
    show_progress=True
  )

  docs = pdf_loader.load() + markdown_loader.load()
  print(f"loaded : {len(docs)} Document")
  return docs

def main():
  document = load_document

  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
    length_funtion = len,
  )

  chunks = text_splitter.split_documents(document)

  print(f"Split into {len(chunks)} chunks")

  for chunk in chunks:
    # in here add metadata to better citations
    source = chunk.metadata.get("source","unkown")
    chunk.metadata["filename"] = os.path.basename(source)

  embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

  # store chunks in vectore db
  vectorStore = Chroma.from_documents(
    document=chunks,
    embeddings=embeddings,
    persist_directory=CHROMA_DIR,
    collection_name=COLLECTION_NAME
  )

  print(f"successfully store {len(chunks)} chunks in chroma at '{CHROMA_DIR}'")

if __name__ == "__main__":
  main()