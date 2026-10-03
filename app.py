# Groq LLM + local embeddings
# import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

CHORMA_DIR = "chroma_db"
COLLECTION_NAME = "knowmesh"

st.set_page_config(page_title="knowmesh")
st.title("🧠 Personal Second Brain (Groq + Local Embeddings)")

@st.cache_resource
def local_vectorstore():
  embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
  return Chroma(
    persist_directory=CHORMA_DIR,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME
  )

vectorstore = local_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k":4})

llm = ChatGroq(
  model="llama-3.3-70b-versatile",
  temperature=0
)

prompt = ChatPromptTemplate.from_tamplate(
  """
You are a helpful assistant that answers questions based only on the provided context from the user's personal knowledge base.

Context:
{context}

Question: {question}

Instructions:
- Answer clearly and concisely.
- Always cite the source using the filename.
- If the answer is not in the context, say "I couldn't find this information in your documents."

Answer:
"""
)

def format_docs(docs):
  return "\n\n".join(
        f"[Source: {doc.metadata.get('filename', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )
