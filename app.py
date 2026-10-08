import os
from typing import TypedDict, List, Literal
from dotenv import load_dotenv

import streamlit as st
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "knowmesh"
MODEL_NAME = "openai/gpt-oss-20b"   # Working free-tier model

class AgentState(TypedDict):
    question: str
    rewritten_question: str
    documents: List[Document]
    relevant_docs: List[Document]
    generation: str


@st.cache_resource
def get_llm():
    return ChatGroq(model=MODEL_NAME, temperature=0)


@st.cache_resource
def get_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    return vectorstore.as_retriever(search_kwargs={"k": 6})


llm = get_llm()
retriever = get_retriever()


def rewrite_query(state: AgentState):
    """Improve the user question for better retrieval"""
    question = state["question"]

    prompt = ChatPromptTemplate.from_template(
        """You are a query rewriting expert.
Rewrite the following question to make it clearer and better for semantic search.
Keep the meaning the same. Return only the rewritten question.

Question: {question}

Rewritten question:"""
    )

    chain = prompt | llm | StrOutputParser()
    rewritten = chain.invoke({"question": question})

    return {"rewritten_question": rewritten.strip()}


def retrieve_documents(state: AgentState):
    """Retrieve documents using the rewritten question"""
    query = state.get("rewritten_question") or state["question"]
    docs = retriever.invoke(query)
    return {"documents": docs}


def grade_documents(state: AgentState):
    """Grade each document for relevance"""
    question = state["question"]
    documents = state["documents"]

    grade_prompt = ChatPromptTemplate.from_template(
        """You are a grader assessing relevance of a retrieved document to a user question.

Retrieved document:
{document}

User question: {question}

If the document contains information relevant to the question, reply with "yes".
Otherwise reply with "no".

Answer with only "yes" or "no":"""
    )

    chain = grade_prompt | llm | StrOutputParser()

    relevant_docs = []
    for doc in documents:
        score = chain.invoke({
            "document": doc.page_content,
            "question": question
        }).strip().lower()

        if "yes" in score:
            relevant_docs.append(doc)

    return {"relevant_docs": relevant_docs}


def generate_answer(state: AgentState):
    """Generate final answer from relevant documents"""
    question = state["question"]
    docs = state["relevant_docs"]

    if not docs:
        return {
            "generation": "I couldn't find relevant information in your documents to answer this question."
        }

    context = "\n\n".join(
        f"[Source: {doc.metadata.get('filename', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant answering questions based only on the provided context from the user's personal knowledge base.

Context:
{context}

Question: {question}

Instructions:
- Answer clearly and concisely.
- Always cite the source filename.
- If the answer is not in the context, say you couldn't find it.

Answer:"""
    )

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    return {"generation": answer}

def should_generate(state: AgentState) -> Literal["generate", "no_docs"]:
    if len(state.get("relevant_docs", [])) > 0:
        return "generate"
    return "no_docs"


def no_relevant_docs(state: AgentState):
    return {
        "generation": "I searched your documents but couldn't find relevant information for this question."
    }

def build_graph():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("rewrite", rewrite_query)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("grade", grade_documents)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("no_docs", no_relevant_docs)

    # Entry point
    workflow.set_entry_point("rewrite")

    # Edges
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "grade")

    # Conditional after grading
    workflow.add_conditional_edges(
        "grade",
        should_generate,
        {
            "generate": "generate",
            "no_docs": "no_docs"
        }
    )

    workflow.add_edge("generate", END)
    workflow.add_edge("no_docs", END)

    return workflow.compile()


app = build_graph()
