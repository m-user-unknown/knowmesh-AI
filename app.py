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

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="KnowMesh Agent",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Custom CSS: Black, White, Blue
# --------------------------------------------------
st.markdown("""
<style>
    :root {
        --bg: #080B12;
        --panel: #101521;
        --border: #222B3B;
        --text: #F5F7FA;
        --muted: #8D9AAF;
        --blue: #3B82F6;
        --blue-light: #60A5FA;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: #0C1019;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }

    .brand {
        font-size: 1.55rem;
        font-weight: 750;
        letter-spacing: -0.8px;
        color: var(--text);
        margin-bottom: 4px;
    }

    .brand span {
        color: var(--blue-light);
    }

    .brand-caption {
        color: var(--muted);
        font-size: 0.83rem;
        margin-bottom: 30px;
    }

    .hero {
        padding: 36px 0 24px 0;
    }

    .eyebrow {
        color: var(--blue-light);
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .hero h1 {
        color: var(--text);
        font-size: clamp(2rem, 5vw, 2.7rem);
        font-weight: 750;
        letter-spacing: -1.8px;
        line-height: 1.15;
        margin: 0;
    }

    .hero p {
        color: var(--muted);
        font-size: 0.98rem;
        line-height: 1.7;
        margin-top: 12px;
        max-width: 550px;
    }

    .section-label {
        color: var(--muted);
        font-size: 0.75rem;
        font-weight: 650;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin: 20px 0 10px 0;
    }

    .feature {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 15px;
        min-height: 105px;
    }

    .feature-title {
        color: var(--text);
        font-size: 0.91rem;
        font-weight: 650;
        margin-bottom: 7px;
    }

    .feature-description {
        color: var(--muted);
        font-size: 0.78rem;
        line-height: 1.5;
    }

    [data-testid="stChatMessage"] {
        background: transparent;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 15px 17px;
        margin-bottom: 12px;
    }

    [data-testid="stChatMessage"]:has(
        [data-testid="chatAvatarIcon-user"]
    ) {
        background: #111B2B;
        border-color: #233957;
    }

    [data-testid="stChatInput"] {
        border: 1px solid #2A3B56;
        border-radius: 14px;
        background: #101521;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: var(--blue);
        box-shadow: 0 0 0 1px var(--blue);
    }

    .stButton > button {
        border: 1px solid var(--border);
        background: var(--panel);
        color: var(--text);
        border-radius: 9px;
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        border-color: var(--blue);
        color: var(--blue-light);
        background: #111B2B;
    }

    [data-testid="stExpander"] {
        background: #0D121D;
        border: 1px solid var(--border);
        border-radius: 10px;
    }

    .status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border: 1px solid #24436B;
        background: #101E33;
        color: #93C5FD;
        padding: 6px 11px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .status-dot {
        height: 7px;
        width: 7px;
        border-radius: 50%;
        background: #60A5FA;
    }

    .footer {
        color: #64748B;
        font-size: 0.75rem;
        text-align: center;
        padding: 24px 0 12px;
    }

    hr {
        border-color: var(--border);
    }

    #MainMenu, footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="brand">KnowMesh<span>.</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="brand-caption">Your personal knowledge agent</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("### Workspace")
    st.markdown(
        '<div class="status">'
        '<span class="status-dot"></span>'
        'Phase 2 · LangGraph'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")
    if st.button("＋  New conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("**Agent capabilities**")
    st.caption("01 · Query rewriting")
    st.caption("02 · Document retrieval")
    st.caption("03 · Relevance grading")
    st.caption("04 · Context-grounded answers")

    st.markdown("---")
    st.caption("KNOWMESH AI · PHASE 02")


# --------------------------------------------------
# Main Header
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">PERSONAL KNOWLEDGE BASE</div>
    <h1>Think with your<br>documents.</h1>
    <p>
        Ask questions, explore your knowledge, and get answers
        through an intelligent retrieval pipeline powered by LangGraph.
    </p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Session State
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# Empty State / Feature Cards
# --------------------------------------------------
if not st.session_state.messages:
    st.markdown('<div class="section-label">What happens behind the scenes</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="feature">
            <div class="feature-title">01 / Query rewrite</div>
            <div class="feature-description">
                Refines your question to improve retrieval quality.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature">
            <div class="feature-title">02 / Smart retrieval</div>
            <div class="feature-description">
                Finds relevant document chunks from your knowledge base.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("""
        <div class="feature">
            <div class="feature-title">03 / Document grading</div>
            <div class="feature-description">
                Evaluates retrieved content for relevance to your query.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="feature">
            <div class="feature-title">04 / Grounded answers</div>
            <div class="feature-description">
                Generates a response using the selected context.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")


# --------------------------------------------------
# Chat History
# --------------------------------------------------
for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display debug details for assistant responses
        if message["role"] == "assistant" and message.get("debug"):
            with st.expander("Agent execution details"):
                debug = message["debug"]

                st.markdown("**Rewritten question**")
                st.write(debug.get("rewritten_question") or "Not available")

                st.markdown("**Retrieved documents**")
                st.write(debug.get("retrieved_count", 0))

                st.markdown("**Relevant documents**")
                st.write(debug.get("relevant_count", 0))

                sources = debug.get("sources", [])
                if sources:
                    st.markdown("**Sources used**")
                    for source in sources:
                        st.markdown(f"- `{source}`")


# --------------------------------------------------
# Chat Input + LangGraph
# --------------------------------------------------
if question := st.chat_input("Ask something about your documents..."):

    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your knowledge..."):

            try:
                result = app.invoke({
                    "question": question,
                    "rewritten_question": "",
                    "documents": [],
                    "relevant_docs": [],
                    "generation": "",
                })

                answer = result.get(
                    "generation",
                    "I couldn't generate an answer for this question.",
                )

                st.markdown(answer)

                relevant_docs = result.get("relevant_docs", [])

                debug = {
                    "rewritten_question": result.get(
                        "rewritten_question", ""
                    ),
                    "retrieved_count": len(result.get("documents", [])),
                    "relevant_count": len(relevant_docs),
                    "sources": list(dict.fromkeys(
                        doc.metadata.get("filename", "unknown")
                        for doc in relevant_docs
                    )),
                }

                with st.expander("Agent execution details"):
                    st.markdown("**Rewritten question**")
                    st.write(debug["rewritten_question"] or "Not available")

                    st.markdown("**Retrieved documents**")
                    st.write(debug["retrieved_count"])

                    st.markdown("**Relevant documents**")
                    st.write(debug["relevant_count"])

                    if debug["sources"]:
                        st.markdown("**Sources used**")
                        for source in debug["sources"]:
                            st.markdown(f"- `{source}`")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "debug": debug,
                })

            except Exception as error:
                error_message = (
                    "Something went wrong while processing your question. "
                    "Please check the agent configuration and try again."
                )

                st.error(error_message)

                # Detailed errors belong in the server logs.
                print(f"KnowMesh agent error: {error!r}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                })


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("""
<div class="footer">
    KNOWMESH AGENT · BUILT WITH LANGGRAPH & STREAMLIT
</div>
""", unsafe_allow_html=True)