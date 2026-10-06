import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "knowmesh"

st.set_page_config(page_title="Second Brain MVP", page_icon="🖥️")
st.title("🖥️ Personal Second Brain (Groq + Local Embeddings)")

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Groq LLM (very fast + free tier)
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
    # other params...
)

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions based only on the provided context from the user's personal knowledge base.

Context:
{context}

Question: {question}

Instructions:
- Answer clearly and concisely.
- Always cite the source using the filename.
- If the answer is not in the context, say "I couldn't find this information in your documents."

Answer:
""")

def format_docs(docs):
    return "\n\n".join(
        f"[Source: {doc.metadata.get('filename', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask something about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            docs = retriever.invoke(question)
            answer = rag_chain.invoke(question)
            
            st.markdown(answer)
            
            with st.expander("Sources"):
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"**{i}. {doc.metadata.get('filename', 'unknown')}**")
                    st.caption(doc.page_content[:300] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})