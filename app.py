import streamlit as st
import os
import re
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="🚀", layout="centered")

st.title("Zyro Dynamics HR Help Desk 🤖")
st.caption("Get verified answers instantly from internal company policy documents.")

# API Key Validation Configuration
groq_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
if not groq_key:
    groq_key = st.sidebar.text_input("Enter Groq API Key:", type="password")

if not groq_key:
    st.info("Please set up or input your Groq API Key to begin.")
    st.stop()

# Cache heavy components
@st.cache_resource
def setup_engine():
    # Looks for documents folder inside the deployment directory root
    corpus_folder = "zyro-dynamics-hr-corpus"
    if not os.path.exists(corpus_folder):
        os.makedirs(corpus_folder, exist_ok=True)
        
    loader = PyPDFDirectoryLoader(corpus_folder)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4})

try:
    vectorstore, retriever = setup_engine()
except Exception as e:
    st.error(f"Make sure to place policy PDFs in a folder named 'zyro-dynamics-hr-corpus': {e}")
    st.stop()

llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_key, temperature=0.1)

# Guardrail & Generation Setup
OOS_PROMPT = ChatPromptTemplate.from_template("Classify if the question relates strictly to internal corporate HR policies. Reply ONLY with 'OUT_OF_SCOPE' or 'IN_SCOPE'.\\n\\nQuestion: {question}\\nClassification:")
RAG_PROMPT = ChatPromptTemplate.from_template("Answer using ONLY the provided policy text:\\n\\nContext:\\n{context}\\n\\nQuestion: {question}\\n\\nAnswer:")
REFUSAL = "I can only answer HR-related questions from Zyro Dynamics policy documents."

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask an HR policy question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Check Scope
        check_chain = OOS_PROMPT | llm | StrOutputParser()
        scope = check_chain.invoke({"question": prompt}).strip().upper()
        
        if "OUT_OF_SCOPE" in scope:
            st.markdown(REFUSAL)
            st.session_state.messages.append({"role": "assistant", "content": REFUSAL})
        else:
            # Retrieve Docs explicitly for citations display
            retrieved_chunks = retriever.invoke(prompt)
            context_string = "\\n\\n".join(c.page_content for c in retrieved_chunks)
            
            gen_chain = RAG_PROMPT | llm | StrOutputParser()
            response = gen_chain.invoke({"context": context_string, "question": prompt})
            
            # Format and show final presentation
            st.markdown(response)
            
            # Citations block
            with st.expander("References / Source Citations"):
                sources = set(os.path.basename(c.metadata.get('source', 'Policy Doc')) for c in retrieved_chunks)
                for src in sources:
                    st.write(f"📄 {src}")
            
            st.session_state.messages.append({"role": "assistant", "content": response})