import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
import chromadb
import os
from dotenv import load_dotenv

# Disable file watcher for PyTorch compatibility
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

load_dotenv()

# Constants and Configuration
CHROMA_PATH = "./chroma_db"
IPC_PDF_PATH = "IPC.pdf"  # Path to your IPC PDF
BNS_PDF_PATH = "BNS.pdf"  # Path to your BNS PDF
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# Set up embeddings
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")

# Streamlit UI
st.title("Legal Case Analysis: IPC vs BNS")
st.write("Chat with an AI to find applicable clauses from both the Indian Penal Code and Bharatiya Nyaya Sanhita")

# Input the Groq API Key and initialize LLM
api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=api_key, model_name="deepseek-r1-distill-qwen-32b")

# Chat interface
session_id = st.text_input("Session ID", value="default_session")

# Statefully manage chat history
if 'store' not in st.session_state:
    st.session_state.store = {}

# Initialize vector stores if not already in session state
if 'vector_stores_initialized' not in st.session_state:
    st.session_state.vector_stores_initialized = False

# Check if PDF files exist
if not os.path.exists(IPC_PDF_PATH):
    st.error(f"Indian Penal Code PDF not found at {IPC_PDF_PATH}")
    st.stop()

if not os.path.exists(BNS_PDF_PATH):
    st.error(f"Bharatiya Nyaya Sanhita PDF not found at {BNS_PDF_PATH}")
    st.stop()

# Process the pre-set PDFs if not already done
if not st.session_state.vector_stores_initialized:
    try:
        with st.spinner("Loading legal documents (this may take a moment)..."):
            # Process Indian Penal Code
            ipc_loader = PyPDFLoader(IPC_PDF_PATH)
            ipc_docs = ipc_loader.load()
            
            # Process Bharatiya Nyaya Sanhita
            bns_loader = PyPDFLoader(BNS_PDF_PATH)
            bns_docs = bns_loader.load()
            
            # Tag documents with their source
            for doc in ipc_docs:
                doc.metadata["source"] = "Indian Penal Code"
            
            for doc in bns_docs:
                doc.metadata["source"] = "Bharatiya Nyaya Sanhita"
            
            # Combine all documents
            all_docs = ipc_docs + bns_docs
            
            # Split into smaller chunks to keep token count lower
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(all_docs)
            
            # Create vector store
            vectorstore = Chroma.from_documents(
                documents=splits, 
                embedding=embeddings,
                client=chroma_client,
                collection_name="legal_codes_collection"
            )
            st.session_state.retriever = vectorstore.as_retriever(
                search_kwargs={"k": 5}  # Reduce number of documents to avoid token limit
            )
            st.session_state.vector_stores_initialized = True
            st.success("Legal documents loaded successfully!")
    except Exception as e:
        st.error(f"Error loading documents: {e}")
        st.stop()

if st.session_state.vector_stores_initialized:
    retriever = st.session_state.retriever
    
    # Function to filter and limit document content to reduce tokens
    def filter_relevant_documents(docs, query, max_docs=3, max_chars_per_doc=2000):
        # Sort by relevance (assuming retrieval already sorted them)
        # Make sure we have docs from both sources if possible
        ipc_docs = [doc for doc in docs if "Indian Penal Code" in doc.metadata.get("source", "")]
        bns_docs = [doc for doc in docs if "Bharatiya Nyaya Sanhita" in doc.metadata.get("source", "")]
        
        # Take top docs from each source
        selected_ipc = ipc_docs[:2] if ipc_docs else []
        selected_bns = bns_docs[:2] if bns_docs else []
        
        # If we don't have enough from both sources, fill with what we have
        selected_docs = selected_ipc + selected_bns
        if len(selected_docs) < max_docs:
            remaining_docs = [doc for doc in docs if doc not in selected_docs]
            selected_docs.extend(remaining_docs[:max_docs - len(selected_docs)])
        
        # Trim content to reduce token count
        trimmed_docs = []
        for doc in selected_docs[:max_docs]:
            content = doc.page_content
            if len(content) > max_chars_per_doc:
                content = content[:max_chars_per_doc] + "..."
            
            trimmed_docs.append(
                Document(page_content=content, metadata=doc.metadata)
            )
        
        return trimmed_docs
    
    # Create a history-aware retriever with improved contextualization
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which describes a legal case or scenario, "
        "formulate a concise standalone query that will help identify applicable clauses. "
        "The reformulated query should extract key legal issues and circumstances without unnecessary details. "
        "Be very brief and focus only on the core elements needed to match legal clauses."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    # Fixed CustomRetriever class using RunnableLambda for proper compatibility
    class CustomRetriever(RunnableLambda):
        def __init__(self, base_retriever):
            self.base_retriever = base_retriever
            super().__init__(self._retrieve_and_filter)
        
        def _retrieve_and_filter(self, query, **kwargs):
            docs = self.base_retriever.invoke(query, **kwargs)
            return filter_relevant_documents(docs, query)
            
        async def _aretrieve_and_filter(self, query, **kwargs):
            docs = await self.base_retriever.ainvoke(query, **kwargs)
            return filter_relevant_documents(docs, query)
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    custom_retriever = CustomRetriever(history_aware_retriever)
    
    # Answer question with specific legal comparison - shortened prompt
    system_prompt = (
        "You are a legal assistant. Analyze the case and identify applicable clauses from: "
        "1. Indian Penal Code (IPC) "
        "2. Bharatiya Nyaya Sanhita (BNS) "
        "\n\n"
        "Cite exact section numbers and briefly explain each. "
        "Highlight key differences between frameworks. "
        "Be concise and precise. Context: {context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(custom_retriever, question_answer_chain)
    
    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    
    # Create chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Get user input
    user_input = st.chat_input("Describe your legal case or scenario:")
    
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing legal codes..."):
                try:
                    # Clear old chat history if it's getting too large
                    session_history = get_session_history(session_id)
                    if len(session_history.messages) > 6:  # Keep history limited
                        # Keep only the last 3 pairs of messages
                        session_history.messages = session_history.messages[-6:]
                    
                    response = conversational_rag_chain.invoke(
                        {"input": user_input},
                        config={
                            "configurable": {"session_id": session_id}
                        }
                    )
                    st.markdown(response['answer'])
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response['answer']})
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    st.info("Try asking a more specific question or using fewer words to stay within token limits.")

else:
    st.warning("Please ensure both legal document PDFs are available at the specified paths.")