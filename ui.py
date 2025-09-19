import streamlit as st
import os
from src.app import build_or_get_vector_store, create_main_chain, load_vector_store
from langchain_core.messages import AIMessage, HumanMessage

# --- UI Configuration ---
st.set_page_config(page_title="DocuChat 4.0", page_icon="📚", layout="wide")

# --- Custom CSS for Chat Bubbles ---
st.markdown("""
    <style>
        /* This targets the container for each chat message */
        div[data-testid="stChatMessage"] {
            border-radius: 20px;
            padding: 1rem 1.25rem;
            margin-bottom: 1rem;
            max-width: 75%;
            border: 1px solid transparent;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        }

        /* AI (assistant) messages styling */
        div[data-testid="stChatMessage"]:has(div[data-testid="stAvatarIcon-ai"]) {
            background-color: #f0f2f6; /* Light grey background */
            float: left;
            clear: both;
        }

        /* User messages styling */
        div[data-testid="stChatMessage"]:has(div[data-testid="stAvatarIcon-user"]) {
            background-color: #dcf8c6; /* Light green background */
            float: right;
            clear: both;
        }

        /* This is a fix to ensure the chat input area isn't pushed down */
        .st-emotion-cache-4oy321 {
            clear: both;
        }

        /* This is to align the avatar and message content vertically */
         .st-emotion-cache-1c7y2kd {
            align-items: center;
        }
    </style>
""", unsafe_allow_html=True)


# --- State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None

# On first run, try to load an existing vector store and create the chain
if st.session_state.chain is None:
    vector_store = load_vector_store()
    if vector_store:
        st.session_state.chain = create_main_chain(vector_store)

# --- UI Elements ---
st.title("       DocuChat 4.0: Multi-Document Q&A")
st.info("Upload one or more documents to build your knowledge base, then ask questions.")

# --- Sidebar for File Upload and Processing ---
with st.sidebar:
    st.header("Your Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload your PDFs, DOCX, or TXT files", 
        type=["pdf", "docx", "txt"], 
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        if st.button("Build Knowledge Base"):
            with st.spinner("Processing documents... This may take a moment."):
                temp_dir = "temp_files"
                os.makedirs(temp_dir, exist_ok=True)
                file_paths = []
                for uploaded_file in uploaded_files:
                    path = os.path.join(temp_dir, uploaded_file.name)
                    with open(path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(path)
                
                vector_store = build_or_get_vector_store(file_paths)
                
                if vector_store:
                    st.session_state.chain = create_main_chain(vector_store)
                    st.success("Knowledge base built successfully!")
                else:
                    st.error("Failed to build knowledge base. Please upload documents.")
    
    st.divider()
    if "messages" in st.session_state and st.session_state.messages:
        if st.button("✨ Start New Chat"):
            st.session_state.messages = []
            st.rerun()

# --- Main Chat Interface ---
# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

# Handle user input
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("human"):
        st.markdown(prompt)

    if st.session_state.chain is not None:
        with st.chat_message("ai"):
            # This is the new streaming logic
            response_stream = st.session_state.chain.stream({
                "question": prompt,
                "chat_history": st.session_state.messages[:-1] # Exclude the current prompt
            })
            full_response = st.write_stream(response_stream)
        
        # Append the complete response to history once the stream is done
        st.session_state.messages.append(AIMessage(content=full_response))
    else:
        st.info("Please build a knowledge base first by uploading documents in the sidebar.")