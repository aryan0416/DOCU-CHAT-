from langchain_community.document_loaders import Docx2txtLoader, TextLoader
import os
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain_core.output_parsers import StrOutputParser

# --- Global Configurations ---
CHROMA_DB_PATH = "./chroma_db"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = GoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Backend Functions ---

# In app.py

def build_or_get_vector_store(file_paths):
    """
    Creates a persistent Chroma vector store from a list of files.
    Handles PDF, DOCX, and TXT files.
    """
    all_chunks = []
    for path in file_paths:
        file_extension = os.path.splitext(path)[1].lower()
        loader = None

        if file_extension == ".pdf":
            loader = PyPDFLoader(path)
        elif file_extension == ".docx":
            loader = Docx2txtLoader(path)
        elif file_extension == ".txt":
            loader = TextLoader(path)
        else:
            print(f"Warning: Unsupported file type '{file_extension}' for file '{os.path.basename(path)}'. Skipping.")
            continue

        if loader:
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)
            all_chunks.extend(chunks)

    if os.path.exists(CHROMA_DB_PATH) and os.listdir(CHROMA_DB_PATH):
        vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_model)
        if all_chunks:
            vector_store.add_documents(all_chunks)
            print(f"Added {len(all_chunks)} new chunks to the existing vector store.")
    else:
        if not all_chunks:
            return None
        vector_store = Chroma.from_documents(
            documents=all_chunks,
            embedding=embedding_model,
            persist_directory=CHROMA_DB_PATH
        )
        print(f"Created a new vector store with {len(all_chunks)} chunks.")

    return vector_store

def load_vector_store():
    """Loads an existing vector store."""
    if os.path.exists(CHROMA_DB_PATH):
        return Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_model)
    return None

def create_main_chain(vector_store):
    """
    Creates the main conversational chain. This function is now simpler as it just
    receives an already-built vector_store.
    """
    if not vector_store:
        return None
    
    retriever = vector_store.as_retriever(search_type="mmr")
    
    # The rest of this function is identical to the previous version
    condense_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    question_rewriter_chain = condense_q_prompt | llm | StrOutputParser()

    rag_prompt_template = """
You are an expert assistant for questioning documents. Your goal is to provide comprehensive, synthesized answers based on the provided context from multiple documents.
When answering, follow these rules:
1. Synthesize information from the context to form a coherent answer. Do not simply copy-paste excerpts.
2. If the context does not contain the answer, explicitly state that the document does not provide this information.
3. After your answer, list the sources you used in a 'Sources:' section. Refer to them by the page number provided in the context.

CONTEXT:
{context}

QUESTION:
{question}

YOUR ANSWER:"""
    rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)
    
    def format_docs_with_sources(docs):
        formatted_docs = []
        for doc in docs:
            page = doc.metadata.get('page', 'N/A')
            page_number = page + 1 if isinstance(page, int) else page
            source_info = f"Source from Page: {page_number}"
            doc_string = f"--- {source_info} ---\n{doc.page_content}"
            formatted_docs.append(doc_string)
        return "\n\n".join(formatted_docs)
        
    rag_chain_base = RunnablePassthrough.assign(
        context=(lambda x: x["question"]) | retriever | format_docs_with_sources
    ) | rag_prompt | llm | StrOutputParser()
    
    conversational_rag_chain = RunnablePassthrough.assign(
        question=question_rewriter_chain
    ).assign(
        answer=rag_chain_base
    ) | (lambda x: x["answer"])

    general_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful general knowledge assistant. Provide clear and concise answers to the user's questions based on your internal knowledge."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    general_chain = RunnablePassthrough.assign(
        question=question_rewriter_chain
    ) | general_prompt | llm | StrOutputParser()

    router_prompt = ChatPromptTemplate.from_template(
        """Given the user question, classify it as either being about a "specific_document" or "general_knowledge". Do not answer the question. Just return the single word classification.

Question: {question}
Classification:"""
    )
    router_chain = router_prompt | llm | StrOutputParser()

    branch = RunnableBranch(
        (lambda x: "specific_document" in x["topic"].lower(), conversational_rag_chain),
        general_chain,
    )

    return RunnablePassthrough.assign(
        topic= (lambda x: x["question"]) | router_chain
    ) | branch
