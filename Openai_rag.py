import streamlit as st
import openai
from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from tempfile import NamedTemporaryFile
from streamlit_pdf_viewer import pdf_viewer

Open_api_token = st.sidebar.text_input("OpenAI API Token", "sk-", type="password")

example_pdfs = {
    "Short Stories": "/Users/riccardo/Desktop/Github/LLM_RAG/16_Kurzgeschichten.pdf",
    "Marketing Results": "/Users/riccardo/Desktop/Github/LLM_RAG/s12943-023-01867-y.pdf",
}

# Streamlit Main
st.sidebar.title("OpenAI RAG")

st.sidebar.markdown("### Example PDFs")
selected_example_pdfs = st.sidebar.selectbox('Choose your PDF example', options=["Short Stories", "Marketing Results", "Upload your own data"])

st.title("OpenAI RAG")

st.write("""This is a simple implementation of OpenAI's 
          Retrieval Augmented Generation (RAG) model. 
          The model is trained on a combination of 
          supervised and reinforcement learning. 
          It is capable of generating long-form answers 
          to questions, and can be used for a variety 
          of tasks, such as question answering, 
          summarization, and translation.""")

chatbot_tab, Pdf_reader_tab = st.tabs(["Chatbot", "PDF Reader"])

# Function to load and split PDFs
def load_and_split_pdf(uploaded_file, selected_example):
    chunks = []
    # Load the uploaded file
    if uploaded_file:
        try:
            with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
            loader = PyPDFLoader(temp_file.name)
            chunks += loader.load_and_split()
            os.unlink(temp_file.name)
        except Exception as e:
            st.error(f"Error loading and splitting the PDF: {e}")
    
    # Load the selected example PDF
    elif selected_example and example_pdfs[selected_example]:
        try:
            pdf_path = example_pdfs[selected_example]
            loader = PyPDFLoader(pdf_path)
            chunks += loader.load_and_split()
        except Exception as e:
            st.error(f"Error loading and splitting the example PDF {pdf_path}: {e}")

    return chunks

# Function to initialize embeddings
def initialize_embeddings():
    try:
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return embedding_function
    except Exception as e:
        st.error(f"Error initializing embeddings: {e}")
        return None

# Function to initialize Chroma database
def initialize_chroma(chunks, embedding_function):
    try:
        db = Chroma.from_documents(chunks, embedding_function)
        return db
    except Exception as e:
        st.error(f"Error initializing Chroma database: {e}")
        return None

# Function to retrieve documents
def retrieve_documents(db, query):
    try:
        retriever = db.as_retriever(search_kwargs={"k": 2})
        retriever.invoke(query)
        return retriever
    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        return None

# Function to initialize LLM model
def initialize_llm_model():
    try:
        llm = ChatOpenAI(
            openai_api_key=Open_api_token,
            model_name="gpt-3.5-turbo",
            temperature=0.0,
            max_tokens=300
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing OpenAI model: {e}")
        return None

# Function to perform QA with sources
def qa_with_sources(llm_instance, chunks, embedding_instance, query):
    try:
        text_splitter_instance = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            length_function=len,
        )
        db = initialize_chroma(chunks, embedding_instance)
        if not db:
            return {"answer": "Error initializing Chroma database."}
        retriever_instance = retrieve_documents(db, query)
        if not retriever_instance:
            return {"answer": "Error retrieving documents."}
        qa_with_sources_result = RetrievalQAWithSourcesChain.from_chain_type(llm=llm_instance, chain_type="stuff", retriever=retriever_instance)
        return qa_with_sources_result.invoke(query)
    except Exception as e:
        st.error(f"Error processing QA: {e}")
        return {"answer": f"Error processing request: {e}"}

# Main execution
if chatbot_tab:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Show file uploader only if "Upload your own data" is selected
    uploaded_file = None
    if selected_example_pdfs == "Upload your own data":
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
        if "messages" not in st.session_state:
            st.session_state.messages = []
        with st.chat_message("assistant"):
            st.markdown("""
                        Welcome to Prime! 🤖 I am your personal document detective! Send me your PDFs and I will put them through their paces. From 
                        - "What's this about?" to 
                        - "What are the key points?" and even 
                        - "What's the scoop on topic X?" - 

                        I'm your man! 🕵️‍♂️ Uh, your bot. Never mind, you know. Let us crack your PDFs! 💼
                        Don't forget to upload a PDF or select an example PDF! 📎
                        """)
    else:
        with st.chat_message("assistant"):
            st.markdown("""
                        Welcome to Prime! 🤖 I am your personal document detective! Send me your PDFs and I will put them through their paces. From 
                        - "What's this about?" to 
                        - "What are the key points?" and even 
                        - "What's the scoop on topic X?" - 

                        I'm your man! 🕵️‍♂️ Uh, your bot. Never mind, you know. Let us crack your PDFs! 💼
                        Don't forget to upload a PDF or select an example PDF! 📎
                        """)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if prompt := st.chat_input("Ask a question about the uploaded or selected PDFs"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.spinner("Thinking..."):  # Display spinner while processing
            llm_instance = initialize_llm_model()
            if not llm_instance:
                st.write("Error initializing the LLM model.")
            else:
                chunks = load_and_split_pdf(uploaded_file, selected_example_pdfs)
                embedding_instance = initialize_embeddings()
                if not embedding_instance:
                    st.write("Error initializing embeddings.")
                else:
                    answer = qa_with_sources(llm_instance, chunks, embedding_instance, prompt)
                    st.write(answer["answer"])
                    st.session_state.messages.append({"role": "assistant", "content": answer["answer"]})
with Pdf_reader_tab:
    if Pdf_reader_tab:
        st.write("PDF Reader")
        pdf_file = st.file_uploader("Upload PDF file", type=('pdf'))

        if pdf_file:
            binary_data = pdf_file.getvalue()
            # pdf_viewer(input=binary_data, width=700)  # You would implement this function to view PDF content
