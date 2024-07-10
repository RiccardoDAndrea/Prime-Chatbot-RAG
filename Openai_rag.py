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

# Streamlit Sidebar for OpenAI API Token
Open_api_token = st.sidebar.text_input("OpenAI API Token", "sk-", type="password")
if len(Open_api_token) == 0:
    st.warning("Please enter your OpenAI API Token")
    st.stop()

# OpenAI API Token


# Sidebar for selecting example PDFs or uploading a PDF
st.sidebar.title("OpenAI RAG")

st.sidebar.markdown("### Example PDFs")
example_pdfs = {
    "Short Stories": "/Users/riccardo/Desktop/Github/LLM_RAG/16_Kurzgeschichten.pdf",
    "Marketing Results": "/Users/riccardo/Desktop/Github/LLM_RAG/s12943-023-01867-y.pdf",
}

selected_example_pdfs = st.sidebar.selectbox('Choose your PDF example', options=["Upload your own data", "Short Stories", "Marketing Results"])

st.title("OpenAI RAG")

st.write("""This is a simple implementation of OpenAI's 
          Retrieval Augmented Generation (RAG) model. 
          The model is trained on a combination of 
          supervised and reinforcement learning. 
          It is capable of generating long-form answers 
          to questions, and can be used for a variety 
          of tasks, such as question answering, 
          summarization, and translation.""")

chatbot_tab, pdf_reader_tab = st.tabs(["Chatbot", "PDF Reader"])

with chatbot_tab:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Show file uploader only if "Upload your own data" is selected
    uploaded_file = None    
    if selected_example_pdfs == "Upload your own data":
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], key="pdf_uploader")
    
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
            
    if prompt := st.chat_input("Ask a question about: " + uploaded_file.name if uploaded_file else "Ask a question about the selected PDF"):
        if uploaded_file or selected_example_pdfs:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            with st.spinner("Thinking..."):  # Display spinner while processing
                
                # Initializing text splitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=200,
                    chunk_overlap=50,
                    length_function=len,
                )
                
                # Loading and splitting PDF into chunks
                chunks = []
                if uploaded_file:
                    try:
                        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                            temp_file.write(uploaded_file.read())
                        loader = PyPDFLoader(temp_file.name)
                        chunks += loader.load_and_split()
                        os.unlink(temp_file.name)
                    except Exception as e:
                        st.error(f"Error loading and splitting the PDF: {e}")

                if selected_example_pdfs != "Upload your own data":
                    try:
                        pdf_path = example_pdfs[selected_example_pdfs]
                        loader = PyPDFLoader(pdf_path)
                        chunks += loader.load_and_split()
                    except Exception as e:
                        st.error(f"Error loading and splitting the example PDF {pdf_path}: {e}")

                # Initializing embeddings
                try:
                    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                except Exception as e:
                    st.error(f"Error initializing embeddings: {e}")
                    embedding_function = None

                # Initializing Chroma database
                try:
                    if embedding_function:
                        db = Chroma.from_documents(chunks, embedding_function)
                    else:
                        db = None
                except Exception as e:
                    st.error(f"Error initializing Chroma database: {e}")
                    db = None

                # Retrieving documents
                try:
                    if db:
                        retriever = db.as_retriever(search_kwargs={"k": 2})
                        retriever.invoke(prompt)
                    else:
                        retriever = None
                except Exception as e:
                    st.error(f"Error retrieving documents: {e}")
                    retriever = None

                # Initializing LLM model
                try:
                    llm = ChatOpenAI(
                        openai_api_key=Open_api_token,
                        model_name="gpt-3.5-turbo",
                        temperature=0.0,
                        max_tokens=300
                    )
                except Exception as e:
                    st.error(f"Error initializing OpenAI model: {e}")
                    llm = None

                # Processing QA with sources
                try:
                    if llm and retriever:
                        qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
                        antwort = qa_with_sources.invoke(prompt)
                    else:
                        antwort = {"answer": "Error initializing components."}
                except Exception as e:
                    st.error(f"Error processing QA: {e}")
                    antwort = {"answer": f"Error processing request: {e}"}

            with st.chat_message("assistant"):
                st.write(antwort["answer"])
                st.session_state.messages.append({"role": "assistant", "content": antwort["answer"]})

with pdf_reader_tab:
    st.write("PDF Reader")
    
    if selected_example_pdfs == "Upload your own data" and uploaded_file is not None:
        # Read the binary data from the uploaded file
        binary_data = uploaded_file.read()
        pdf_viewer(input=binary_data, width=700)
    
    elif selected_example_pdfs == "Short Stories":
        pdf_path_short_stories = "/Users/riccardo/Desktop/Github/LLM_RAG/16_Kurzgeschichten.pdf"
        with open(pdf_path_short_stories, "rb") as f:
            binary_data = f.read()
        pdf_viewer(input=binary_data, width=700)    
    
    elif selected_example_pdfs == "Marketing Results":
        pdf_path_marketing = "/Users/riccardo/Desktop/Github/LLM_RAG/s12943-023-01867-y.pdf"
        with open(pdf_path_marketing, "rb") as f:
            binary_data = f.read()
        pdf_viewer(input=binary_data, width=700)
    
    else:
        st.write("Please upload a PDF or select an example PDF.")
 