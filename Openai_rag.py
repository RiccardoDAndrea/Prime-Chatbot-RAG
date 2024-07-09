import streamlit as st
import openai
from langchain_community.document_loaders import PyPDFLoader
import os 
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.embeddings import HuggingFaceEmbeddings
import json
from tempfile import NamedTemporaryFile

openai_token = st.sidebar.text_input("OpenAI API Token", "sk-")
if len(openai_token) == 0:
    st.warning("Please enter your OpenAI API Token")
    st.stop()

# OpenAI API Token
Open_api_token = openai_token

class OpenAI_RAG:
    def __init__(self, Open_api_token: str, uploaded_file: str):
        self.Open_api_token = Open_api_token
        self.uploaded_file = uploaded_file

    def text_splitter(self):
        return RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50, 
            length_function=len,
        )

    def loader_for_chunks(self, text_splitter):
        if self.uploaded_file:
            try:
                with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(self.uploaded_file.read())
                loader = PyPDFLoader(temp_file.name)
                chunks = loader.load_and_split()
                os.unlink(temp_file.name)
            except Exception as e:
                st.error(f"Fehler beim Laden und Aufteilen der PDF: {e}")
                chunks = []
        else:
            chunks = []
        return chunks

    def embedding(self):
        try:
            embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            return embedding_function
        except Exception as e:
            st.error(f"Fehler beim Initialisieren der Embeddings: {e}")
            return None

    def initialise_chroma(self, chunks, embedding_function):
        try:
            db = Chroma.from_documents(chunks, embedding_function)
            return db
        except Exception as e:
            st.error(f"Fehler beim Initialisieren der Chroma-Datenbank: {e}")
            return None
    
    def retriever(self, db, query):
        try:
            retriever = db.as_retriever(search_kwargs={"k": 2})
            retriever.invoke(query)
            return retriever
        except Exception as e:
            st.error(f"Fehler beim Abrufen der Dokumente: {e}")
            return None

    def llm_model(self):
        try:
            llm = ChatOpenAI(
                openai_api_key=Open_api_token,
                model_name="gpt-3.5-turbo",
                temperature=0.0,
                max_tokens=300
            )
            return llm
        except Exception as e:
            st.error(f"Fehler beim Initialisieren des OpenAI-Modells: {e}")
            return None

    def qa_with_sources(self, query):
        try:
            llm = self.llm_model()
            if not llm:
                return {"answer": "Fehler beim Initialisieren des LLM-Modells."}
            text_splitter_instance = self.text_splitter()
            chunks = self.loader_for_chunks(text_splitter_instance)
            embedding_instance = self.embedding()
            if not embedding_instance:
                return {"answer": "Fehler beim Initialisieren der Embeddings."}
            db = self.initialise_chroma(chunks, embedding_instance)
            if not db:
                return {"answer": "Fehler beim Initialisieren der Chroma-Datenbank."}
            retriever_instance = self.retriever(db, query)
            if not retriever_instance:
                return {"answer": "Fehler beim Abrufen der Dokumente."}
            qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever_instance)
            return qa_with_sources.invoke(query)
        except Exception as e:
            st.error(f"Fehler bei der QA-Verarbeitung: {e}")
            return {"answer": f"Fehler bei der Verarbeitung der Anfrage: {e}"}

# Streamlit Main
st.sidebar.title("OpenAI RAG")

st.sidebar.markdown("### Beispiel-PDFs")
pdf_1 = '/Users/riccardo/Desktop/Github/LLM_RAG/N_BA_Riccardo_DAndrea_966697.pdf'


st.sidebar.multiselect('PDF_1', options= ["Test"]) 

st.title("OpenAI RAG")
st.write("""This is a simple implementation of OpenAI's 
          Retrieval Augmented Generation (RAG) model. 
          The model is trained on a combination of 
          supervised and reinforcement learning. 
          It is capable of generating long-form answers 
          to questions, and can be used for a variety 
          of tasks, such as question answering, 
          summarization, and translation.""")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

openai_rag = OpenAI_RAG(Open_api_token, uploaded_file)

# Chat
if uploaded_file:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about: " + uploaded_file.name if uploaded_file else ""):
        if uploaded_file:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            with st.spinner("Thinking..."):  # Display spinner while processing
                antwort = openai_rag.qa_with_sources(prompt)
            
            with st.chat_message("assistant"):
                st.write(antwort["answer"])
                st.session_state.messages.append({"role": "assistant", "content": antwort["answer"]})
else:
    with st.chat_message("assistant"):
        st.markdown("""
                    Welcome to Prime! 🤖 I am your personal document detective! Send me your PDFs and I will put them through their paces. From 
                    - "What's this about?" to 
                    - "What are the key points?" and even 
                    - "What's the scoop on topic X?" - 

                    I'm your man! 🕵️‍♂️ Uh, your bot. Never mind, you know. Let us crack your PDFs! 💼
                    Don't forget to upload a PDF! 📎
                    """)
