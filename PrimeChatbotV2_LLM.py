from ollama import ChatResponse, chat
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_chroma import Chroma
import chromadb
import uuid

## TODO Write a function for debuing specially for chroma and retriver


class PrimeChatbot:
    def __init__(self, file_path, model, chunk_size, chunk_overlap, k_int):
        self.file_path=file_path
        self.model = model
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap
        self.k_int = k_int


    def pdfloader(self):
        pdf_files = [f for f in os.listdir(self.file_path) if f.endswith(".pdf")]
        print(f"📂 Gefundene PDFs: {pdf_files}")  # Debug: Zeige gefundene PDFs an

        all_docs = []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.file_path, pdf_file)
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            print(f"📄 {pdf_file}: {len(docs)} Seiten geladen")  # Debug
            all_docs.extend(docs)

        return all_docs


    def chunkssplitter(self):
        all_docs = self.pdfloader()  # `self.file_path` ist bereits in der Methode definiert
        print(f"📄 Geladene Dokumente: {len(all_docs)}")  # Debug

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )

        doc_splits = text_splitter.split_documents(all_docs)
        print(f"🧩 Anzahl der Chunks nach Split: {len(doc_splits)}")  # Debug
        return doc_splits
    
    
    def embedding(self):
        embeddings = OllamaEmbeddings(model='all-minilm')  # Korrekte Embeddings
        return embeddings

    def persistent_clientChroma(self):
        persistent_client = chromadb.PersistentClient(path="chroma_db")  # Verzeichnis für Speicherung
        heartbeat = persistent_client.heartbeat()
        print(f"ChromaDB läuft. Serverzeit: {heartbeat}")

        collection = persistent_client.get_or_create_collection("collection_name")
        return collection
    
    def create_unqiueIds(self):
        doc_splits = self.chunkssplitter()
        ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in doc_splits]
        unique_ids = list(set(ids))
        return unique_ids
    
    def add_doc_to_Chroma(self):
        doc_splits = self.chunkssplitter()
        unique_ids = self.create_unqiueIds()

        doc_texts = [doc.page_content for doc in doc_splits]  # Extrahiere den Text
        #doc_ids = [f"doc_{i}" for i in range(len(doc_texts))]  # Einzigartige IDs
        collection = self.persistent_clientChroma()
        collection_db = collection.add(ids=unique_ids, documents=doc_texts)  # Speichern in ChromaDB
        return collection_db
    
    def vector_store_from_clientChroma(self):
        # Schritt 5: Chroma-VectorStore mit gespeicherten Daten initialisieren
        
        embeddings = self.embedding()
        vector_store_from_client = Chroma(
            persist_directory="chroma_db",
            collection_name="collection_name",
            embedding_function=embeddings,
        )
        return vector_store_from_client



    def Retriever(self):
        """
        Retrieves a specified number of relevant documents.
        Retrives:
        ---
            self.create_vectorstore(): Creates a vector store from a document by 
                                       splitting it into chunks and embedding them.

        Parameter:
            k_int (int): The number of top-scoring documents to retrieve. Defaults to 5.

        Returns:
            A retriever object that can be used to search for documents based on 
            their similarity score.
        """
        vector_store_from_client = self.vector_store_from_clientChroma()

        retriever = vector_store_from_client.as_retriever(search_kwargs={"k": self.k_int})
        return retriever


    def promptTemplate(self):
        """
        Creates the prompt Instruction template for the language model.

        Retrives:
            Question: str
                Query from the user

        Returns:
            Intruction for the LLM-Model
        """

        prompt = PromptTemplate(
            template="""<s>[INST] <<SYS>>
            
            <</SYS>>
            Question: {question}
            Documents: {documents}
            Answer:""",
            input_variables=["question", "documents"],
            )
        return prompt


    def llm(self):
        """
        Initializes the language model.

        Retrives:
        ---
            self.model(): Creates a vector store from a document by 
                          splitting it into chunks and embedding them.

        Parameter:
            k_int (int): The number of top-scoring documents to retrieve. Defaults to 5.

        Returns:
            A retriever object that can be used to search for documents based on 
            their similarity score.
        """
        
        llm = ChatOllama(
            model=self.model,
            temperature=0.7
        )
        return llm


    def ragchain(self):
        """

        
        Creates the RAG (Retrieval-Augmented Generation) chain.
        
        Retrives:
        ---
            self.promptTemplate(): 
                Creates the prompt Instruction template for the language model.
            self.llm(): 
                Initializes the language model.
        Returns
        ---
        A RagChain object that combines an Instruction template and a language model.
        The RagChain object contains a sequence of operations, starting with the creation of a prompt instruction template,
        followed by the initialization of a language model. This chain can be used for text generation tasks.

        """
        rag_chain = self.promptTemplate() | self.llm() | StrOutputParser()

        return rag_chain


    def initializeChatbot(self, question):
        retriever = self.Retriever()
        documents = retriever.invoke(question)

        if not documents:  # Falls keine relevanten Dokumente gefunden wurden
            print("⚠️ No relevant documents found in Chroma!")
            return "I don't know. The document might not contain the answer."

        print(f"✅ Retrieved {len(documents)} document(s)")  # Debugging
        
        llm_chain = self.ragchain()
        answer = llm_chain.invoke({"question": question, "documents": documents})
        return answer

# retriever = PrimeChatbot.Retriever()
# documents = retriever.invoke(question)



# doc_splits = PrimeChatbot.chunkssplitter()
# print(f"📄 Anzahl der Chunks: {len(doc_splits)}")
















#########################################################################
### G E T _ A N S W E R _ F R O M _ T H E _ P R I M E _ C H A T B O T ###
#########################################################################
# Erstelle eine Instanz der Klasse
prime_chatbot = PrimeChatbot(file_path="PDF_docs/", 
                             model="llama3.2:1b", 
                             chunk_size=1000, 
                             chunk_overlap=200, 
                             k_int=10)


question = """What information do you have about Germany?"""
answer = prime_chatbot.initializeChatbot(question=question)
print(answer)

# # Rufe die Methode auf der Instanz auf
# collection = prime_chatbot.persistent_clientChroma()
# print(collection.count())  # Sollte die Chroma-Sammlung zurückgeben

prime_chatbot.add_doc_to_Chroma()  # Füge Dokumente zur Chroma-Datenbank hinzu
print("Anzahl gespeicherter Dokumente:", prime_chatbot.persistent_clientChroma().count())
