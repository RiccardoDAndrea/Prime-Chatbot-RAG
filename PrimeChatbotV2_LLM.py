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

class PrimeChatbot:
    def __init__(self, file_path, model, chunk_size, chunk_overlap, k_int):
        self.file_path=file_path
        self.model = model
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap
        self.k_int = k_int


    def pdfloader(self):
        """
        Extracts data from the PDF File as a String.

        Retrieves
        ----------
        self.file_path : str
            Path to the PDF-File.

        Returns
        -------
        Strings form Dodcument.

        """

        loader = PyPDFLoader(self.file_path)
        docs = loader.load()

        return docs


    def chunkssplitter(self):
        """
        Splits the document into chunks.

        Retrieves
        ----------
        self.chunk_size : int
            - Number of chunks of text
        self.chunk_overlap : int
            - Number of characters of the previous text

        Returns
        -------
        Strings form Dodcument.

        """
        
        docs = self.pdfloader()       
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap)
        
        doc_splits = text_splitter.split_documents(docs)

        return doc_splits
    

    # def create_vectorstore(self):
    #     """
    #     Creates a vector store from a document by splitting it into chunks and embedding them.
        
    #     Retrives:
    #     ---
    #         self.chunkssplitter(): The split document into chunks and overlaps between the chunks

    #     Returns:
    #     ---
    #     Collection of vectors 

    #     """

    #     doc_splits = self.chunkssplitter()
    #     vectorstore = SKLearnVectorStore.from_documents(
    #         documents=doc_splits,
    #         embedding=OllamaEmbeddings(model="llama3.2:latest")
    #     )

    #     return vectorstore
    
    def embeddings(self):
        embeddings = OllamaEmbeddings(model="llama3.2:latest")  # Korrekte Embeddings-Funktion
        return embeddings

    def vector_store(self):
        embeddings = self.embeddings()
        doc_splits = self.chunkssplitter()  
        print(f"Number of document chunks: {len(doc_splits)}")  # Debugging

        persistent_client = chromadb.PersistentClient()
        collection = persistent_client.get_or_create_collection("collection_name")
        
        collection.add(ids=["1", "2", "3"], documents=["a", "b", "c"])

        vector_store_from_client = Chroma(
            client=persistent_client,
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
        embeddings = self.embeddings()
        vector_store_chroma = self.vector_store()
        vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=embeddings,
            persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
        )
        retriever = vector_store_chroma.as_retriever(search_kwargs={"k": self.k_int})
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
            You are an expert AI assistant. Answer the given question based on the provided documents. 
            If the documents do not contain the answer, say 'I don't know'. Do not summarize.
            <</SYS>>
            Question: {question}
            Documents: {documents}
            Answer:""",
            input_variables=["question", "documents"],
            )
        return prompt


    def llm(self, model):
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
            temperature=0.5
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
        rag_chain = self.promptTemplate() | self.llm(self.model) #| StrOutputParser()

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


# Initialize the RAG application
PrimeChatbot = PrimeChatbot(file_path='PDF_docs/doc_0.pdf', 
                            model= "llama3.1", 
                            chunk_size=700, 
                            chunk_overlap=300,
                            k_int=3)

question = "Can you summaries the paper?"
answer= PrimeChatbot.initializeChatbot(question)
print( answer)





question = "Can you summaries the paper Two Hundred Years of Cancer Research?"
answer = PrimeChatbot.initializeChatbot(question)
print("Question:", question)
print("Answer:", answer)
