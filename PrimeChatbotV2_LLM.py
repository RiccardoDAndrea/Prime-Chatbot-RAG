from ollama import ChatResponse, chat
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.output_parsers import StrOutputParser
import os

class PrimeChatbot:
    def __init__(self, file_path, model, chunk_size, chunk_overlap):
        self.file_path=file_path
        self.model = model
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap


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


    def create_vectorstore(self):
        """
        Creates a vector store from a document by splitting it into chunks and embedding them.
        
        Retrives:
        ---
            self.chunkssplitter(): The split document into chunks and overlaps between the chunks

        Returns:
        ---
        Collection of vectors 

        """

        doc_splits = self.chunkssplitter()
        vectorstore = SKLearnVectorStore.from_documents(
            documents=doc_splits,
            embedding=OllamaEmbeddings(model="llama3.2:latest")
        )

        return vectorstore


    def Retriever(self, k_int=5):
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

        vectorstore = self.create_vectorstore()
        retriever = vectorstore.as_retriever(k=k_int,
                                            search_type="similarity",
                                            search_kwargs={"k": k_int}
                                            )
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
        rag_chain = self.promptTemplate() | self.llm(self.model) | StrOutputParser()

        return rag_chain


    def initaliseChatbot(self, question):
        """
        Initializes the chatbot and processes the input question.
        """
        # Retrieve relevant documents
        retriever = self.Retriever(k_int=1)  # Get top 5 relevant documents
        documents = retriever.invoke(question)

        # Get the answer from the language model
        llm_chain = self.ragchain()
        answer = llm_chain.invoke({"question": question, "documents": documents})
        return answer


# Initialize the RAG application
PrimeChatbot = PrimeChatbot(file_path='PDF_docs/doc_0.pdf', 
                            model= "llama3.1", 
                            chunk_size=500, 
                            chunk_overlap=300)


docs = PrimeChatbot.Retriever()
#print(docs)

question = "Can you give me some sources that have been added to the document?"
answer = PrimeChatbot.initaliseChatbot(question)
print("Question:", question)
print("Answer:", answer)
