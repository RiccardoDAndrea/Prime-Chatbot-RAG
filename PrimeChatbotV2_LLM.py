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
    def __init__(self, file_path, chunk_size, chunk_overlap):
        self.file_path = file_path
        self.chunk_size=chunk_size
        self.chunk_overlap = chunk_overlap

    def pdfloader(self):
        """
        Load PDF files for the loader.
        """
        loader = PyPDFLoader(self.file_path)
        docs = loader.load()
        return docs

    def chunkssplitter(self):
        """
        Splits the document into chunks.
        """
        docs = self.pdfloader()       
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        doc_splits = text_splitter.split_documents(docs)
        return doc_splits

    def create_vectorstore(self):
        """
        Creates a vector store from a document by splitting it into chunks and embedding them.
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
        """
        vectorstore = self.create_vectorstore()
        retriever = vectorstore.as_retriever(k=k_int,
                                            search_type="similarity_score_threshold",
                                            search_kwargs={'score_threshold': 0.8})
        return retriever

    def promptTemplate(self):
        """
        Creates the prompt template for the language model.
        """
        prompt = PromptTemplate(
            template="""<s>[INST] <<SYS>>You are a friendly assistant called “Prime Chatbot” who 
            answers questions about the added source. If you do not have an answer, say that you 
            cannot answer the question satisfactorily. <</SYS>>
            Question: {question}
            Documents: {documents}
            [/INST]""",
            input_variables=["question", "documents"],
        )
        return prompt

    def llm(self, model):
        """
        Initializes the language model.
        """
        llm = ChatOllama(
            model=model,
            temperature=0.5
        )
        return llm

    def ragchain(self):
        """
        Creates the RAG (Retrieval-Augmented Generation) chain.
        """
        prompt = self.promptTemplate()
        llm = self.llm(model='llama3.2:latest')
        rag_chain = prompt | llm 
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
PrimeChatbot = PrimeChatbot(file_path='PDF_docs/doc_0.pdf', chunk_size=500, chunk_overlap=300)
docs = PrimeChatbot.chunkssplitter()
print(docs)


question = "Can you tell me the first topic?"
answer = PrimeChatbot.initaliseChatbot(question)
print("Question:", question)
print("Answer:", answer)
