from ollama import chat
from ollama import ChatResponse
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import SKLearnVectorStore
import os

#os.environ['USER_AGENT'] = 'myagent'

def pdfloader(file_path):
    """
    Load PDF files  for the loader.

    Retrieves:
        - docs: A List with the String of the PDF file

    Parameters
    ----------
    file_path : str
        Path to the PDF files.

    Returns
    -------
    A List with the String of the PDF File
    """

    file_path = "PDF_docs/the-economic-potential-of-generative-ai-the-next-productivity-frontier-vf.pdf"  
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    return docs

# docs = pdfloader("PDF_docs/the-economic-potential-of-generative-ai-the-next-productivity-frontier-vf.pdf")
# In docs sind alle Inhalt der Seite enthalten



# Initialize a text splitter with specified chunk size and overlap

def chunkssplitter(chunk_size = int, chunk_overlap=int):
    """
    Retrieves:
        - docs splitts: A List with the String of the PDF file

    Parameters
    ----------
    chunk_size : int
        - Number of chunks through which the text is divided
     chunk_overlap: int
        - Number of overlaps between the chunks

    Returns
    -------
    Text that has been split into chunks
    """

    docs = pdfloader("PDF_docs/the-economic-potential-of-generative-ai-the-next-productivity-frontier-vf.pdf")       
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    # Split the documents into chunks
    doc_splits = text_splitter.split_documents(docs)
    return doc_splits

#doc_splits = chunkssplitter(chunk_size= 4500, chunk_overlap=300) # Seite ist auf page_lage nicht page
#print(doc_splits[0])

# Create embeddings for documents and store them in a vector store

def create_vectorstore():
    """
    Creates a vector store from a document by splitting it into chunks and embedding them.

    Retrieves:
        - doc_splits: A list of strings representing the text chunks from the PDF document.

    Parameters
    ----------
    None

    Returns
    -------
    vectorstore : SKLearnVectorStore
        - A vector store containing the embedded document chunks.
    """
    doc_splits = chunkssplitter(chunk_size=4500, chunk_overlap=300)
    vectorstore = SKLearnVectorStore.from_documents(
                documents=doc_splits,
                embedding=OllamaEmbeddings(model="llama3.2"))
    return vectorstore

#vectorstore = create_vectorstore()



def retriever(k_int:int):
    """
    Retrieves a amount of Information-

    Retrieves:
        - retriever : Pages of the retreives PDF-Document.

    Parameters
    ----------
    chunk_size : int
        - The number of characters each chunk should contain.
    chunk_overlap : int
        - The number of characters that should overlap between consecutive chunks.

    Returns
    -------
    vectorstore : SKLearnVectorStore
        - A vector store containing the embedded document chunks.
    """
    vectorstore = create_vectorstore()
    retriever = vectorstore.as_retriever(k=k_int)
    return retriever

#Retriever = retriever()


def promptTemplate():
    """
    Retrieves a amount of Information-

    Retrieves:
        - retriever : Pages of the retreives PDF-Document.

    Parameters
    ----------
    chunk_size : int
        - The number of characters each chunk should contain.
    chunk_overlap : int
        - The number of characters that should overlap between consecutive chunks.

    Returns
    -------
    vectorstore : SKLearnVectorStore
        - A vector store containing the embedded document chunks.
    """
    # Define the prompt template for the LLM
    prompt = PromptTemplate(
    template="""<s>[INST] <<SYS>>You are a friendly assistant called “Prime Chatbot”, 
    that summarizes documents briefly and pragmatically, focusing on the most important points.
    If you don't have an answer, say that you can't answer the question satisfactorily. <</SYS>>
    Question: {question}
    Documents: {documents}
    [/INST]""",
    input_variables=["question", "documents"],
        )

    return prompt
#prompt = promptTemplate()


# Initialize the LLM with Llama 3.1 model
def llm(model= str):

    llm = ChatOllama(
            model=model,
            temperature=0.2)
    return llm

#llm = llm(model="llama3.2:1b")



# # Create a chain combining the prompt template and LLM
# rag_chain = prompt | llm | StrOutputParser()

#  # Define the RAG application class
# class RAGApplication:
#     def __init__(self, Retriever, rag_chain):
#         self.Retriever = Retriever
#         self.rag_chain = rag_chain
#     def run(self, question):
#         # Retrieve relevant documents
#         documents = self.Retriever.invoke(question)
#         # Extract content from retrieved documents
#         doc_texts = "\\n".join([doc.page_content for doc in documents])
#         # Get the answer from the language model
        
#         answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})
#         return answer


# # Initialize the RAG application
# rag_application = RAGApplication(Retriever, rag_chain)
# # Example usage
# question = "Can you sumarries the paper?"
# answer = rag_application.run(question)
# print("Question:", question)
# print("Answer:", answer)
