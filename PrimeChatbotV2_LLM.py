import ollama
from ollama import chat
from ollama import ChatResponse
from langchain_ollama import OllamaEmbeddings
import os 
import bs4 
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import SKLearnVectorStore
import os

#os.environ['USER_AGENT'] = 'myagent'
## List of URLs to load documents from
#urls = [
#    "<https://lilianweng.github.io/posts/2023-06-23-agent/>",
#    "<https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/>",
#    "<https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/>",
#]


#loader = WebBaseLoader("https://python.langchain.com/docs/tutorials/rag/")


def pdfloader(file_path):
    """
    description: Load a PDF document from a file path 
    """

    file_path = "PDF_docs/the-economic-potential-of-generative-ai-the-next-productivity-frontier-vf.pdf"  
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return loader, docs

pdf, docs = pdfloader("PDF_docs/the-economic-potential-of-generative-ai-the-next-productivity-frontier-vf.pdf")


# Initialize a text splitter with specified chunk size and overlap

def chunkssplitter(chunk_size = int, chunk_overlap=int):
        
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    # Split the documents into chunks
    doc_splits = text_splitter.split_documents(docs)
    #print(doc_splits)
    return doc_splits

doc_splits = chunkssplitter(chunk_size= 300, chunk_overlap=50)



# Create embeddings for documents and store them in a vector store

def vectorstore():
    vectorstore = SKLearnVectorStore.from_documents(
                documents=doc_splits,
                embedding = OllamaEmbeddings(model="llama3.2"))
    return vectorstore
vectorstore = vectorstore()

def retriever():
    retriever = vectorstore.as_retriever(k=4)
    return retriever

retriever = retriever()

# Define the prompt template for the LLM
prompt = PromptTemplate(
    template="""
    You are an assistant for question-answering tasks.
    Use the following documents to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise:
    Question: {question}
    Documents: {documents}
    Answer:
    """,
    input_variables=["question", "documents"],
)


# Initialize the LLM with Llama 3.1 model
def llm(model= str):

    llm = ChatOllama(
            model=model,
            temperature=0.20)
    return llm

llm = llm(model="llama3.2")


# Create a chain combining the prompt template and LLM
rag_chain = prompt | llm | StrOutputParser()

# Define the RAG application class
class RAGApplication:
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain
    def run(self, question):
        # Retrieve relevant documents
        documents = self.retriever.invoke(question)
        # Extract content from retrieved documents
        doc_texts = "\\n".join([doc.page_content for doc in documents])
        # Get the answer from the language model
        answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})
        return answer


# Initialize the RAG application
rag_application = RAGApplication(retriever, rag_chain)
# Example usage
question = "How product R&D could be transformed"
answer = rag_application.run(question)
print("Question:", question)
print("Answer:", answer)
