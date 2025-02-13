from langchain_core.output_parsers import StrOutputParser
from PrimeChatbotV2_LLM import pdfloader, chunkssplitter, create_vectorstore, retriever, llm, promptTemplate


loader, docs = pdfloader("PDF_docs/NEJMra1204479.pdf")
doc_splits = chunkssplitter(chunk_size= 4500, chunk_overlap=300) # Seite ist auf "page_lage" nicht "page"
vectorstore = create_vectorstore()
Retriever = retriever()
prompt = promptTemplate()
llm = llm(model="llama3.2:1b")


def initalise_PrimeV2(question):
    # Retrieve relevant documentssssss
    rag_chain = prompt | llm | StrOutputParser()
    documents = Retriever.invoke(question)
    # Extract content from retrieved documents
    doc_texts = "\\n".join([doc.page_content for doc in documents])
    # Get the answer from the language model
    
    answer = rag_chain.invoke({"question": question, "documents": doc_texts})
    return answer

question = "Can you tell me what is the text about"
answer = initalise_PrimeV2(question)
print("Question:", question)
print("Answer:", answer)


