from langchain_core.output_parsers import StrOutputParser
from PrimeChatbotV2_LLM import pdfloader, chunkssplitter, create_vectorstore, retriever, llm, promptTemplate


loader, docs = pdfloader("PDF_docs/the-economic-potential-of-generative-ai-the-next-productivity-frontier-vf.pdf")
doc_splits = chunkssplitter(chunk_size= 4500, chunk_overlap=300) # Seite ist auf page_lage nicht page
vectorstore = create_vectorstore()
Retriever = retriever()
prompt = promptTemplate()
llm = llm(model="llama3.2")


def initalise_PrimeV2(question):
    # Retrieve relevant documents
    rag_chain = prompt | llm | StrOutputParser()
    documents = Retriever.invoke(question)
    # Extract content from retrieved documents
    doc_texts = "\\n".join([doc.page_content for doc in documents])
    # Get the answer from the language model
    
    answer = rag_chain.invoke({"question": question, "documents": doc_texts})
    return answer

question = "Was steht im Appendix"
answer = initalise_PrimeV2.run(question)
print("Question:", question)
print("Answer:", answer)


