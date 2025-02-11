from langchain_core.output_parsers import StrOutputParser
from PrimeChatbotV2_LLM import pdfloader, chunkssplitter, create_vectorstore, retriever, llm, promptTemplate


# 1️⃣ Neue Dokumente laden
docs = pdfloader("PDF_docs/NEJMra1204479.pdf")  # Ändere den Pfad zum neuen PDF

# 2️⃣ Chunks neu erstellen
doc_splits = chunkssplitter(chunk_size=4500, chunk_overlap=300)

# 3️⃣ Vektorstore aktualisieren
vectorstore = create_vectorstore()

# 4️⃣ Retriever neu initialisieren
Retriever = retriever()

# 5️⃣ LLM & PromptTemplate laden
prompt = promptTemplate()
llm = llm(model="llama3.2:1b")

# 6️⃣ Chatbot auf neue Daten setzen
def initalise_PrimeV2(question):
    rag_chain = prompt | llm | StrOutputParser()
    documents = Retriever.invoke(question)
    doc_texts = "\\n".join([doc.page_content for doc in documents])
    answer = rag_chain.invoke({"question": question, "documents": doc_texts})
    return answer

# 7️⃣ Neue Frage mit dem aktualisierten System stellen
question = "Was steht im neuen Dokument?"
answer = initalise_PrimeV2(question)

print("Question:", question)
print("Answer:", answer)


