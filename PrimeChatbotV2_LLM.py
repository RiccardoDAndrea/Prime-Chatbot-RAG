from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
import chromadb
import os
import uuid

class PrimeChatbot:
    def __init__(self, file_path, model, chunk_size, chunk_overlap, k_int):
        self.file_path = file_path
        self.model = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k_int = k_int
        self.doc_splits = None

    def pdfloader(self):
        pdf_files = [f for f in os.listdir(self.file_path) if f.endswith(".pdf")]
        print(f"📂 Gefundene PDFs: {pdf_files}")
        all_docs = []
        for pdf_file in pdf_files:
            loader = PyPDFLoader(os.path.join(self.file_path, pdf_file))
            docs = loader.load()
            print(f"📄 {pdf_file}: {len(docs)} Seiten geladen")
            all_docs.extend(docs)
        return all_docs

    def chunkssplitter(self):
        if self.doc_splits:
            return self.doc_splits
        all_docs = self.pdfloader()
        print(f"📄 Geladene Dokumente: {len(all_docs)}")
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        self.doc_splits = splitter.split_documents(all_docs)
        print(f"🧩 Anzahl der Chunks nach Split: {len(self.doc_splits)}")
        return self.doc_splits

    def embedding(self):
        return OllamaEmbeddings(model='all-minilm')

    def persistent_clientChroma(self):
        client = chromadb.PersistentClient(path="chroma_db")
        print(f"🟢 ChromaDB läuft – Serverzeit: {client.heartbeat()}")
        return client.get_or_create_collection("collection_name")

    def add_only_new_docs_to_chroma(self):
        doc_splits = self.chunkssplitter()
        texts = [doc.page_content for doc in doc_splits]
        ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, text)) for text in texts]

        # IDs einmalig halten für Lookup (aber Reihenfolge im zip beibehalten!)
        id_set = set()
        filtered_ids = []
        filtered_texts = []
        for id_, text in zip(ids, texts):
            if id_ not in id_set:
                id_set.add(id_)
                filtered_ids.append(id_)
                filtered_texts.append(text)

        collection = self.persistent_clientChroma()
        
        try:
            existing = set(collection.get(ids=filtered_ids).get("ids", []))
        except Exception as e:
            print(f"⚠️ Fehler beim Abrufen bestehender IDs: {e}")
            existing = set()

        new_ids = []
        new_texts = []
        for id_, text in zip(filtered_ids, filtered_texts):
            if id_ not in existing:
                new_ids.append(id_)
                new_texts.append(text)

        print(f"🔄 Neue Dokumente erkannt: {len(new_texts)}")
        if not new_texts:
            print("✅ Keine neuen Inhalte zum Hinzufügen.")
            return

        embeddings = self.embedding()
        vectors = embeddings.embed_documents(new_texts)
        collection.add(ids=new_ids, documents=new_texts, embeddings=vectors)
        print(f"✅ {len(new_texts)} neue Chunks wurden zu Chroma hinzugefügt.")

    def vector_store_from_clientChroma(self):
        return Chroma(
            persist_directory="chroma_db",
            collection_name="collection_name",
            embedding_function=self.embedding()
        )

    def Retriever(self):
        store = self.vector_store_from_clientChroma()
        return store.as_retriever(
            search_type="similarity", search_kwargs={"k": self.k_int}
        )

    def promptTemplate(self):
        return PromptTemplate(
            template="""You are an assistant for question-answering tasks.
            Use the following documents to answer the question.
            If you don't know the answer, just say you don't know.
            Keep the answer concise.

            Question: {question}
            Documents: {documents}
            Answer:""",
            input_variables=["question", "documents"],
        )

    def llm(self):
        return ChatOllama(model=self.model, temperature=0.7)

    def ragchain(self):
        return self.promptTemplate() | self.llm() | StrOutputParser()

    def initializeChatbot(self, question):
        retriever = self.Retriever()
        docs = retriever.invoke(question)
        print(f"📥 Gefundene Dokumente: {len(docs)}")

        if not docs:
            return "⚠️ Ich konnte keine relevanten Inhalte in den Dokumenten finden."

        context = "\n\n".join([doc.page_content for doc in docs])
        return self.ragchain().invoke({"question": question, "documents": context})

    def debug_chroma_retriever(self, query: str):
        collection = self.persistent_clientChroma()
        print(f"📦 Dokumente in Chroma: {collection.count()}")
        sample = collection.get(limit=1)
        if sample.get("documents"):
            print("📄 Beispielinhalt:", sample["documents"][0][:300], "...")
        retriever = self.Retriever()
        found = retriever.invoke(query)
        print(f"🔎 {len(found)} Dokument(e) für '{query}' gefunden.")
        for i, doc in enumerate(found):
            print(f"\n--- [Dokument {i+1}] ---\n{doc.page_content[:300]}...")



# Bot instanziieren
prime_chatbot = PrimeChatbot(
    file_path="PDF_docs/",
    model="llama3.1:latest",
    chunk_size=400,
    chunk_overlap=40,
    k_int=5
)

# Nur neue Chunks hinzufügen
prime_chatbot.add_only_new_docs_to_chroma()

# Frage stellen
frage = "Difference between ETL and ELT in Data engeeniring"
antwort = prime_chatbot.initializeChatbot(frage)
print("🤖", antwort)
