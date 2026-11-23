from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.data_ingesations.load_docs import load_web_page, URLS
from src.data_ingesations.chunking import create_chunks

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


vectorstore = Chroma(
    collection_name="sklearn_docs",
    embedding_function=embedding,
    persist_directory="./chroma_db",
)

documents = []
documents_ids = []

for doc_num, url in enumerate(URLS):

    text = load_web_page(url)
    chunks = create_chunks(text)

    for chunk_id, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={"source": url, "doc_nume": doc_num, "chunk_num": chunk_id},
        )

        documents.append(doc)
        documents_ids.append(f"doc_{doc_num}_chunk_{chunk_id}")

    vectorstore.add_documents(documents=documents, ids=documents_ids)
