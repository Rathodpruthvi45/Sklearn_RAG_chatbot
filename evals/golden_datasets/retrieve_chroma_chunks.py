from langchain_chroma import Chroma
from deepeval.dataset import Golden
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


vectorstore = Chroma(
    collection_name="sklearn_docs",
    embedding_function=embedding,
    persist_directory="./chroma_db",
)