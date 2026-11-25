from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    collection_name="sklearn_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# MMR retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,
        "lambda_mult": 0.7
    }
)

def retrieve(query):
    results = retriever.invoke(query)
    return results



if __name__ == "__main__":
    query = "Why should a model be evaluated on unseen data?"

    results = retrieve(query)

    print(f"\nQuestion: {query}")
    print(f"Retrieved chunks: {len(results)}")

    for i, doc in enumerate(results, start=1):
        print(f"\n--- Chunk {i} ---")
        print("Source:", doc.metadata.get("source"))
        print("Content:", doc.page_content[:500])