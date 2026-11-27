from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load the same embedding model used during indexing
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Connect to our existing ChromaDB
vectorstore = Chroma(
    collection_name="sklearn_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

# 3. Create a retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})


# 4. Function to retrieve chunks for a query
def retrieve(query, top_k=5):
    results = vectorstore.similarity_search(query, k=top_k)
    return results


# 5. Run a quick test
if __name__ == "__main__":
    query = "Why should a model be evaluated on unseen data?"

    results = retrieve(query, top_k=5)

    print("\nUSER QUESTION:")
    print(query)

    print("\nRETRIEVED CHUNKS:")

    for i, doc in enumerate(results, start=1):
        print("\n" + "=" * 70)
        print(f"CHUNK {i}")
        print("=" * 70)

        print("SOURCE:", doc.metadata.get("source"))
        print("CHUNK NUMBER:", doc.metadata.get("chunk_num"))
        print("\nCONTENT:")
        print(doc.page_content)
