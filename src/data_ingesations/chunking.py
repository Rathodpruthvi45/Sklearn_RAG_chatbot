from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=150, separators=["\n\n", "\n", ".", " ", ""]
    )

    chunk = splitter.split_text(text)

    return chunk
