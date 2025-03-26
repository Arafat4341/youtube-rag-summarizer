from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(text, chunk_size=200, chunk_overlap=20):
    """Splits text into chunks for indexing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)
