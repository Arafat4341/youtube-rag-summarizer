from modules.video_processing import get_transcript, process_transcript
from modules.text_processing import split_text
from modules.embeddings import get_watsonx_embeddings
from modules.vectorstore import build_faiss_index, retrieve_similar
from modules.llm import get_watsonx_llm
from modules.summarizer import summarize_transcript
from modules.qa import generate_answer

def main():
    url = input("Enter YouTube video URL: ")
    
    # 1️⃣ Fetch & process transcript
    transcript = get_transcript(url)
    processed_transcript = process_transcript(transcript)

    # 2️⃣ Split transcript into chunks
    text_chunks = split_text(processed_transcript)

    # 3️⃣ Setup embeddings & FAISS index
    embeddings = get_watsonx_embeddings()
    faiss_index = build_faiss_index(text_chunks, embeddings)

    # 4️⃣ Load LLM model
    llm = get_watsonx_llm()

    # 5️⃣ Summarization
    summary = summarize_transcript(llm, processed_transcript)
    print("\n📌 Summary of the Video:\n", summary)

    # 6️⃣ Q&A
    while True:
        query = input("\nAsk a question (or type 'exit' to quit): ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = generate_answer(llm, query, faiss_index)
        print("\n🤖 Answer:", answer)

if __name__ == "__main__":
    main()
