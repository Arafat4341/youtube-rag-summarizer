# 🎥 **YouTube RAG Summarizer & Q&A Assistant**  

_Automated video summarization and Q&A system powered by **LangChain, FAISS, and IBM Watsonx LLM**_  

---

## 📌 **Overview**  

This project simplifies **video content analysis** by automatically:  
✅ Extracting transcripts from **YouTube videos**  
✅ Generating **concise summaries**  
✅ Answering **user queries based on video content**  

Instead of watching **long videos**, this AI-powered assistant allows you to **quickly understand** the key takeaways and retrieve **precise answers** to your questions.

---

## 🚀 **Features**  

🔹 Fetches **YouTube transcripts** automatically using `youtube-transcript-api`  
🔹 **Splits, embeds & indexes** transcript chunks for retrieval  
🔹 Uses **FAISS** for fast similarity search on transcript data  
🔹 Supports **Q&A system** using **Watsonx LLM + RAG**  
🔹 Built with **modular & scalable architecture**  

---

## 🏗️ **Project Workflow**  

1️⃣ **Fetch Transcript** – Extracts full YouTube video transcript  
2️⃣ **Process & Split** – Cleans text & divides into smaller chunks  
3️⃣ **Embed & Store** – Converts text into embeddings & stores in **FAISS vector DB**  
4️⃣ **Summarization** – Uses **Watsonx LLM** to summarize key insights  
5️⃣ **Q&A System** – Retrieves relevant transcript parts & generates **context-aware responses**  

---

## 🛠️ **Tech Stack**  

| Component                     | Technology Used                            |
|--------------------------------|-------------------------------------------|
| **Transcript Retrieval**       | `youtube-transcript-api`                   |
| **Text Processing**            | `LangChain` (RecursiveCharacterTextSplitter) |
| **Vector Storage**             | `FAISS` (Facebook AI Similarity Search)   |
| **Embedding Model**            | `Watsonx Embeddings (IBM SLATE 30M ENG)`  |
| **LLM (Text Generation)**      | `Watsonx LLM (Granite 8B Instruct)`       |
| **Retrieval-Augmented QA**     | `LangChain RetrievalQA Chain`             |
| **Environment Management**     | `dotenv` for API keys                      |
| **Programming Language**       | `Python`                                  |

---

## 📂 **Project Structure**  

```
youtube-rag-summarizer/
├── data/                         # (Optional) Store transcript text files
├── config/
│   └── .env                      # API keys & Watsonx credentials
├── modules/
│   ├── video_processing.py       # Fetch & process YouTube transcripts
│   ├── text_processing.py        # Splitting & preprocessing text
│   ├── embeddings.py             # Watsonx embeddings setup
│   ├── vectorstore.py            # FAISS vector storage & retrieval
│   ├── llm.py                    # Watsonx LLM setup
│   ├── summarizer.py             # Summarization logic
│   ├── qa.py                     # Q&A retrieval & response generation
├── main.py                       # Orchestrates everything
├── requirements.txt              # Dependencies
└── README.md                     # Documentation
```

---

## 🔧 **Setup & Installation**  

### 1️⃣ **Clone the Repository**  
```bash
git clone https://github.com/yourusername/youtube-rag-summarizer.git
cd youtube-rag-summarizer
```

### 2️⃣ **Install Dependencies**  
```bash
pip install -r requirements.txt
```

### 3️⃣ **Set Up Environment Variables**  
Create a `.env` file inside `config/` and add your **IBM Watsonx API credentials**:  

```ini
IBM_URL=your_ibm_url
IBM_API_KEY=your_ibm_api_key
IBM_PROJECT_ID=your_ibm_project_id
```

---

## 🎬 **How to Run the Project**  

1️⃣ **Run the script**  
```bash
python main.py
```
2️⃣ **Enter a YouTube video URL**  
```
Enter YouTube video URL: https://www.youtube.com/watch?v=your_video_id
```
3️⃣ **Get a Summary of the Video**  
```
📌 Summary of the Video:
This video explains how RAG (Retrieval-Augmented Generation) helps improve LLM accuracy...
```
4️⃣ **Ask Questions About the Video**  
```
Ask a question (or type 'exit' to quit): What was the main topic discussed?

🤖 Answer: The video primarily focused on AI-powered summarization using RAG systems.
```

---

## ✨ **Key Learnings & Insights**  

✔ **Learned to extract and process YouTube transcripts automatically**  
✔ **Implemented FAISS for efficient similarity search**  
✔ **Integrated IBM Watsonx LLM for contextual text generation**  
✔ **Developed a modular RAG-based system for real-world applications**  

---

## 🔥 **Future Improvements**  

🚀 **Enhance UI** – Convert this into a **Streamlit/Flask web app**  
🚀 **Support More LLMs** – Switch from **Watsonx to Hugging Face/OpenAI**  
🚀 **Multi-Video Analysis** – Compare & summarize **multiple videos at once**  

---

## 📜 **License**  

This project is licensed under the **MIT License** – Feel free to modify and use it!  

---

## 👋 **Connect & Contribute**  

Want to improve this project? **Fork it, open a PR, or reach out!**  
🔗 [GitHub Repo](https://github.com/yourusername/youtube-rag-summarizer)  

Always happy to discuss **RAG, AI-powered assistants, and automation!** 🚀  

#LLM #RAG #LangChain #FAISS #Watsonx #YouTubeAI #AI #MachineLearning #Python  
