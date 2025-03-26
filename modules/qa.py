from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def generate_answer(llm, query, faiss_index):
    """Retrieves relevant context & generates an answer."""
    relevant_context = faiss_index.similarity_search(query, k=7)

    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are an expert assistant providing detailed answers based on the following video content.

        Relevant Video Context: {context}

        Question: {question}
        """
    )

    qa_chain = LLMChain(llm=llm, prompt=qa_prompt, verbose=True)
    return qa_chain.predict(context=relevant_context, question=query)
