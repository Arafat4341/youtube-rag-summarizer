from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def summarize_transcript(llm, transcript):
    """Summarizes a YouTube transcript."""
    prompt = PromptTemplate(
        input_variables=["transcript"],
        template="""
        Summarize the following YouTube video transcript in paragraphs:

        {transcript}

        Provide a concise summary. Ignore timestamps.
        """
    )

    summarization_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    return summarization_chain.predict(transcript=transcript)
