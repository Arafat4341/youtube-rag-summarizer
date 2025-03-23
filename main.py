from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from langchain_ibm import WatsonxLLM
from ibm_watsonx_ai.foundation_models.utils import get_embedding_model_specs
from langchain_ibm import WatsonxEmbeddings
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import re


def get_video_id(url):    
    # Define a regular expression pattern to match YouTube video URLs
    # The pattern captures 11 alphanumeric characters (plus hyphen or underscore) after '?v='
    pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
    
    # Search the provided URL for the pattern
    match = re.search(pattern, url)
    
    # If a match is found, return the captured video ID group
    # Otherwise, return None
    return match.group(1) if match else None


def get_transcript(url):
    video_id = get_video_id(url)
    # Fetches the list of available transcripts for the given YouTube video
    srt = YouTubeTranscriptApi.list_transcripts(video_id)

    transcript = ""
    for i in srt:
        # Check if the transcript is auto-generated
        if i.is_generated:
            # If no transcript has been set yet, use the auto-generated one
            if len(transcript) == 0:
                transcript = i.fetch()
        else:
            # If a manually created transcript is found, use it (overrides auto-generated)
            transcript = i.fetch()

    return transcript

def process(transcript):
    # Initialize an empty string to accumulate processed text
    txt = ""

    # Iterate over each segment in the transcript
    for i in transcript:
        try:
            # Format the text and start time, then add to the accumulated string
            txt += f"Text: {i['text']} Start: {i['start']}\n"
        except:
            # If an error occurs (e.g., missing keys), skip the entry
            pass

    # Return the processed text
    return txt


# Retrieve the transcript for the specified YouTube video URL
transcript = get_transcript("https://www.youtube.com/watch?v=kEOCrtkLvEo&t=24s")

# Display the first 10 entries of the transcript
# Each entry is a dictionary containing 'text', 'start', and 'duration'
# print(transcript[:10])

processed_transcript = process(transcript)
res = processed_transcript[:100] # Display the first 100 characters of the processed transcript
print(res)