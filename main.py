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
from dotenv import load_dotenv
import os


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
# res = processed_transcript[:100] # Display the first 100 characters of the processed transcript
# print(res)

# Indexing starts
# chunking the transcription

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,  # Maximum chunk size of 200 characters
    chunk_overlap=20  # Overlap of 20 characters between chunks
)

chunks = text_splitter.split_text(processed_transcript)
res = chunks[:10]  # Display the first 10 chunks
print(res)

# setting up watsonx model

# load_dotenv('config/.env')
load_dotenv(os.path.join(os.path.dirname(__file__), 'conf/.env'))

# Define the model ID for the Granite 8B Instruct Generation 3 model
model_id = "ibm/granite-3-8b-instruct"

# Set up the credentials needed to access the IBM Watson services
credentials = Credentials(
    url=os.getenv("IBM_URL"),
    api_key=os.getenv("IBM_API_KEY")
    # url='https://jp-tok.ml.cloud.ibm.com',
)

# Initialize the API client with the given credentials
client = APIClient(credentials)
project_id = os.getenv("IBM_PROJECT_ID")

# Defining Parameters for watsonx Model
parameters = {
    # Specifies the decoding method as greedy decoding
    # This means the model always chooses the most probable next token
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    
    # Sets the minimum number of new tokens to generate to 1
    # The model will always produce at least this many tokens
    GenParams.MIN_NEW_TOKENS: 1,
    
    # Sets the maximum number of new tokens to generate to 500
    # The model will stop generating after reaching this limit
    GenParams.MAX_NEW_TOKENS: 500,
    
    # Defines sequences that will cause the generation to stop
    # In this case, generation will stop when encountering two consecutive newlines
    GenParams.STOP_SEQUENCES: ["\n\n"],
}

# Initializing watsonx LLM
watsonx_granite = WatsonxLLM(
    model_id=model_id,
    url=credentials.get("url"),
    apikey=credentials.get("api_key"),
    project_id=project_id,
    params=parameters
)

# initializing embedding model
# Fetch specifications for available embedding models from the Watson service
get_embedding_model_specs(credentials.get('url'))

# Part 1: Create Embedding Model
# Set up the WatsonxEmbeddings object
embeddings = WatsonxEmbeddings(
    # Specifies the ID of the embedding model to be used
    # In this case, it's using the IBM SLATE 30M English model
    model_id=EmbeddingTypes.IBM_SLATE_30M_ENG.value,
    
    # The URL endpoint for the Watson service
    # This is retrieved from the credentials dictionary
    url=credentials["url"],
    apikey=credentials["api_key"],
    
    # The ID of the project in which this embedding model will operate
    # This helps in organizing and managing different model instances
    project_id=project_id
)

# Implementing FAISS for Similarity Search
# INDEXING - embedding and storing
# (from_texts first does the embedding, and then stores them in FAISS index)
faiss_index = FAISS.from_texts(chunks, embeddings)

# perform simillarity search
# Define the query string we want to search for
query = "Which company they were talking about?"

# Perform a similarity search on the FAISS index
# The search returns the 'k' most similar chunks to the query
# In this case, k=3, so it returns the top 3 most similar results
results = faiss_index.similarity_search(query, k=3)

# Iterate through the results and print each one
for result in results:
    print(result)


# Summarizing the Transcript with LLMChain
# defining promt template
prompt = PromptTemplate(
    # Specify the input variables that will be used in the template
    input_variables=["transcript"],
    
    # Define the actual template string
    template="""
        Summarize the following YouTube video transcript in terms of paragraph:

        {transcript}

        Your summary should have concise summary in terms of paragraph. Ignore any timestamps.
    """
)