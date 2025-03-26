from dotenv import load_dotenv
import os
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
from langchain_ibm import WatsonxEmbeddings

load_dotenv(os.path.join(os.path.dirname(__file__), '../conf/.env'))

def get_watsonx_embeddings():
    """Returns a Watsonx Embedding model."""
    return WatsonxEmbeddings(
        model_id=EmbeddingTypes.IBM_SLATE_30M_ENG.value,
        url=os.getenv("IBM_URL"),
        apikey=os.getenv("IBM_API_KEY"),
        project_id=os.getenv("IBM_PROJECT_ID")
    )
