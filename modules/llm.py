from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from langchain_ibm import WatsonxLLM
import os

def get_watsonx_llm():
    """Initializes Watsonx LLM for text generation."""
    credentials = Credentials(
        url=os.getenv("IBM_URL"),
        api_key=os.getenv("IBM_API_KEY")
    )

    parameters = {
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        GenParams.MIN_NEW_TOKENS: 1,
        GenParams.MAX_NEW_TOKENS: 500,
        GenParams.STOP_SEQUENCES: ["\n\n"],
    }

    return WatsonxLLM(
        model_id="ibm/granite-3-8b-instruct",
        url=credentials.get("url"),
        apikey=credentials.get("api_key"),
        project_id=os.getenv("IBM_PROJECT_ID"),
        params=parameters
    )
