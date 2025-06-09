from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
import os
from dotenv import load_dotenv
from nltk.corpus.reader import documents

# Load the .env file
load_dotenv()

# Retrieve HF_TOKEN from the environment variables
hf_token = os.getenv("HF_TOKEN")

llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
    temperature=0.7,
    max_tokens=100,
    token=hf_token,
)

response = llm.complete("Hello, how are you?")
print(response)



