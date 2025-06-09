import chromadb
import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.vector_stores.chroma import ChromaVectorStore

# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Initialize Hugging Face components
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
    api_key=hf_token
)

# 1. Load documents
reader = SimpleDirectoryReader(input_dir="data")
documents = reader.load_data()

# 2. Create vector store
db = chromadb.PersistentClient(path="./alfred_chroma_db")
chroma_collection = db.get_or_create_collection("alfred")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# 3. Process documents with ingestion pipeline
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=20),
        embed_model
    ],
    vector_store=vector_store,
)

# Run the pipeline synchronously
nodes = pipeline.run(documents=documents)

# 4. Create index
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

# 5. Create query engine
query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=3,
    response_mode="compact"
)

# Example query
response = query_engine.query("What is the main topic of my document?")
print(f"Answer: {response}")
print(f"Source: {response.get_formatted_sources()}")

