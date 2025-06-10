import chromadb
import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama  # FREE local LLM
from llama_index.vector_stores.chroma import ChromaVectorStore

# Initialize FREE LOCAL components
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")  # Local embedding
llm = Ollama(model="mistral:7b-instruct-v0.2-q4_K_M", request_timeout=200.0)  # Local LLM

# 1. Load documents
reader = SimpleDirectoryReader(input_dir="data")
documents = reader.load_data()

# 2. Create vector store (local ChromaDB)
db = chromadb.PersistentClient(path="./data_chroma_db")

# Clear existing collection before processing new documents,
# especially if you changed the content of the documents
try:
    db.delete_collection("data")
    print("Existing collection deleted")
except:
    print("Creating new collection")

chroma_collection = db.get_or_create_collection("data")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# 3. Process documents with pipeline
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=20),
        embed_model
    ],
    vector_store=vector_store,
)
nodes = pipeline.run(documents=documents)

# 4. Create index
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

# 5. Create query engine (adjust for low RAM)
query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=2,  # Reduce to save RAM
    streaming=False,      # Disable streaming to reduce memory
    device_map="auto"     # Auto GPU offload
)

# Query example
response = query_engine.query("When did the main character die?")
print(f"Answer: {response}")
print(f"Sources: {response.get_formatted_sources()}")