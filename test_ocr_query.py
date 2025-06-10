import chromadb
import pytesseract
from PIL import Image
import logging
from pathlib import Path
from typing import List
from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

# Initialize components
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
llm = Ollama(model="mistral:7b-instruct-v0.2-q4_K_M", request_timeout=400.0)

# Set up logging
logging.basicConfig(filename='test_ocr_debug.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

def ocr_image_to_text(file_path: str) -> List[Document]:
    """Perform OCR on image files using pytesseract"""
    documents = []
    try:
        # Open image file
        img = Image.open(file_path)
        
        # Perform OCR
        text = pytesseract.image_to_string(img)
        
        # Log extracted content
        logging.info(f"OCR SUCCESS: {file_path}\nExtracted text:\n{text[:200]}...")
        
        # Create document with metadata
        doc = Document(
            text=text,
            metadata={
                "file_path": file_path,
                "file_type": "IMAGE",
                "ocr_engine": "Tesseract"
            }
        )
        documents.append(doc)
    except Exception as e:
        logging.error(f"OCR FAILED: {file_path} - {str(e)}")
    return documents

def main():
    # 1. Test OCR on a single image
    image_path = "data/image.png"
    documents = ocr_image_to_text(image_path)
    
    if not documents:
        print("No documents were extracted from the image")
        return
        
    # 2. Create vector store
    db = chromadb.PersistentClient(path="./test_chroma_db")
    
    # Clear existing collection
    try:
        db.delete_collection("test_data")
        print("Existing collection deleted")
    except:
        print("Creating new collection")
        
    chroma_collection = db.get_or_create_collection("test_data")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # 3. Process documents with pipeline
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=1024, chunk_overlap=100, include_metadata=True),
            embed_model
        ],
        vector_store=vector_store,
    )
    nodes = pipeline.run(documents=documents)
    print(f"Processed {len(nodes)} document chunks")
    
    # 4. Create index
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    
    # 5. Create query engine
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=5,
        streaming=False,
        device_map="auto",
    )
    
    # 6. Test queries
    print("\nTesting queries on OCR content:")
    
    # Query 1: Direct content question
    response1 = query_engine.query("What text is in the image?")
    print(f"\nQuery 1 - What text is in the image?")
    print(f"Response: {response1}")
    
    
    # Query 2: General question
    response3 = query_engine.query("What is the purpose of the text in the image?")
    print(f"\nQuery 2 - What is the purpose of the text in the image?")
    print(f"Response: {response3}")

if __name__ == "__main__":
    main() 