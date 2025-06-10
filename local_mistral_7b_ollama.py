import chromadb
import os
import fitz  # PyMuPDF
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

# Initialize FREE LOCAL components
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
llm = Ollama(model="mistral:7b-instruct-v0.2-q4_K_M", request_timeout=400.0)

# Set up logging for OCR extraction
logging.basicConfig(filename='ocr_debug.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')


def extract_pdf_text(file_path: str) -> List[Document]:
    """Extract text from PDF including metadata using PyMuPDF"""
    documents = []
    try:
        with fitz.open(file_path) as pdf:
            full_text = ""
            for page_num in range(len(pdf)):
                page = pdf.load_page(page_num)
                text = page.get_text()
                full_text += f"\n--- PAGE {page_num + 1} ---\n{text}\n"

            # Log extracted content
            logging.info(f"PDF EXTRACTION SUCCESS: {file_path}\nExtracted {len(full_text)} characters")

            # Create document with metadata
            doc = Document(
                text=full_text,
                metadata={
                    "file_path": file_path,
                    "file_type": "PDF",
                    "pages": len(pdf)
                }
            )
            documents.append(doc)
    except Exception as e:
        logging.error(f"PDF EXTRACTION FAILED: {file_path} - {str(e)}")
    return documents


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
        
        # Create document with metadata and explicit OCR text
        doc = Document(
            text=f"OCR Text from image: {text}",  # Make OCR text more explicit
            metadata={
                "file_path": file_path,
                "file_type": "IMAGE",
                "ocr_engine": "Tesseract",
                "has_ocr_text": True,  # Add flag for OCR content
                "ocr_text": text  # Store original OCR text
            }
        )
        documents.append(doc)
    except Exception as e:
        logging.error(f"OCR FAILED: {file_path} - {str(e)}")
    return documents


def load_documents(data_dir: str) -> List[Document]:
    """Load documents from directory including PDFs, images, text and markdown files"""
    documents = []
    data_path = Path(data_dir)

    # Process all files in directory
    for file_path in data_path.glob('*'):
        file_extension = file_path.suffix.lower()

        # PDF files
        if file_extension == '.pdf':
            documents.extend(extract_pdf_text(str(file_path)))

        # Image files
        elif file_extension in ['.png', '.jpg', '.jpeg']:
            documents.extend(ocr_image_to_text(str(file_path)))

        # Text and markdown files
        elif file_extension in ['.txt', '.md']:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                doc = Document(
                    text=text,
                    metadata={
                        "file_path": str(file_path),
                        "file_type": "TEXT",
                        "extension": file_extension
                    }
                )
                documents.append(doc)
                logging.info(f"TEXT FILE LOADED: {file_path}, extracted text: {text[:200]}...")
            except Exception as e:
                logging.error(f"TEXT LOADING FAILED: {file_path} - {str(e)}")

    return documents


# ----------------- MAIN PROCESSING -----------------
def main():
    # 1. Load documents from all sources
    documents = load_documents("data")
    
    if not documents:
        print("No documents were loaded")
        return

    # Print loaded documents for debugging
    print("\nLoaded documents:")
    for doc in documents:
        print(f"Type: {doc.metadata.get('file_type')}, Path: {doc.metadata.get('file_path')}")
        if doc.metadata.get('file_type') == 'IMAGE':
            print(f"OCR Text: {doc.metadata.get('ocr_text', '')[:100]}...")

    # 2. Create vector store (local ChromaDB)
    db = chromadb.PersistentClient(path="./data_chroma_db")

    # Clear existing collection
    try:
        db.delete_collection("data")
        print("\nExisting collection deleted")
    except:
        print("\nCreating new collection")

    chroma_collection = db.get_or_create_collection("data")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # 3. Process documents with pipeline
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(
                chunk_size=1024,
                chunk_overlap=100,
                include_metadata=True,
                include_prev_next_rel=True
            ),
            embed_model
        ],
        vector_store=vector_store,
    )
    nodes = pipeline.run(documents=documents)
    print(f"\nProcessed {len(nodes)} document chunks")

    # 4. Create index
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
        show_progress=True
    )

    # 5. Create query engine with better context handling
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=5,
        streaming=False,
        device_map="auto",
        response_mode="tree_summarize"
    )

    # Test queries for different file types
    print("\nTesting queries on different file types:")


    # Image query 
    #img_response = query_engine.query("What information was extracted from the PNG image?")
    #print(f"\nImage Query - What information was extracted from the PNG image?")
    #print(f"Response: {img_response}")

    # PDF query
    #pdf_response = query_engine.query("What name appears on the certificate?")
    #print(f"\nPDF Query - What name appears on the certificate?")
    #print(f"Response: {pdf_response}")

    # Text query
    text_response = query_engine.query("Who is Anaximenes?")
    print(f"\nText Query - Who is Anaximenes?")
    print(f"Response: {text_response}")


if __name__ == "__main__":
    main()