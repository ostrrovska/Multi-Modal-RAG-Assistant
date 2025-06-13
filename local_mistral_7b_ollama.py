import chromadb
import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import logging
from pathlib import Path
from typing import List, Optional
from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from hf_BLIP import image_to_document
from pytesseract_OCR import ocr_image_to_text

# Initialize FREE LOCAL components
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
llm = Ollama(model="mistral:7b-instruct-v0.2-q4_K_M", request_timeout=400.0)

# Set up logging for OCR extraction
logging.basicConfig(filename='debug.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

def generate_image_caption(image_path: str) -> List[Document]:
    """Generate caption for image using BLIP model"""
    # This function now passes through the list of documents from image_to_document.
    return image_to_document(image_path)

def generate_image_text_ocr(image_path: str) -> List[Document]:
    """Generate text from image using OCR"""
    return ocr_image_to_text(image_path)

def process_image(caption_docs: List[Document], ocr_docs: List[Document]) -> List[Document]:
    """Combine image caption and OCR text into a single document."""
    
    caption_doc: Optional[Document] = caption_docs[0] if caption_docs else None
    ocr_doc: Optional[Document] = ocr_docs[0] if ocr_docs else None

    # If neither process yielded a result, there's nothing to do.
    if not caption_doc and not ocr_doc:
        return []

    # Safely get text from each document
    caption_text = caption_doc.text if caption_doc else ""
    ocr_text = ocr_doc.text if ocr_doc else ""
    
    # Safely get metadata, preferring the caption document's path
    file_path = ""
    if caption_doc:
        file_path = caption_doc.metadata.get("file_path", "")
    elif ocr_doc:
        file_path = ocr_doc.metadata.get("file_path", "")

    # Extract filename from path for inclusion in text
    filename = os.path.basename(file_path)
    
    # Combine the text from both sources, including the filename
    combined_text = (
        f"Content of file: {filename}\n"
        f"Caption: {caption_text}\n"
        f"OCR Text: {ocr_text}"
    ).strip()

    # Create a new, combined document
    return [Document(
        text=combined_text,
        metadata={
            "file_path": file_path,
            "file_type": "IMAGE",
            "caption": caption_text,
            "ocr_text": ocr_text,
        }
    )]

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
            caption_docs = generate_image_caption(str(file_path))
            ocr_docs = generate_image_text_ocr(str(file_path))
            documents.extend(process_image(caption_docs, ocr_docs))

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
    documents = load_documents("data")
    if not documents:
        print("No documents were loaded")
        return

    print("\nLoaded documents:")
    for doc in documents:
        print(f"Type: {doc.metadata.get('file_type')}, Path: {doc.metadata.get('file_path')}")
        if doc.metadata.get('file_type') == 'IMAGE':
            print(f"OCR Text: {doc.metadata.get('ocr_text', '')[:100]}...")

    db = chromadb.PersistentClient(path="./data_chroma_db")

    try:
        db.delete_collection("data")
        print("\nExisting collection deleted")
    except:
        print("\nCreating new collection")

    chroma_collection = db.get_or_create_collection("data")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

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

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
        show_progress=True
    )

    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=5,
        streaming=False,
        device_map="auto",
        response_mode="tree_summarize"
    )

    print("\nTesting queries on different file types:")

    img_response3 = query_engine.query("Which object does the woman hold on the image?")
    print(f"\nImage Query - Which object does the woman hold on the image?")
    print(f"Response: {img_response3}")

    # Image query
    img_response2 = query_engine.query("What is the article about?")
    print(f"\nImage Query - What is the article about?")
    print(f"Response: {img_response2}")

    # PDF query
    pdf_response = query_engine.query("What name appears on the certificate?")
    print(f"\nPDF Query - What name appears on the certificate?")
    print(f"Response: {pdf_response}")

    # Text query
    text_response = query_engine.query("Anaximenes's philosophy was centered on which theory?")
    print(f"\nText Query - Anaximenes's philosophy was centered on which theory?")
    print(f"Response: {text_response}")



if __name__ == "__main__":
    main()