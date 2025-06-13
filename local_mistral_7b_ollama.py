import asyncio
import chromadb
import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import logging
from pathlib import Path
from typing import List, Dict, Any
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from hf_BLIP import image_to_document
from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
from llama_index.core.workflow import Context

class DocumentProcessor:
    
    """Handles document processing and indexing for different file types"""
    
    def __init__(self, data_dir: str = "data", db_path: str = "./data_chroma_db"):
        # Initialize components
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.llm = Ollama(model="mistral:7b-instruct-v0.2-q4_K_M", request_timeout=400.0)
        self.data_dir = data_dir
        self.db_path = db_path
        self.index = None
        
        # Set up logging
        logging.basicConfig(
            filename='document_processor.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    async def _create_document(self, text: str, metadata: Dict[str, Any]) -> Document:
        """Helper method to create documents with consistent metadata structure"""
        return Document(text=text, metadata=metadata)

    async def process_text_file(self, file_path: str) -> List[Document]:
        """Process text and markdown files"""
        documents = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            doc = await self._create_document(
                text=text,
                metadata={
                    "file_path": file_path,
                    "file_type": "TEXT",
                    "extension": Path(file_path).suffix.lower()
                }
            )
            documents.append(doc)
            self.logger.info(f"TEXT FILE LOADED: {file_path}, extracted text: {text[:200]}...")
        except Exception as e:
            self.logger.error(f"TEXT LOADING FAILED: {file_path} - {str(e)}")
        return documents

    async def generate_image_caption(self, image_path: str) -> List[Document]:
        """Generate caption for image using BLIP model"""
        documents = []
        try:
            # Handle both full paths and filenames
            if not os.path.exists(image_path):
                image_path = os.path.join(self.data_dir, image_path)
                
            if not os.path.exists(image_path):
                self.logger.error(f"Image file not found: {image_path}")
                return documents
                
            caption = image_to_document(image_path)
            if caption:
                doc = await self._create_document(
                    text=f"Caption for {os.path.basename(image_path)}: {caption}",
                    metadata={
                        "file_path": image_path,
                        "file_type": "IMAGE",
                        "has_caption": True
                    }
                )
                documents.append(doc)
            else:
                self.logger.warning(f"No caption could be generated for: {image_path}")
        except Exception as e:
            self.logger.error(f"IMAGE CAPTION FAILED: {str(e)}")
        return documents

    async def extract_pdf_text(self, file_path: str) -> List[Document]:
        """Extract text from PDF including metadata using PyMuPDF"""
        documents = []
        try:
            with fitz.open(file_path) as pdf:
                full_text = ""
                for page_num in range(len(pdf)):
                    page = pdf.load_page(page_num)
                    text = page.get_text()
                    full_text += f"\n--- PAGE {page_num + 1} ---\n{text}\n"

                self.logger.info(f"PDF EXTRACTION SUCCESS: {file_path}\nExtracted {len(full_text)} characters")

                doc = await self._create_document(
                    text=full_text,
                    metadata={
                        "file_path": file_path,
                        "file_type": "PDF",
                        "pages": len(pdf)
                    }
                )
                documents.append(doc)
        except Exception as e:
            self.logger.error(f"PDF EXTRACTION FAILED: {file_path} - {str(e)}")
        return documents

    async def ocr_image_to_text(self, file_path: str) -> List[Document]:
        """Perform OCR on image files using pytesseract"""
        documents = []
        try:
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)
            
            self.logger.info(f"OCR SUCCESS: {file_path}\nExtracted text:\n{text[:200]}...")
            
            doc = await self._create_document(
                text=f"OCR Text from image: {text}",
                metadata={
                    "file_path": file_path,
                    "file_type": "IMAGE",
                    "ocr_engine": "Tesseract",
                    "has_ocr_text": True,
                    "ocr_text": text
                }
            )
            documents.append(doc)
        except Exception as e:
            self.logger.error(f"OCR FAILED: {file_path} - {str(e)}")
        return documents

    async def load_documents(self) -> List[Document]:
        """Load documents from directory including PDFs, images, text and markdown files"""
        documents = []
        data_path = Path(self.data_dir)

        for file_path in data_path.glob('*'):
            file_extension = file_path.suffix.lower()
            file_path_str = str(file_path)

            if file_extension == '.pdf':
                docs = await self.extract_pdf_text(file_path_str)
                documents.extend(docs)
            elif file_extension in ['.png', '.jpg', '.jpeg']:
                # First, do OCR
                ocr_docs = await self.ocr_image_to_text(file_path_str)
                documents.extend(ocr_docs)
                # Then, generate caption
                caption_docs = await self.generate_image_caption(file_path_str)
                if caption_docs:
                    documents.extend(caption_docs)
            elif file_extension in ['.txt', '.md']:
                text_docs = await self.process_text_file(file_path_str)
                documents.extend(text_docs)

        return documents

    async def create_index(self) -> VectorStoreIndex:
        """Create and return a vector store index from processed documents"""
        # Load documents
        documents = await self.load_documents()
        if not documents:
            self.logger.error("No documents were loaded")
            return None

        # Create vector store
        db = chromadb.PersistentClient(path=self.db_path)
        try:
            db.delete_collection("data")
            self.logger.info("Existing collection deleted")
        except:
            self.logger.info("Creating new collection")

        chroma_collection = db.get_or_create_collection("data")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # Process documents with pipeline
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(
                    chunk_size=1024,
                    chunk_overlap=100,
                    include_metadata=True,
                    include_prev_next_rel=True
                ),
                self.embed_model
            ],
            vector_store=vector_store,
        )
        nodes = await pipeline.arun(documents=documents)
        self.logger.info(f"Processed {len(nodes)} document chunks")

        # Create and return index
        self.index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=self.embed_model,
            show_progress=True
        )
        return self.index

    def create_query_engine(self):
        """Create and return a query engine from the index"""
        if not self.index:
            raise ValueError("Index not created. Call create_index first.")
        return self.index.as_query_engine(
            llm=self.llm,
            similarity_top_k=5,
            streaming=False,
            response_mode="tree_summarize"
        )

# Initialize document processor
processor = DocumentProcessor()

# Specialized agents
pdf_agent = ReActAgent(
    name="pdf_agent",
    description="Extracts and searches text from PDFs in the data directory",
    system_prompt="""You are a PDF processing expert. When asked about PDF content:
    1. First check if the PDF exists in the data directory
    2. Use the extract_pdf_text tool to get all text
    3. Search through the text for the requested information
    4. Return the relevant excerpt with page number
    
    Available tool: extract_pdf_text(file_path: str) -> List[Document]""",
    tools=[processor.extract_pdf_text],
    llm=processor.llm
)

image_text_agent = ReActAgent(
    name="image_text_agent",
    description="Extracts text from images using OCR",
    system_prompt="""You are an OCR expert. When asked about text in images:
    1. First check if the image exists in the data directory
    2. Use the ocr_image_to_text tool to extract any text from the image
    3. Return the extracted text with the image filename
    
    Available tool: ocr_image_to_text(file_path: str) -> List[Document]""",
    tools=[processor.ocr_image_to_text],
    llm=processor.llm
)

image_caption_agent = ReActAgent(
    name="image_caption_agent",
    description="Generates captions for images",
    system_prompt="""You are an image captioning expert. When asked about image captions:
    1. First check if the image exists in the data directory
    2. Use the generate_image_caption tool to create a descriptive caption
    3. Return the caption with the image filename
    
    Available tool: generate_image_caption(file_path: str) -> List[Document]""",
    tools=[processor.generate_image_caption],
    llm=processor.llm
)

text_agent = ReActAgent(
    name="text_agent",
    description="Processes text and markdown files",
    system_prompt="""You are a text processing expert. When asked about text files:
    1. First check if the text file exists in the data directory
    2. Use the process_text_file tool to read the file contents
    3. Return the relevant text with the filename
    
    Available tool: process_text_file(file_path: str) -> List[Document]""",
    tools=[processor.process_text_file],
    llm=processor.llm
)

root_agent = ReActAgent(
    name="root_agent",
    description="Coordinates document processing and querying",
    system_prompt="""You are the coordinator for a document processing system. Your tasks:
    1. For PDF questions: Delegate to pdf_agent
    2. For image caption requests: Delegate to image_caption_agent
    3. For image text content requests: Delegate to image_text_agent
    4. For text or markdown file queries: Delegate to text_agent
    5. For other document queries: Use the query_engine from context
    6. If unsure: First ensure documents are loaded, then use query_engine
    
    Available tools:
    - load_documents() -> List[Document]
    - query_engine (from context)""",
    tools=[processor.load_documents],
    llm=processor.llm
)

# Enhanced workflow setup
workflow = AgentWorkflow(
    agents=[pdf_agent, image_text_agent, image_caption_agent, text_agent, root_agent],
    root_agent="root_agent",
    state_prompt="Current state: {state}. User message: {msg}"
)

async def main():
    try:
        ctx = Context(workflow)
        
        print("\nCreating search index...")
        index = await processor.create_index()
        if not index:
            print("Failed to create index - no documents were processed")
            return

        # Store components in context
        await ctx.set("query_engine", processor.create_query_engine())
        await ctx.set("processor", processor)
        
        print("\n🚀 Document Query System is ready!")
        while True:
            query = input("\nQuestion: ").strip()
            if query.lower() in ['exit', 'quit']:
                break

            try:
                # First try with workflow
                response = await workflow.run(user_msg=query, ctx=ctx)
                
                #if "error" in str(response).lower() or "cannot" in str(response).lower():
                    # Fallback to direct query
                #    query_response = processor.create_query_engine().query(query)
                #    print(f"\n[Direct Query Response] {query_response}")
                #else:
                print(f"\n[Agent Response] {response}")
                    
            except Exception as e:
                print(f"\nError processing query: {str(e)}")

    except Exception as e:
        print(f"System error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())