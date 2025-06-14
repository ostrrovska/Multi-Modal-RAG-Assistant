import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import threading
import queue
import os
from pathlib import Path
import logging
from main import (
    load_documents, VectorStoreIndex, ChromaVectorStore, 
    IngestionPipeline, SentenceSplitter, Ollama, 
    HuggingFaceEmbedding, chromadb
)

class RAGApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Modal RAG Assistant")
        self.root.geometry("1200x800")
        
        # Initialize components
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.llm = Ollama(model="llama3", request_timeout=400.0)
        self.query_engine = None
        self.message_queue = queue.Queue()
        
        # Set up logging
        logging.basicConfig(
            filename='ui_debug.log',
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        
        self.setup_ui()
        self.process_queue()
        
    def setup_ui(self):
        # Create main frame with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Data Directory Selection
        dir_frame = ttk.LabelFrame(main_frame, text="Data Directory", padding="5")
        dir_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.dir_path = tk.StringVar()
        ttk.Entry(dir_frame, textvariable=self.dir_path, width=70).grid(row=0, column=0, padx=5)
        ttk.Button(dir_frame, text="Browse", command=self.browse_directory).grid(row=0, column=1, padx=5)
        
        # Process Button
        self.process_btn = ttk.Button(dir_frame, text="Process Documents", command=self.start_processing)
        self.process_btn.grid(row=0, column=2, padx=5)
        
        # Progress Frame
        progress_frame = ttk.LabelFrame(main_frame, text="Processing Progress", padding="5")
        progress_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Document Loading Progress
        ttk.Label(progress_frame, text="Document Loading:").grid(row=0, column=0, sticky=tk.W)
        self.doc_progress = ttk.Progressbar(progress_frame, length=400, mode='determinate')
        self.doc_progress.grid(row=0, column=1, padx=5, pady=2)
        
        # Embedding Progress
        ttk.Label(progress_frame, text="Embedding Generation:").grid(row=1, column=0, sticky=tk.W)
        self.embed_progress = ttk.Progressbar(progress_frame, length=400, mode='determinate')
        self.embed_progress.grid(row=1, column=1, padx=5, pady=2)
        
        # Status Messages
        self.status_text = scrolledtext.ScrolledText(progress_frame, height=6, width=70)
        self.status_text.grid(row=2, column=0, columnspan=2, pady=5)
        
        # Query Section
        query_frame = ttk.LabelFrame(main_frame, text="Query Section", padding="5")
        query_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Query Input
        ttk.Label(query_frame, text="Enter your query:").grid(row=0, column=0, sticky=tk.W)
        self.query_input = ttk.Entry(query_frame, width=70)
        self.query_input.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(query_frame, text="Ask", command=self.process_query).grid(row=0, column=2)
        
        # Response Area
        ttk.Label(query_frame, text="Response:").grid(row=1, column=0, sticky=tk.W)
        self.response_text = scrolledtext.ScrolledText(query_frame, height=15, width=70)
        self.response_text.grid(row=2, column=0, columnspan=3, pady=5)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
    def browse_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.dir_path.set(directory)
            
    def update_status(self, message):
        self.message_queue.put(("status", message))
        logging.info(message)
        
    def update_progress(self, progress_type, value):
        self.message_queue.put(("progress", (progress_type, value)))
        
    def process_queue(self):
        try:
            while True:
                message_type, content = self.message_queue.get_nowait()
                if message_type == "status":
                    self.status_text.insert(tk.END, content + "\n")
                    self.status_text.see(tk.END)
                elif message_type == "progress":
                    progress_type, value = content
                    if progress_type == "doc":
                        self.doc_progress["value"] = value
                    elif progress_type == "embed":
                        self.embed_progress["value"] = value
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_queue)
            
    def start_processing(self):
        if not self.dir_path.get():
            self.update_status("Please select a data directory first!")
            return
            
        self.process_btn["state"] = "disabled"
        self.doc_progress["value"] = 0
        self.embed_progress["value"] = 0
        self.status_text.delete(1.0, tk.END)
        
        # Start processing in a separate thread
        thread = threading.Thread(target=self.process_documents)
        thread.daemon = True
        thread.start()
        
    def process_documents(self):
        try:
            self.update_status("Loading documents...")
            documents = load_documents(self.dir_path.get())
            self.update_progress("doc", 50)
            
            if not documents:
                self.update_status("No documents were loaded!")
                return
                
            self.update_status(f"Loaded {len(documents)} documents")
            self.update_progress("doc", 100)
            
            # Initialize ChromaDB
            self.update_status("Initializing vector store...")
            db = chromadb.PersistentClient(path="./data_chroma_db")
            try:
                db.delete_collection("data")
                self.update_status("Existing collection deleted")
            except:
                self.update_status("Creating new collection")
                
            chroma_collection = db.get_or_create_collection("data")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            
            # Process documents
            self.update_status("Processing documents...")
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
            
            nodes = pipeline.run(documents=documents)
            self.update_progress("embed", 50)
            
            self.update_status(f"Processed {len(nodes)} document chunks")
            
            # Create index and query engine
            self.update_status("Creating index...")
            index = VectorStoreIndex.from_vector_store(
                vector_store,
                embed_model=self.embed_model,
                show_progress=True
            )
            
            self.query_engine = index.as_query_engine(
                llm=self.llm,
                similarity_top_k=5,
                streaming=False,
                device_map="auto",
                response_mode="tree_summarize"
            )
            
            self.update_progress("embed", 100)
            self.update_status("Processing complete! You can now ask questions.")
            self.process_btn["state"] = "normal"
            
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            logging.error(f"Processing error: {str(e)}")
            self.process_btn["state"] = "normal"
            
    def process_query(self):
        if not self.query_engine:
            self.update_status("Please process documents first!")
            return
            
        query = self.query_input.get()
        if not query:
            return
            
        self.response_text.delete(1.0, tk.END)
        self.response_text.insert(tk.END, "Processing query...\n")
        
        def run_query():
            try:
                response = self.query_engine.query(query)
                self.response_text.delete(1.0, tk.END)
                self.response_text.insert(tk.END, str(response))
                logging.info(f"Query processed: {query}")
            except Exception as e:
                self.response_text.delete(1.0, tk.END)
                self.response_text.insert(tk.END, f"Error: {str(e)}")
                logging.error(f"Query error: {str(e)}")
                
        thread = threading.Thread(target=run_query)
        thread.daemon = True
        thread.start()

if __name__ == "__main__":
    root = tk.Tk()
    app = RAGApplication(root)
    root.mainloop() 