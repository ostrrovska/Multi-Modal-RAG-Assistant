# Multi-Modal RAG Assistant

A powerful Retrieval-Augmented Generation (RAG) system that can process and query multiple types of documents including text, PDFs, and images. The system uses advanced AI models for document understanding, including OCR for text extraction from images and BLIP for image captioning.

## Features

- **Multi-Modal Document Processing**
  - Text files (.txt, .md)
  - PDF documents
  - Images (.png, .jpg, .jpeg) with:
    - OCR text extraction using Tesseract
    - Image captioning using BLIP model

- **Advanced RAG Pipeline**
  - Document chunking with configurable size and overlap
  - Vector embeddings using BGE-small-en model
  - ChromaDB for vector storage
  - LLM-powered querying using Ollama

- **User-Friendly GUI**
  - Document directory selection
  - Real-time processing progress
  - Interactive query interface
  - Detailed status updates and logging

## Prerequisites

- Python 3.8 or higher
- Tesseract OCR engine installed on your system
- CUDA-capable GPU (recommended for better performance)
- Ollama installed on your system

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Multi-Modal-RAG-Assistant.git
cd Multi-Modal-RAG-Assistant
```

2. Install the required Python packages:
```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR:
   - Windows: Download and install from [Tesseract GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)
   - Linux: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`

4. Install Ollama:
   - Windows: Download and install from [Ollama's official website](https://ollama.ai/download)
   - Linux: 
     ```bash
     curl https://ollama.ai/install.sh | sh
     ```
   - macOS: 
     ```bash
     curl https://ollama.ai/install.sh | sh
     ```

5. Download the required LLM model:
   ```bash
   ollama pull llama3
   ```
   Note: The first time you run this command, it will download the model which may take some time depending on your internet connection.

## Usage

1. Start the application:
```bash
python rag_ui.py
```

2. Using the GUI:
   - Click "Browse" to select a directory containing your documents
   - Click "Process Documents" to start the ingestion pipeline
   - Wait for the processing to complete
   - Enter your query in the query section
   - Click "Ask" to get the response

3. Supported Document Types:
   - Text files (.txt, .md)
   - PDF documents
   - Images (.png, .jpg, .jpeg)

## Project Structure

- `rag_ui.py`: Main GUI application
- `main.py`: Core RAG pipeline implementation
- `hf_BLIP.py`: Image captioning using BLIP model
- `pytesseract_OCR.py`: OCR text extraction from images
- `data/`: Directory for storing documents
- `data_chroma_db/`: Vector store database

## Configuration

The system uses the following default configurations:
- Embedding model: BAAI/bge-small-en-v1.5
- LLM: Ollama with llama3 model
- Chunk size: 1024 tokens
- Chunk overlap: 100 tokens
- Top-k retrieval: 5 documents

## Logging

The application maintains two log files:
- `ui_debug.log`: GUI-related operations
- `debug.log`: Document processing and OCR operations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 