from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import hashlib
import json
from pathlib import Path
from datetime import datetime
from simple_rag_chatbot import SimpleRAGChatbot

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv not installed.")

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="A FastAPI wrapper for the RAG chatbot with document retrieval",
    version="1.0.0"
)

# Request/Response models
class ChatRequest(BaseModel):
    question: str
    max_tokens: Optional[int] = 500
    temperature: Optional[float] = 0.7

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[list] = None
    status: str = "success"

# Global chatbot instance and PDF tracking
chatbot = None
processed_pdfs = {}  # Track processed PDFs with their hashes
pdf_tracking_file = "pdf_tracking.json"

def get_file_hash(file_path: Path) -> str:
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_pdf_tracking():
    """Load PDF tracking data from file"""
    global processed_pdfs
    try:
        if Path(pdf_tracking_file).exists():
            with open(pdf_tracking_file, 'r') as f:
                processed_pdfs = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load PDF tracking: {e}")
        processed_pdfs = {}

def save_pdf_tracking():
    """Save PDF tracking data to file"""
    try:
        with open(pdf_tracking_file, 'w') as f:
            json.dump(processed_pdfs, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save PDF tracking: {e}")

def scan_for_new_pdfs() -> list:
    """Scan PDF_Data folder for new or modified PDFs"""
    pdf_dir = Path("./PDF_Data")
    new_pdfs = []
    
    if not pdf_dir.exists():
        return new_pdfs
    
    for pdf_file in pdf_dir.glob("*.pdf"):
        file_path = str(pdf_file)
        current_hash = get_file_hash(pdf_file)
        
        # Check if file is new or modified
        if file_path not in processed_pdfs or processed_pdfs[file_path]['hash'] != current_hash:
            new_pdfs.append({
                'path': file_path,
                'name': pdf_file.name,
                'hash': current_hash,
                'size': pdf_file.stat().st_size,
                'modified': datetime.fromtimestamp(pdf_file.stat().st_mtime).isoformat()
            })
    
    return new_pdfs

def process_new_pdf(pdf_info: dict) -> bool:
    """Process a new PDF and add it to the vector store"""
    global chatbot
    
    if not chatbot:
        return False
    
    try:
        print(f"📄 Processing new PDF: {pdf_info['name']}")
        
        # Load PDF documents
        pdf_docs = chatbot.load_pdf_documents(pdf_info['path'])
        
        # Add to existing vector store
        if chatbot.vectorstore:
            # Add new documents to existing vector store
            chatbot.vectorstore.add_documents(pdf_docs)
            
            # Recreate the QA chain to include new documents
            chatbot.setup_qa_chain()
        else:
            # Create new vector store if none exists
            chatbot.create_vector_store(pdf_docs)
        
        # Update tracking
        processed_pdfs[pdf_info['path']] = {
            'hash': pdf_info['hash'],
            'processed_at': datetime.now().isoformat(),
            'name': pdf_info['name'],
            'size': pdf_info['size']
        }
        
        save_pdf_tracking()
        print(f"✅ Successfully processed: {pdf_info['name']}")
        return True
        
    except Exception as e:
        print(f"❌ Error processing {pdf_info['name']}: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize the chatbot on startup"""
    global chatbot
    try:
        # Check if OpenAI API key is set
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env file or environment variable.")
        
        print("🚀 Initializing RAG Chatbot...")
        
        # Load PDF tracking data
        load_pdf_tracking()
        
        # Initialize the RAG chatbot
        chatbot = SimpleRAGChatbot()
        
        # Check for existing PDF files first
        pdf_dir = Path("./PDF_Data")
        documents = []
        
        if pdf_dir.exists() and any(pdf_dir.glob("*.pdf")):
            print("📚 Loading existing PDF documents...")
            for pdf_file in pdf_dir.glob("*.pdf"):
                print(f"📄 Processing: {pdf_file.name}")
                pdf_docs = chatbot.load_pdf_documents(str(pdf_file))
                documents.extend(pdf_docs)
        else:
            print("📄 No PDFs found, creating sample document...")
            sample_file = chatbot.create_sample_pdf()
            documents = chatbot.load_from_text_file(sample_file)
        
        print(f"📝 Total documents loaded: {len(documents)}")
        
        # Create vector store
        print("🔍 Creating vector store...")
        chatbot.create_vector_store(documents)
        
        # Set up QA chain
        print("⚙️ Setting up QA chain...")
        chatbot.setup_qa_chain()
        
        print("✅ RAG Chatbot fully initialized and ready!")
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {str(e)}")
        raise e

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "RAG Chatbot API is running",
        "status": "healthy",
        "endpoints": {
            "chat": "/chat",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    global chatbot
    return {
        "status": "healthy" if chatbot is not None else "unhealthy",
        "chatbot_initialized": chatbot is not None,
        "openai_key_set": bool(os.getenv("OPENAI_API_KEY"))
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint for RAG queries"""
    global chatbot
    
    if chatbot is None:
        raise HTTPException(
            status_code=500, 
            detail="Chatbot not initialized. Check server logs for details."
        )
    
    try:
        # Get answer from the RAG chatbot
        result = chatbot.ask_question(request.question)
        answer = result.get('answer', 'No answer generated')
        
        return ChatResponse(
            answer=answer,
            status="success"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )

@app.get("/documents")
async def list_documents():
    """List available documents in the knowledge base"""
    global chatbot
    
    if chatbot is None:
        raise HTTPException(
            status_code=500,
            detail="Chatbot not initialized"
        )
    
    try:
        # Get document count from vector store
        doc_count = chatbot.vectorstore._collection.count() if hasattr(chatbot, 'vectorstore') else 0
        
        return {
            "status": "success",
            "document_count": doc_count,
            "processed_pdfs": len(processed_pdfs),
            "message": f"Knowledge base contains {doc_count} document chunks from {len(processed_pdfs)} PDFs"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error retrieving document info: {str(e)}"
        }

@app.get("/pdfs/scan")
async def scan_pdfs():
    """Scan for new PDFs in the PDF_Data folder"""
    try:
        new_pdfs = scan_for_new_pdfs()
        
        return {
            "status": "success",
            "new_pdfs_found": len(new_pdfs),
            "new_pdfs": new_pdfs,
            "message": f"Found {len(new_pdfs)} new or modified PDFs"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error scanning PDFs: {str(e)}"
        )

@app.post("/pdfs/process")
async def process_new_pdfs():
    """Process all new PDFs found in the PDF_Data folder"""
    global chatbot
    
    if chatbot is None:
        raise HTTPException(
            status_code=500,
            detail="Chatbot not initialized"
        )
    
    try:
        new_pdfs = scan_for_new_pdfs()
        
        if not new_pdfs:
            return {
                "status": "success",
                "processed_count": 0,
                "message": "No new PDFs to process"
            }
        
        processed_count = 0
        failed_pdfs = []
        
        for pdf_info in new_pdfs:
            if process_new_pdf(pdf_info):
                processed_count += 1
            else:
                failed_pdfs.append(pdf_info['name'])
        
        result = {
            "status": "success",
            "processed_count": processed_count,
            "failed_count": len(failed_pdfs),
            "message": f"Successfully processed {processed_count} PDFs"
        }
        
        if failed_pdfs:
            result["failed_pdfs"] = failed_pdfs
            result["message"] += f", {len(failed_pdfs)} failed"
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDFs: {str(e)}"
        )

@app.get("/pdfs/status")
async def get_pdf_status():
    """Get status of all processed PDFs"""
    try:
        pdf_dir = Path("./PDF_Data")
        all_pdfs = list(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
        
        pdf_status = []
        for pdf_file in all_pdfs:
            file_path = str(pdf_file)
            is_processed = file_path in processed_pdfs
            
            status_info = {
                "name": pdf_file.name,
                "path": file_path,
                "size": pdf_file.stat().st_size,
                "modified": datetime.fromtimestamp(pdf_file.stat().st_mtime).isoformat(),
                "processed": is_processed
            }
            
            if is_processed:
                status_info.update(processed_pdfs[file_path])
            
            pdf_status.append(status_info)
        
        return {
            "status": "success",
            "total_pdfs": len(all_pdfs),
            "processed_pdfs": len(processed_pdfs),
            "unprocessed_pdfs": len(all_pdfs) - len(processed_pdfs),
            "pdfs": pdf_status
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting PDF status: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting RAG Chatbot API...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Interactive API: http://localhost:8000/redoc")
    
    uvicorn.run(
        "fastapi_chatbot:app",
        # host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )