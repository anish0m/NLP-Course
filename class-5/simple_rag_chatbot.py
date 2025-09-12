# Simple RAG Chatbot with LangChain, LangGraph & ChromaDB
# "PDF-Powered AI Assistant Made Simple" 📚🤖

import os
from pathlib import Path
from typing import List, Dict, Any

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("💡 Tip: Install python-dotenv to use .env files: pip install python-dotenv")

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# Note: LangGraph features removed for simplicity
# Focus on core RAG functionality with LangChain

class SimpleRAGChatbot:
    """
    A simple RAG chatbot that reads PDFs and answers questions
    """
    
    def __init__(self, openai_api_key: str = None):
        # Set up API key
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required!")
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo",
            openai_api_key=self.api_key
        )
        
        # Vector store
        self.vectorstore = None
        self.qa_chain = None
        
        # Create data directory
        self.data_dir = Path("./data")
        self.data_dir.mkdir(exist_ok=True)
        
        print("✅ RAG Chatbot initialized!")
    
    def load_pdf_documents(self, pdf_path: str) -> List[Dict]:
        """
        Load and process PDF documents
        """
        print(f"📄 Loading PDF: {pdf_path}")
        
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"📝 Created {len(chunks)} text chunks")
        
        return chunks
    
    def create_vector_store(self, documents: List[Dict]):
        """
        Create ChromaDB vector store from documents
        """
        print("🔄 Creating vector store with ChromaDB...")
        
        # Create ChromaDB vector store
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        print("✅ Vector store created successfully!")
    
    def setup_qa_chain(self):
        """
        Set up the Question-Answering chain
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Load documents first!")
        
        # Custom prompt template
        prompt_template = """
        You are a dumb AI assistant that answers questions based on the provided context.
        Use the following pieces as context at the end."
        
        Context: {context}
        
        Question: {question}
        
        Answer: """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}  # Retrieve top 3 relevant chunks
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        print("✅ QA Chain set up successfully!")
    
    def create_sample_pdf(self):
        """
        Create a sample PDF with some content for demonstration
        """
        sample_content = """
        # AI and Machine Learning Guide
        
        ## What is Artificial Intelligence?
        Artificial Intelligence (AI) is a branch of computer science that aims to create 
        intelligent machines that can think and act like humans. AI systems can perform 
        tasks that typically require human intelligence, such as visual perception, 
        speech recognition, decision-making, and language translation.
        
        ## Machine Learning Basics
        Machine Learning (ML) is a subset of AI that enables computers to learn and 
        improve from experience without being explicitly programmed. There are three 
        main types of machine learning:
        
        1. **Supervised Learning**: Learning with labeled examples
        2. **Unsupervised Learning**: Finding patterns in data without labels
        3. **Reinforcement Learning**: Learning through trial and error
        
        ## Deep Learning
        Deep Learning is a subset of machine learning that uses neural networks with 
        multiple layers (hence "deep") to model and understand complex patterns in data. 
        It's particularly effective for tasks like image recognition, natural language 
        processing, and speech recognition.
        
        ## Natural Language Processing
        Natural Language Processing (NLP) is a field of AI that focuses on the 
        interaction between computers and human language. It enables machines to 
        understand, interpret, and generate human language in a valuable way.
        
        ## Applications of AI
        - Healthcare: Medical diagnosis and drug discovery
        - Transportation: Autonomous vehicles
        - Finance: Fraud detection and algorithmic trading
        - Entertainment: Recommendation systems
        - Education: Personalized learning platforms
        
        ## The Future of AI
        AI continues to evolve rapidly, with new breakthroughs in areas like 
        generative AI, robotics, and quantum computing. As AI becomes more 
        sophisticated, it will likely transform many aspects of our daily lives 
        and work.
        """
        
        # Save as text file (simulating PDF content)
        sample_file = self.data_dir / "ai_guide.txt"
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write(sample_content)
        
        print(f"📄 Sample document created: {sample_file}")
        return str(sample_file)
    
    def load_from_text_file(self, file_path: str) -> List[Dict]:
        """
        Load content from text file (simulating PDF)
        """
        print(f"📄 Loading text file: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Create document-like structure
        from langchain.schema import Document
        doc = Document(page_content=content, metadata={"source": file_path})
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        chunks = text_splitter.split_documents([doc])
        print(f"📝 Created {len(chunks)} text chunks")
        
        return chunks
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get an answer from the RAG system
        """
        if not self.qa_chain:
            raise ValueError("QA chain not set up. Call setup_qa_chain() first!")
        
        print(f"\n🤔 Question: {question}")
        print("🔍 Searching for relevant information...")
        
        # Get answer from QA chain
        result = self.qa_chain({"query": question})
        
        answer = result["result"]
        sources = result.get("source_documents", [])
        
        print(f"\n🤖 Answer: {answer}")
        
        if sources:
            print(f"\n📚 Based on {len(sources)} relevant document chunks")
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }
    
    def chat_loop(self):
        """
        Interactive chat loop
        """
        print("\n" + "="*50)
        print("🤖 RAG CHATBOT - Ask me anything!")
        print("="*50)
        print("Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Goodbye! Thanks for chatting!")
                    break
                
                if not question:
                    continue
                
                # Get answer
                self.ask_question(question)
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

def main():
    """
    Main function to demonstrate the RAG chatbot
    """
    print("🚀 Starting Simple RAG Chatbot Demo")
    print("=" * 40)
    
    try:
        # Initialize chatbot
        chatbot = SimpleRAGChatbot()
        
        # Create sample document
        sample_file = chatbot.create_sample_pdf()
        
        # Load documents
        documents = chatbot.load_from_text_file(sample_file)
        
        # Create vector store
        chatbot.create_vector_store(documents)
        
        # Set up QA chain
        chatbot.setup_qa_chain()
        
        # Demo questions
        demo_questions = [
            "What is artificial intelligence?",
            "What are the types of machine learning?",
            "How is deep learning different from machine learning?",
            "What are some applications of AI?"
        ]
        
        print("\n🎭 DEMO: Sample Questions")
        print("=" * 30)
        
        for question in demo_questions:
            chatbot.ask_question(question)
            print("-" * 50)
        
        # Interactive chat
        print("\n🎯 Ready for interactive chat!")
        chatbot.chat_loop()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure to set your OpenAI API key:")
        print('   export OPENAI_API_KEY="your-api-key"')

if __name__ == "__main__":
    main()