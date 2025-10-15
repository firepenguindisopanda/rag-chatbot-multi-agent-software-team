from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import tempfile
import shutil
import traceback
import logging
import random
import uvicorn
import gradio as gr
from typing import Optional

# Optional dotenv load (will be no-op if not installed)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Configure matplotlib for non-GUI environment
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot

import fitz  # PyMuPDF
from typing import List
import logging
import traceback
import random
from copy import deepcopy
from datetime import datetime
from operator import itemgetter

from langchain_community.document_transformers import LongContextReorder
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables.passthrough import RunnableAssign
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA

from langserve import RemoteRunnable

# Import multi-agent software team components
from multi_agent_software_team.schemas import TeamRole, ProjectRequest
from multi_agent_software_team.modern_langgraph_orchestrator import ModernSoftwareTeamOrchestrator
from multi_agent_software_team.utils import (
    format_agent_responses, 
    read_file_content,
    get_agent_summary,
    validate_team_composition,
    save_response_to_md
)

# Import fixed enhanced multi-agent integration
from fixed_enhanced_multi_agent_integration import (
    FixedEnhancedMultiAgentTeam,
    run_fixed_enhanced_multi_agent_collaboration
)

# Import chat with data components
from chat_with_data import (
    DataRequest,
    validate_file_upload,
    save_uploaded_file,
    cleanup_temp_files,
    format_analysis_output,
    get_sample_data_info,
    extract_column_suggestions
)
from chat_with_data.enhanced_langgraph_agents import EnhancedLangGraphDataAnalysisAgent, EnhancedLangGraphDataChatAgent
from chat_with_data.vectorstore_manager import DataVectorStoreManager
from chat_with_data.data_processor import DataProcessor

# Assignment Evaluator
from assignment_evaluator.evaluator import grade_submission
from assignment_evaluator.rubric import load_rubric

def get_traceback(e):
    lines = traceback.format_exception(type(e), e, e.__traceback__)
    return ''.join(lines)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Secure API key loading (removed hardcoded key)
def load_nvidia_api_key() -> Optional[str]:
    """Load NVIDIA API key from (priority):
    1. Existing env var NVIDIA_API_KEY
    2. Streamlit secrets (if running under Streamlit)
    3. .env file (via python-dotenv already loaded)
    Returns the key or None if not found.
    """
    # 1. Direct env
    key = os.environ.get("NVIDIA_API_KEY")
    if key:
        return key
    # 2. Streamlit secrets (lazy import to avoid dependency at server runtime)
    try:
        import streamlit as st  # type: ignore
        if hasattr(st, "secrets") and "NVIDIA_API_KEY" in st.secrets:
            return st.secrets["NVIDIA_API_KEY"]
    except Exception:
        pass
    # 3. Fallback already attempted via load_dotenv
    return os.environ.get("NVIDIA_API_KEY")  # may be None

NVIDIA_API_KEY = load_nvidia_api_key()

# Initialize embeddings model and LLM with error handling
try:
    if not NVIDIA_API_KEY:
        raise ValueError("NVIDIA_API_KEY not set; using mock models")
    embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")
    llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", api_key=NVIDIA_API_KEY)
    logger.info("NVIDIA AI endpoints initialized successfully (secure load)")
except Exception as e:
    logger.error(f"Failed to initialize NVIDIA endpoints (falling back to mock): {e}")
    logger.info("Set NVIDIA_API_KEY via environment, .env, or Streamlit secrets for real responses.")
    # For now, create a mock LLM for testing
    class MockLLM:
        def invoke(self, text):
            class MockResponse:
                content = f"Mock response for: {text}"
            return MockResponse()
        
        def bind(self, **kwargs):
            # Return self for chaining
            return self
        
        def bind_tools(self, tools):
            # Return self for LangGraph compatibility
            return self
        
        def with_config(self, config):
            # Return self for configuration chaining
            return self
    
    class MockEmbedder:
        def __init__(self):
            # Set required properties for LangChain Embeddings compatibility
            self.model = "mock-embedder"
            self.dimension = 768
            # Add model name property for compatibility
            self.model_name = "mock-embedder"
        
        def embed_query(self, text):
            # Return a simple vector of random values for testing
            import hashlib
            import random
            # Use hash to make embeddings consistent for same text
            hash_obj = hashlib.md5(str(text).encode())
            random.seed(hash_obj.hexdigest())
            return [random.uniform(-1, 1) for _ in range(768)]
        
        def embed_documents(self, texts):
            return [self.embed_query(text) for text in texts]
        
        def __call__(self, text):
            # Make the embedder callable - this is what FAISS tries to use
            if isinstance(text, list):
                return self.embed_documents(text)
            else:
                return self.embed_query(text)
        
        # Add async methods for compatibility
        async def aembed_query(self, text):
            return self.embed_query(text)
        
        async def aembed_documents(self, texts):
            return self.embed_documents(texts)
        
        # Add client property for compatibility with some vectorstores
        @property
        def client(self):
            return None
        
        def embed_documents_async(self, texts):
            return self.embed_documents(texts)
        
        # Additional methods for full compatibility
        def _embed_documents(self, texts):
            return self.embed_documents(texts)
        
        def _embed_query(self, text):
            return self.embed_query(text)
        
        # For compatibility with different versions of LangChain
        def embed(self, text):
            if isinstance(text, list):
                return self.embed_documents(text)
            else:
                return self.embed_query(text)
        
        # Make sure the object is properly callable and handles all attribute access
        def __getattr__(self, name):
            # If someone tries to access a method we don't have, 
            # return a function that logs and returns appropriate values
            if name.startswith('embed'):
                def fallback_method(*args, **kwargs):
                    if len(args) > 0:
                        if isinstance(args[0], list):
                            return self.embed_documents(args[0])
                        else:
                            return self.embed_query(args[0])
                    return []
                return fallback_method
            # For FAISS compatibility - return callable self for embedding function
            elif name == 'embedding_function':
                return self
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    embedder = MockEmbedder()
    llm = MockLLM()
    logger.info("Using mock LLM and embedder for testing")

# Load existing docstore if available, otherwise it will be created when first PDF is uploaded
docstore = None
try:
    if os.path.exists("docstore_index.tgz"):
        os.system("tar xzvf docstore_index.tgz")
    
    if os.path.exists("docstore_index"):
        docstore = FAISS.load_local("docstore_index", embedder, allow_dangerous_deserialization=True)
        logger.info("Loaded existing docstore")
    else:
        logger.info("No existing docstore found, will create new one when first PDF is uploaded")
        docstore = None
        
except Exception as e:
    logger.error(f"Could not load existing docstore: {e}")
    logger.info("Will create new docstore when first PDF is uploaded")
    docstore = None

# Enhanced PDF processing function with summarization
def generate_pdf_summary(text_content, title="Document"):
    """Generate a concise summary of the PDF content using NVIDIA LLM"""
    try:
        # Limit text for summarization (to avoid token limits)
        max_chars = 8000  # Adjust based on model limits
        if len(text_content) > max_chars:
            # Take text from beginning, middle, and end for better coverage
            part_size = max_chars // 3
            summary_text = (
                text_content[:part_size] + 
                "\n...\n" + 
                text_content[len(text_content)//2 - part_size//2:len(text_content)//2 + part_size//2] + 
                "\n...\n" + 
                text_content[-part_size:]
            )
        else:
            summary_text = text_content
        
        # Create summarization prompt
        summary_prompt = ChatPromptTemplate.from_template(
            "You are an expert document analyzer. Please provide a concise, well-structured summary of the following document.\n\n"
            "Document Title: {title}\n\n"
            "Document Content:\n{content}\n\n"
            "Please provide a summary that includes:\n"
            "1. **Main Topic**: What is this document primarily about?\n"
            "2. **Key Points**: List 3-5 most important points or findings\n"
            "3. **Document Type**: What type of document is this? (e.g., research paper, report, manual, etc.)\n"
            "4. **Target Audience**: Who would benefit from reading this document?\n"
            "5. **Key Takeaways**: What are the main conclusions or actionable insights?\n\n"
            "Keep the summary concise but informative (approximately 200-400 words)."
        )
        
        # Generate summary using the LLM
        summary_chain = summary_prompt | llm
        response = summary_chain.invoke({
            "title": title,
            "content": summary_text
        })
        
        return response.content
        
    except Exception as e:
        logger.error(f"Error generating summary: {get_traceback(e)}")
        return f"Summary generation failed: {str(e)}"

def extract_document_metadata(text_content, filename):
    """Extract basic metadata from document content"""
    try:
        # Basic stats
        word_count = len(text_content.split())
        char_count = len(text_content)
        line_count = len(text_content.split('\n'))
        
        # Try to detect document type based on content patterns
        doc_type = "General Document"
        content_lower = text_content.lower()
        
        if any(keyword in content_lower for keyword in ['abstract', 'introduction', 'methodology', 'conclusion', 'references']):
            doc_type = "Research Paper/Academic Document"
        elif any(keyword in content_lower for keyword in ['executive summary', 'quarterly', 'financial', 'revenue']):
            doc_type = "Business Report"
        elif any(keyword in content_lower for keyword in ['manual', 'instructions', 'steps', 'procedure']):
            doc_type = "Manual/Guide"
        elif any(keyword in content_lower for keyword in ['policy', 'regulation', 'compliance', 'legal']):
            doc_type = "Policy/Legal Document"
        
        return {
            "word_count": word_count,
            "character_count": char_count,
            "line_count": line_count,
            "estimated_reading_time": f"{max(1, word_count // 200)} minutes",
            "document_type": doc_type,
            "filename": filename
        }
    except Exception as e:
        logger.error(f"Error extracting metadata: {e}")
        return {"error": "Could not extract metadata"}

# Process PDF function
def process_pdf(pdf_path):
    # Open the PDF file
    doc = fitz.open(pdf_path)
    
    # Extract text from each page
    text_content = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text_content += page.get_text()
    
    # Close the PDF document
    doc.close()
    
    # Create metadata
    filename = os.path.basename(pdf_path)
    metadata = {
        "Title": filename,
        "Source": "User uploaded PDF",
        "Published": datetime.now().strftime("%Y-%m-%d")
    }
    
    # Extract additional document metadata
    doc_metadata = extract_document_metadata(text_content, filename)
    
    # Generate summary
    summary = generate_pdf_summary(text_content, filename)
    
    # Create a document
    document = Document(page_content=text_content, metadata=metadata)
    
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", ";", ",", " "],
    )
    
    # Split document into chunks
    chunks = text_splitter.split_documents([document])
    
    # Add chunks to vector store - create new one if needed
    global docstore
    if docstore is None:
        # Create new vector store with the first document
        docstore = FAISS.from_documents(chunks, embedder)
        logger.info("Created new docstore with first PDF")
    else:
        # Add to existing vector store
        docstore.add_documents(chunks)
        logger.info("Added documents to existing docstore")
    
    # Save updated docstore
    docstore.save_local("docstore_index")
    os.system("tar czvf docstore_index.tgz docstore_index")
    
    return {
        "status": "success",
        "message": f"PDF processed with {len(chunks)} chunks added to vector store",
        "title": metadata["Title"],
        "chunks": len(chunks),
        "summary": summary,
        "metadata": doc_metadata,
        "text_length": len(text_content)
    }

# Enhanced PDF upload function for Gradio interface with progress tracking
def pdf_uploader_with_progress(file, progress=gr.Progress()):
    """Enhanced PDF uploader with progress tracking and better UX"""
    if file is None:
        return "❌ **No file uploaded**", "", "", gr.Button(interactive=True)
    
    try:
        # Step 1: Initial validation
        progress(0.1, desc="🔍 Validating PDF file...")
        
        # Handle the file path directly (Gradio provides a file path, not a file object)
        if isinstance(file, str):
            pdf_path = file
            filename = os.path.basename(pdf_path)
        elif hasattr(file, 'name'):
            # Copy the file to a temporary location
            temp_dir = tempfile.mkdtemp()
            filename = getattr(file, 'orig_name', getattr(file, 'name', 'uploaded.pdf'))
            pdf_path = os.path.join(temp_dir, filename)
            shutil.copy2(file.name, pdf_path)
        else:
            return "❌ **Invalid file format received**", "", "", gr.Button(interactive=True)
        
        # Step 2: Extract text
        progress(0.2, desc="📄 Extracting text from PDF...")
        doc = fitz.open(pdf_path)
        text_content = ""
        total_pages = len(doc)
        
        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            text_content += page.get_text()
            progress(0.2 + (0.3 * (page_num + 1) / total_pages), 
                    desc=f"� Processing page {page_num + 1}/{total_pages}")
        
        doc.close()
        
        # Step 3: Extract metadata
        progress(0.5, desc="📊 Analyzing document metadata...")
        doc_metadata = extract_document_metadata(text_content, filename)
        
        # Step 4: Generate summary
        progress(0.6, desc="🤖 Generating AI summary...")
        summary = generate_pdf_summary(text_content, filename)
        
        # Step 5: Create vector store
        progress(0.7, desc="🔍 Creating vector embeddings...")
        
        # Create metadata
        metadata = {
            "Title": filename,
            "Source": "User uploaded PDF",
            "Published": datetime.now().strftime("%Y-%m-%d")
        }
        
        # Create a document
        document = Document(page_content=text_content, metadata=metadata)
        
        # Split the document into chunks
        progress(0.8, desc="✂️ Splitting document into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", ";", ",", " "],
        )
        
        chunks = text_splitter.split_documents([document])
        
        # Add chunks to vector store
        progress(0.9, desc="💾 Updating vector store...")
        global docstore
        if docstore is None:
            docstore = FAISS.from_documents(chunks, embedder)
            logger.info("Created new docstore with first PDF")
        else:
            docstore.add_documents(chunks)
            logger.info("Added documents to existing docstore")
        
        # Save updated docstore
        docstore.save_local("docstore_index")
        os.system("tar czvf docstore_index.tgz docstore_index")
        
        # Step 6: Generate questions
        progress(0.95, desc="💡 Generating suggested questions...")
        preliminary_questions = generate_preliminary_questions()
        questions_text = "\n\n📝 **Here are some questions you might want to ask about your document:**\n\n"
        for i, q in enumerate(preliminary_questions, 1):
            questions_text += f"{i}. {q}\n"
        
        # Complete!
        progress(1.0, desc="✅ Processing complete!")
        
        # Format processing status
        status_text = "✅ **PDF Successfully Processed**\n\n"
        status_text += f"📄 **File**: {filename}\n"
        status_text += f"📊 **Chunks Created**: {len(chunks)}\n"
        status_text += f"📝 **Text Length**: {len(text_content):,} characters\n"
        status_text += f"📄 **Pages**: {total_pages}\n"
        
        # Add metadata if available
        if doc_metadata and not doc_metadata.get('error'):
            if 'word_count' in doc_metadata:
                status_text += f"📖 **Word Count**: {doc_metadata['word_count']:,} words\n"
            if 'estimated_reading_time' in doc_metadata:
                status_text += f"⏱️ **Reading Time**: ~{doc_metadata['estimated_reading_time']}\n"
            if 'document_type' in doc_metadata:
                status_text += f"📋 **Document Type**: {doc_metadata['document_type']}\n"
        
        status_text += "\n🔍 **Vector Store**: Updated with new document chunks for RAG chat"
        
        # Format summary
        summary_text = "## 📄 Document Summary\n\n"
        if summary and not summary.startswith("Summary generation failed"):
            summary_text += summary
        else:
            summary_text += "⚠️ Summary generation failed or not available. However, the document has been successfully processed and added to the vector store for chat."
        
        # Clean up temporary directory if used
        if hasattr(file, 'name') and 'temp_dir' in locals():
            shutil.rmtree(temp_dir)
        
        return status_text, questions_text, summary_text, gr.Button(interactive=True)
        
    except Exception as e:
        logger.error(f"Error in pdf_uploader_with_progress: {get_traceback(e)}")
        error_msg = f"❌ **Error processing PDF**: {str(e)}\n\nPlease try again with a different PDF file."
        
        # Clean up temporary directory if used
        if hasattr(file, 'name') and 'temp_dir' in locals():
            shutil.rmtree(temp_dir)
            
        return error_msg, "", "", gr.Button(interactive=True)

# Simple wrapper for backward compatibility and button state management
def pdf_uploader(file):
    """Wrapper function that manages button state and calls the progress version"""
    return pdf_uploader_with_progress(file)

# Batch PDF processing function
def batch_pdf_processor(files):
    """Process multiple PDF files at once"""
    if not files or len(files) == 0:
        return "No files uploaded", "", ""
    
    try:
        results = []
        total_chunks = 0
        total_words = 0
        processed_files = []
        
        for i, file in enumerate(files):
            try:
                logger.info(f"Processing file {i+1}/{len(files)}: {getattr(file, 'name', 'Unknown')}")
                
                # Handle file processing
                if isinstance(file, str):
                    pdf_path = file
                    filename = os.path.basename(pdf_path)
                elif hasattr(file, 'name'):
                    temp_dir = tempfile.mkdtemp()
                    filename = getattr(file, 'orig_name', getattr(file, 'name', f'uploaded_{i+1}.pdf'))
                    pdf_path = os.path.join(temp_dir, filename)
                    shutil.copy2(file.name, pdf_path)
                else:
                    continue
                
                # Process the PDF
                result = process_pdf(pdf_path)
                
                # Clean up temp directory if created
                if hasattr(file, 'name') and 'temp_dir' in locals():
                    shutil.rmtree(temp_dir)
                
                # Collect results
                results.append(result)
                total_chunks += result.get('chunks', 0)
                processed_files.append(filename)
                
                # Add word count if available
                if 'metadata' in result and result['metadata'] and 'word_count' in result['metadata']:
                    total_words += result['metadata']['word_count']
                
            except Exception as e:
                logger.error(f"Error processing file {i+1}: {get_traceback(e)}")
                results.append({
                    'status': 'error',
                    'title': getattr(file, 'name', f'file_{i+1}'),
                    'error': str(e)
                })
        
        # Format batch processing status
        successful_files = [r for r in results if r.get('status') == 'success']
        failed_files = [r for r in results if r.get('status') == 'error']
        
        status_text = f"📦 **Batch Processing Complete**\n\n"
        status_text += f"✅ **Successfully Processed**: {len(successful_files)}/{len(files)} files\n"
        status_text += f"📊 **Total Chunks Created**: {total_chunks:,}\n"
        if total_words > 0:
            status_text += f"📖 **Total Words Processed**: {total_words:,}\n"
        status_text += f"⏱️ **Total Reading Time**: ~{max(1, total_words // 200)} minutes\n\n"
        
        # List processed files
        if successful_files:
            status_text += "📁 **Successfully Processed Files:**\n"
            for result in successful_files:
                status_text += f"   • {result['title']} ({result['chunks']} chunks)\n"
        
        if failed_files:
            status_text += "\n❌ **Failed Files:**\n"
            for result in failed_files:
                status_text += f"   • {result['title']}: {result.get('error', 'Unknown error')}\n"
        
        status_text += f"\n🔍 **Vector Store**: Updated with {total_chunks:,} new document chunks"
        
        # Generate combined summary for the batch
        if successful_files:
            summary_text = "## 📄 Batch Processing Summary\n\n"
            summary_text += f"**Processed {len(successful_files)} documents successfully:**\n\n"
            
            for i, result in enumerate(successful_files[:3], 1):  # Show first 3 summaries
                if 'summary' in result and result['summary']:
                    summary_text += f"### Document {i}: {result['title']}\n"
                    # Show abbreviated summary for batch processing
                    summary_preview = result['summary'][:400] + "..." if len(result['summary']) > 400 else result['summary']
                    summary_text += f"{summary_preview}\n\n"
            
            if len(successful_files) > 3:
                summary_text += f"*... and {len(successful_files) - 3} more documents. Use individual upload for detailed summaries.*\n"
        else:
            summary_text = "No summaries available - all files failed to process."
        
        # Generate questions for the batch
        questions_text = "📝 **Suggested questions for your document collection:**\n\n"
        questions_text += "1. What are the common themes across these documents?\n"
        questions_text += "2. How do these documents relate to each other?\n"
        questions_text += "3. What are the key findings from this document set?\n"
        questions_text += "4. Can you compare the main points from different documents?\n"
        questions_text += "5. What insights can be drawn from the entire collection?\n"
        
        return status_text, questions_text, summary_text
        
    except Exception as e:
        logger.error(f"Error in batch_pdf_processor: {get_traceback(e)}")
        return f"❌ **Batch processing error**: {str(e)}", "", ""

# RAG chain utilities
def docs2str(docs, title="Document"):
    """Useful utility for making chunks into context string."""
    out_str = ""
    for doc in docs:
        doc_name = getattr(doc, 'metadata', {}).get('Title', title)
        if doc_name:
            out_str += f"[Quote from {doc_name}] "
        if isinstance(doc, dict):
            out_str += doc.get('page_content', doc) + "\n"
        else: 
            out_str += getattr(doc, 'page_content', str(doc)) + "\n"
    return out_str

def output_puller(inputs):
    """If you want to support streaming, implement final step as a generator extractor."""
    for token in inputs:
        if token.get('output'):
            yield token.get('output')

# Initialize LLM (already initialized above with error handling)

# Define retriever function
def retrieve_documents(query):
    try:
        if docstore is None:
            logger.warning("No documents uploaded yet. Please upload a PDF first.")
            return []
        
        docs = docstore.as_retriever(search_kwargs={"k": 5}).invoke(query)
        reordered_docs = LongContextReorder().transform_documents(docs)
        return reordered_docs
    except Exception as e:
        logger.error(f"Error retrieving documents: {get_traceback(e)}")
        return []

# Define generator function
def generate_response(context, query):
    prompt = ChatPromptTemplate.from_template(
        "You are a document chatbot. Help the user as they ask questions about documents."
        " User messaged just asked you a question: {input}\n\n"
        " The following information may be useful for your response: "
        " Document Retrieval:\n{context}\n\n"
        " (Answer only from retrieval. Only cite sources that are used. Make your response conversational)"
        "\n\nUser Question: {input}"
    )
    
    chain = prompt | llm
    
    try:
        response = chain.invoke({"context": context, "input": query})
        return response.content
    except Exception as e:
        logger.error(f"Error generating response: {get_traceback(e)}")
        return f"Error generating response: {str(e)}"

# Quiz and evaluation functions
def generate_quiz_questions(num_questions=5):
    """Generate quiz questions based on uploaded documents"""
    try:
        if docstore is None:
            return [], "No documents uploaded yet. Please upload a PDF first."
        
        # Sample documents from the docstore
        docs = list(docstore.docstore._dict.values())
        if len(docs) < 2:
            return [], "Not enough content in the uploaded documents to generate quiz questions."
        
        # Generate questions based on document content
        questions = []
        answers = []
        
        prompt = ChatPromptTemplate.from_template(
            "Based on the following document content, generate a challenging but fair question that tests comprehension."
            " The question should be specific to the content and require understanding, not just memory."
            " Format: Question: [your question]\nAnswer: [correct answer]\n\n"
            "Document content:\n{content}"
        )
        
        # Sample random documents and generate questions
        sampled_docs = random.sample(docs, min(num_questions, len(docs)))
        
        for i, doc in enumerate(sampled_docs):
            try:
                content = doc.page_content[:1500]  # Limit content length
                qa_response = (prompt | llm).invoke({"content": content})
                
                # Parse the response
                lines = qa_response.content.split('\n')
                question_line = next((line for line in lines if line.startswith('Question:')), None)
                answer_line = next((line for line in lines if line.startswith('Answer:')), None)
                
                if question_line and answer_line:
                    question = question_line.replace('Question:', '').strip()
                    answer = answer_line.replace('Answer:', '').strip()
                    questions.append(question)
                    answers.append(answer)
                else:
                    # Fallback if format is not followed
                    questions.append(f"What is the main concept discussed in this section: {content[:200]}...?")
                    answers.append("Please refer to the document content for the answer.")
                    
            except Exception as e:
                logger.error(f"Error generating question {i+1}: {e}")
                questions.append(f"Question {i+1}: What key information can you extract from the uploaded document?")
                answers.append("Please refer to the document content for the answer.")
        
        return questions, answers, "Quiz questions generated successfully!"
        
    except Exception as e:
        logger.error(f"Error in generate_quiz_questions: {get_traceback(e)}")
        return [], [], f"Error generating quiz questions: {str(e)}"

def evaluate_answer(question, user_answer, correct_answer, context=""):
    """Evaluate user's answer against the correct answer"""
    try:
        if not user_answer or not user_answer.strip():
            return "Please provide an answer to evaluate.", 0
        
        eval_prompt = ChatPromptTemplate.from_template(
            "Evaluate the user's answer to the question. Compare it with the correct answer and any provided context."
            " Rate the answer on a scale of 0-10 where:"
            " 0-3: Incorrect or completely off-topic"
            " 4-6: Partially correct but missing key information"
            " 7-8: Mostly correct with minor issues"
            " 9-10: Excellent, comprehensive answer"
            "\n\nQuestion: {question}"
            "\nCorrect Answer: {correct_answer}"
            "\nUser's Answer: {user_answer}"
            "\nContext: {context}"
            "\n\nProvide your evaluation in this format:"
            "\nScore: [0-10]"
            "\nFeedback: [detailed explanation of the score]"
        )
        
        evaluation = (eval_prompt | llm).invoke({
            "question": question,
            "correct_answer": correct_answer,
            "user_answer": user_answer,
            "context": context
        })
        
        # Parse score from response
        lines = evaluation.content.split('\n')
        score_line = next((line for line in lines if line.startswith('Score:')), None)
        feedback_line = next((line for line in lines if line.startswith('Feedback:')), None)
        
        if score_line:
            try:
                score = int(score_line.replace('Score:', '').strip())
                score = max(0, min(10, score))  # Ensure score is between 0-10
            except ValueError:
                score = 5  # Default score if parsing fails
        else:
            score = 5
            
        if feedback_line:
            feedback = feedback_line.replace('Feedback:', '').strip()
        else:
            feedback = "Your answer has been evaluated. Keep studying the document!"
            
        return feedback, score
        
    except Exception as e:
        logger.error(f"Error in evaluate_answer: {get_traceback(e)}")
        return f"Error evaluating answer: {str(e)}", 0

def generate_preliminary_questions():
    """Generate preliminary questions when a document is uploaded"""
    try:
        if docstore is None:
            return []
        
        # Get a sample of document content
        docs = list(docstore.docstore._dict.values())
        if not docs:
            return []
        
        # Sample a few documents to get content variety
        sample_docs = random.sample(docs, min(3, len(docs)))
        combined_content = "\n".join([doc.page_content[:500] for doc in sample_docs])
        
        prompt = ChatPromptTemplate.from_template(
            "Based on this document content, suggest 4-5 good starter questions that someone might ask to explore and understand the document better."
            " Make the questions engaging and varied (some factual, some analytical)."
            " Format each question on a new line starting with '• '"
            "\n\nDocument content:\n{content}"
        )
        
        response = (prompt | llm).invoke({"content": combined_content})
        
        # Parse questions from response
        lines = response.content.split('\n')
        questions = [line.strip() for line in lines if line.strip().startswith('•')]
        
        # Clean up questions and limit to 5
        questions = [q.replace('•', '').strip() for q in questions][:5]
        
        return questions if questions else [
            "What is the main topic of this document?",
            "What are the key points discussed?",
            "Can you summarize the important findings?",
            "What conclusions can be drawn from this content?"
        ]
        
    except Exception as e:
        logger.error(f"Error generating preliminary questions: {get_traceback(e)}")
        return [
            "What is the main topic of this document?",
            "What are the key points discussed?",
            "Can you summarize the important findings?"
        ]

# Define FastAPI app
app = FastAPI()

# Define routes for RAG components
@app.post("/upload_pdf/")
async def upload_pdf_endpoint(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        return JSONResponse(
            status_code=400,
            content={"message": "Only PDF files are accepted"}
        )
    
    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        result = process_pdf(pdf_path)
        shutil.rmtree(temp_dir)
        return result
    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        logger.error(f"Error processing PDF: {get_traceback(e)}")
        return JSONResponse(
            status_code=500,
            content={"message": f"Error processing PDF: {str(e)}"}
        )

@app.post("/retriever/")
async def retriever_endpoint(query: str):
    try:
        docs = retrieve_documents(query)
        return docs
    except Exception as e:
        logger.error(f"Error in retriever endpoint: {get_traceback(e)}")
        return JSONResponse(
            status_code=500,
            content={"message": f"Error retrieving documents: {str(e)}"}
        )

@app.post("/generator/")
async def generator_endpoint(context: str, query: str):
    try:
        response = generate_response(context, query)
        return {"output": response}
    except Exception as e:
        logger.error(f"Error in generator endpoint: {get_traceback(e)}")
        return JSONResponse(
            status_code=500,
            content={"message": f"Error generating response: {str(e)}"}
        )

@app.post("/basic_chat/")
async def basic_chat_endpoint(query: str):
    try:
        response = llm.invoke(query)
        return {"output": response.content}
    except Exception as e:
        logger.error(f"Error in basic chat endpoint: {get_traceback(e)}")
        return JSONResponse(
            status_code=500,
            content={"message": f"Error in basic chat: {str(e)}"}
        )

# ChatBot utilities for Gradio interface

def add_text(history, text):
    # Add user message in messages format
    history = history + [{"role": "user", "content": text}]
    return history, gr.Textbox(value="", interactive=False)

def bot(history, chain_key):
    # Check if history is empty or doesn't have a user message
    if not history or len(history) == 0:
        return
    
    # Get the last user message
    last_message = history[-1]
    if isinstance(last_message, dict) and last_message.get("role") == "user":
        query = last_message["content"]
    else:
        logger.error("Invalid message format in history")
        return
    
    if chain_key == 'Basic':
        try:
            response = llm.invoke(query)
            # Add assistant response in messages format
            history.append({"role": "assistant", "content": response.content})
            yield history
        except Exception as e:
            logger.error(f"Error in basic chat: {get_traceback(e)}")
            history.append({"role": "assistant", "content": f"Error: {str(e)}"})
            yield history
    else:  # RAG mode
        try:
            # Use the RAG chain directly
            docs = retrieve_documents(query)
            context = docs2str(docs)
            response = generate_response(context, query)
            # Add assistant response in messages format
            history.append({"role": "assistant", "content": response})
            yield history
        except Exception as e:
            logger.error(f"Error in RAG chat: {get_traceback(e)}")
            history.append({"role": "assistant", "content": f"Error: {str(e)}"})
            yield history

# Define Gradio interface
def get_demo():
    with gr.Blocks(title="RAG PDF Server", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📚 RAG PDF Server - Chat, Upload, and Test Your Understanding")
        
        with gr.Tabs():
            with gr.TabItem("💬 Chat"):
                gr.Markdown("### Chat with your uploaded documents")
                chatbot = gr.Chatbot(
                    [],
                    elem_id="chatbot",
                    type="messages",
                    height=400,
                    show_label=False
                )
                
                with gr.Row():
                    txt = gr.Textbox(
                        scale=4,
                        show_label=False,
                        placeholder="Ask a question about your uploaded documents...",
                        container=False,
                    )
                    chain_btn = gr.Radio(["Basic", "RAG"], value="RAG", label="Mode")
                
                # Setup text input handling
                txt.submit(
                    fn=add_text,
                    inputs=[chatbot, txt],
                    outputs=[chatbot, txt],
                    queue=False
                ).then(bot, [chatbot, chain_btn], [chatbot]).then(
                    lambda: gr.Textbox(interactive=True), None, [txt], queue=False
                )
            
            with gr.TabItem("📄 Upload PDF"):
                gr.Markdown("### Upload and process your PDF documents")
                gr.Markdown("*Upload PDF files to create vector store embeddings and generate automatic summaries for enhanced RAG-based chat.*")
                
                with gr.Tabs():
                    with gr.TabItem("📄 Single PDF"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                pdf_input = gr.File(label="Select PDF file", file_types=[".pdf"])
                                upload_button = gr.Button("Process PDF", variant="primary")
                                # Progress status section
                                gr.Markdown("### 📊 Processing Status")
                                upload_output = gr.Markdown(
                                    value="*🔄 Select a PDF file and click 'Process PDF' to begin...*",
                                    label="Status"
                                )
                            
                            with gr.Column(scale=2):
                                # Document Summary Section
                                gr.Markdown("### 📄 Document Summary")
                                summary_output = gr.Markdown(
                                    value="*📄 Document summary will appear here after processing...*"
                                )
                        
                        # Suggested Questions Section
                        with gr.Row():
                            gr.Markdown("### 💡 Suggested Questions")
                            questions_output = gr.Markdown(
                                value="*💡 Suggested questions will appear here after processing...*"
                            )
                        
                        # Enhanced button click handler with progress and state management
                        def process_pdf_with_ui_feedback(file):
                            """Process PDF with UI feedback and button state management"""
                            if file is None:
                                return (
                                    "❌ **Please select a PDF file first**",
                                    "*💡 Suggested questions will appear here after processing...*",
                                    "*📄 Document summary will appear here after processing...*",
                                    gr.Button("Process PDF", variant="primary", interactive=True)
                                )
                            
                            # Show processing start
                            yield (
                                "🔄 **Processing started...** Please wait while we analyze your PDF.",
                                "*💡 Processing in progress...*",
                                "*📄 Processing in progress...*", 
                                gr.Button("Processing...", variant="secondary", interactive=False)
                            )
                            
                            # Call the actual processing function
                            result = pdf_uploader_with_progress(file)
                            
                            # Return final results with re-enabled button
                            yield result
                        
                        upload_button.click(
                            fn=process_pdf_with_ui_feedback,
                            inputs=[pdf_input],
                            outputs=[upload_output, questions_output, summary_output, upload_button]
                        )
                    
                    with gr.TabItem("📦 Batch Processing"):
                        gr.Markdown("*Upload multiple PDF files at once for batch processing.*")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                batch_pdf_input = gr.File(
                                    label="Select Multiple PDF files", 
                                    file_types=[".pdf"], 
                                    file_count="multiple"
                                )
                                batch_upload_button = gr.Button("Process All PDFs", variant="primary")
                                batch_upload_output = gr.Markdown(label="Batch Processing Status")
                            
                            with gr.Column(scale=2):
                                # Batch Summary Section
                                batch_summary_output = gr.Markdown(
                                    value="*📄 Batch processing summary will appear here...*",
                                    label="Batch Summary"
                                )
                        
                        # Batch Questions Section
                        with gr.Row():
                            batch_questions_output = gr.Markdown(
                                value="*💡 Questions for document collection will appear here...*",
                                label="Collection Questions"
                            )
                        
                        batch_upload_button.click(
                            fn=batch_pdf_processor,
                            inputs=batch_pdf_input,
                            outputs=[batch_upload_output, batch_questions_output, batch_summary_output]
                        )
            
            with gr.TabItem("🧠 Test Your Understanding"):
                gr.Markdown("### Take a quiz based on your uploaded documents")
                gr.Markdown("*📝 This feature generates questions to test your comprehension of the uploaded content.*")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        num_questions = gr.Slider(
                            minimum=1, 
                            maximum=10, 
                            value=5, 
                            step=1, 
                            label="Number of Questions"
                        )
                        generate_btn = gr.Button("🎯 Generate Quiz", variant="primary")
                        quiz_status = gr.Textbox(label="Quiz Status", lines=2)
                    
                    with gr.Column(scale=2):
                        # Quiz interface - initially hidden
                        quiz_interface = gr.Column(visible=False)
                        
                        with quiz_interface:
                            current_question = gr.Markdown("", label="Question")
                            user_answer = gr.Textbox(
                                label="Your Answer", 
                                placeholder="Type your answer here...",
                                lines=4
                            )
                            
                            with gr.Row():
                                submit_answer_btn = gr.Button("✅ Submit Answer")
                                next_question_btn = gr.Button("➡️ Next Question", visible=False)
                                show_correct_btn = gr.Button("👁️ Show Correct Answer", visible=False)
                            
                            feedback_area = gr.Markdown("", label="Feedback")
                            score_display = gr.Textbox(label="Score", visible=False)
                            correct_answer_display = gr.Markdown("", visible=False)
                
                # Quiz state management
                quiz_questions = gr.State([])
                quiz_answers = gr.State([])
                current_q_index = gr.State(0)
                user_scores = gr.State([])
                
                def start_quiz(num_q):
                    questions, answers, status = generate_quiz_questions(num_q)
                    if questions:
                        return (
                            status,
                            gr.Column(visible=True),  # Show quiz interface
                            questions,
                            answers,
                            0,  # Reset question index
                            [],  # Reset scores
                            f"**Question 1 of {len(questions)}:**\n\n{questions[0]}",
                            "",  # Clear answer box
                            "",  # Clear feedback
                            gr.Button(visible=True),  # Submit button
                            gr.Button(visible=False),  # Next button
                            gr.Button(visible=False),  # Show answer button
                            gr.Textbox(visible=False),  # Score display
                            gr.Markdown(visible=False)  # Correct answer display
                        )
                    else:
                        return (
                            status,
                            gr.Column(visible=False),
                            [],
                            [],
                            0,
                            [],
                            "",
                            "",
                            "",
                            gr.Button(visible=True),
                            gr.Button(visible=False),
                            gr.Button(visible=False),
                            gr.Textbox(visible=False),
                            gr.Markdown(visible=False)
                        )
                
                def submit_answer(questions, answers, q_index, user_ans, scores):
                    if not questions or q_index >= len(questions):
                        return "No active quiz", "", gr.Button(visible=True), gr.Button(visible=False), gr.Button(visible=False), gr.Textbox(visible=False), scores
                    
                    question = questions[q_index]
                    correct_answer = answers[q_index]
                    
                    feedback, score = evaluate_answer(question, user_ans, correct_answer)
                    scores.append(score)
                    
                    is_last_question = q_index >= len(questions) - 1
                    
                    if is_last_question:
                        avg_score = sum(scores) / len(scores)
                        final_feedback = f"{feedback}\n\n🎉 **Quiz Complete!**\n\n📊 **Final Score: {avg_score:.1f}/10**\n\n📈 **Individual Scores:** {', '.join(map(str, scores))}"
                        return (
                            final_feedback,
                            "",
                            gr.Button(visible=False),  # Hide submit
                            gr.Button(visible=False),  # Hide next
                            gr.Button(visible=True),   # Show correct answer
                            gr.Textbox(value=f"{avg_score:.1f}/10", visible=True),
                            scores
                        )
                    else:
                        feedback_with_progress = f"{feedback}\n\n📊 **Question {q_index + 1} Score: {score}/10**"
                        return (
                            feedback_with_progress,
                            "",
                            gr.Button(visible=False),  # Hide submit
                            gr.Button(visible=True),   # Show next
                            gr.Button(visible=True),   # Show correct answer
                            gr.Textbox(visible=False),
                            scores
                        )
                
                def next_question(questions, q_index):
                    new_index = q_index + 1
                    if new_index < len(questions):
                        return (
                            new_index,
                            f"**Question {new_index + 1} of {len(questions)}:**\n\n{questions[new_index]}",
                            "",  # Clear answer
                            "",  # Clear feedback
                            gr.Button(visible=True),   # Show submit
                            gr.Button(visible=False),  # Hide next
                            gr.Button(visible=False),  # Hide show answer
                            gr.Markdown(visible=False)  # Hide correct answer
                        )
                    return q_index, "Quiz completed!", "", "", gr.Button(visible=False), gr.Button(visible=False), gr.Button(visible=False), gr.Markdown(visible=False)
                
                def show_correct_answer(answers, q_index):
                    if answers and q_index < len(answers):
                        return gr.Markdown(value=f"**Correct Answer:**\n\n{answers[q_index]}", visible=True)
                    return gr.Markdown(visible=False)
                
                # Connect the quiz functionality
                generate_btn.click(
                    fn=start_quiz,
                    inputs=[num_questions],
                    outputs=[
                        quiz_status, quiz_interface, quiz_questions, quiz_answers, 
                        current_q_index, user_scores, current_question, user_answer, 
                        feedback_area, submit_answer_btn, next_question_btn, show_correct_btn,
                        score_display, correct_answer_display
                    ]
                )
                
                submit_answer_btn.click(
                    fn=submit_answer,
                    inputs=[quiz_questions, quiz_answers, current_q_index, user_answer, user_scores],
                    outputs=[feedback_area, user_answer, submit_answer_btn, next_question_btn, show_correct_btn, score_display, user_scores]
                )
                
                next_question_btn.click(
                    fn=next_question,
                    inputs=[quiz_questions, current_q_index],
                    outputs=[current_q_index, current_question, user_answer, feedback_area, submit_answer_btn, next_question_btn, show_correct_btn, correct_answer_display]
                )
                
                show_correct_btn.click(
                    fn=show_correct_answer,
                    inputs=[quiz_answers, current_q_index],
                    outputs=[correct_answer_display]
                )
            
            with gr.TabItem("🤖 Software Team"):
                gr.Markdown("### Enhanced Multi-Agent Software Development Team")
                gr.Markdown("*� Transform your ideas into complete software solutions with our collaborative AI team*")
                
                # Add team info section
                with gr.Accordion("👥 Meet Your AI Team", open=False):
                    gr.Markdown("""
**Your multi-agent team consists of 7 specialized AI experts:**

- **📋 Product Owner** - Defines requirements, user stories, and acceptance criteria
- **🔍 Analyst** - Creates detailed technical specifications and functional requirements  
- **🏗️ Architect** - Designs system architecture with visual diagrams and APIs
- **💻 Developer** - Implements code, database design, and technical solutions
- **👀 Reviewer** - Conducts code quality review and security analysis
- **🧪 Tester** - Creates comprehensive testing strategies and test cases
- **📝 Tech Writer** - Produces final documentation and deployment guides

**✨ Enhanced Features:**
- Handoff-based workflow between agents
- Context-aware collaboration
- Automatic Mermaid diagram generation
- Comprehensive software solutions from concept to deployment
                    """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        project_description = gr.Textbox(
                            label="Project Description", 
                            placeholder="Describe your software project in detail...\n\nExample:\n'Create a task management web application for small teams with user authentication, project creation, task assignment, real-time collaboration, and progress tracking dashboards.'",
                            lines=8
                        )
                        project_file = gr.File(
                            label="Upload Context File (Optional)",
                            file_types=[".txt", ".md", ".docx", ".pdf"]
                        )
                        
                        gr.Markdown("### 🎯 Quick Start Examples")
                        with gr.Accordion("💡 Example Project Ideas", open=False):
                            gr.Markdown("""
**Web Application:**
"Create a task management web application for development teams with user authentication, project creation, task assignment, real-time collaboration features, and progress dashboards."

**Mobile App:**
"Build a fitness tracking mobile application with workout logging, progress visualization, social features, and integration with wearable devices."

**API System:**
"Design a RESTful API for an e-commerce platform with product management, order processing, payment integration, and inventory tracking."

**Enterprise System:**
"Develop a customer relationship management (CRM) system with lead tracking, sales pipeline management, reporting, and email integration."
                            """)
                        
                        generate_btn = gr.Button("🚀 Generate Complete Solution", variant="primary", size="lg")
                        status_output = gr.Textbox(label="Status", lines=3)
                    
                    with gr.Column(scale=2):
                        # Add tabs for different output views
                        with gr.Tabs():
                            with gr.TabItem("📋 Team Output"):
                                team_output = gr.Markdown(
                                    label="Team Results",
                                    value="*Select roles and provide a project description to generate a comprehensive software solution.*"
                                )
                            
                            with gr.TabItem("📊 Summary"):
                                team_summary = gr.Markdown(
                                    label="Team Summary",
                                    value="*Summary will appear here after generation.*"
                                )
                        
                        # Add save functionality
                        with gr.Row():
                            save_btn = gr.Button("💾 Save Solution to MD File", variant="secondary", size="lg")
                        
                        save_status = gr.Textbox(
                            label="Save Status", 
                            placeholder="Save status will appear here after generation...", 
                            lines=4,
                            show_label=True
                        )
                
                def run_software_team(description, file):
                    """Run the enhanced multi-agent software team."""
                    if not description and file is None:
                        return "❌ Please provide a project description or upload a file", "*Provide a project description to generate a comprehensive software solution.*", "*Summary will appear here after generation.*"
                    
                    try:
                        # Get file content if provided
                        file_content = read_file_content(file) if file else ""
                        
                        # Use the fixed enhanced multi-agent collaboration (includes all agents automatically)
                        result = run_fixed_enhanced_multi_agent_collaboration(
                            llm, 
                            description, 
                            file_content
                        )
                        
                        if result.startswith("❌"):
                            return result, "*Provide a project description to generate a comprehensive software solution.*", "*Summary will appear here after generation.*"
                        
                        # Create summary from the result
                        summary_output = f"""✅ **Enhanced Multi-Agent Solution Generated Successfully!**

**Project:** {description[:100]}{'...' if len(description) > 100 else ''}
**Team Size:** 7 specialized AI agents
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Solution Length:** {len(result):,} characters

The complete software solution includes:
• Requirements and user stories
• System architecture with diagrams
• Technical implementation details
• Code review and security analysis  
• Testing strategy and test cases
• Complete technical documentation"""
                        
                        return "✅ Enhanced solution generated successfully!", result, summary_output
                        
                    except Exception as e:
                        logger.error(f"Error in enhanced software team: {str(e)}")
                        return f"❌ Error: {str(e)}", "*Provide a project description to generate a comprehensive software solution.*", "*Summary will appear here after generation.*"
                
                def save_team_response(team_output_content, description):
                    """Save the team response to an MD file."""
                    try:
                        if not team_output_content or team_output_content.startswith("*Provide"):
                            return "❌ No solution to save. Please generate a solution first."
                        
                        # Create filename based on description
                        safe_description = "".join(c for c in (description or "software_project")[:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"multi_agent_solution_{safe_description}_{timestamp}.md"
                        
                        # Ensure solutions directory exists
                        solutions_dir = "solutions"
                        os.makedirs(solutions_dir, exist_ok=True)
                        
                        filepath = os.path.join(solutions_dir, filename)
                        
                        # Save the solution
                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write(team_output_content)
                        
                        # Create summary file
                        summary_content = f"""# Solution Summary
                        
**Project:** {description or "Software Development Project"}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**File:** {filename}

## Quick Stats
- **Content Length:** {len(team_output_content):,} characters
- **Estimated Reading Time:** {len(team_output_content.split()) // 200 + 1} minutes

## Next Steps
1. Review the complete solution in {filename}
2. Implement the technical specifications
3. Follow the deployment guidelines
4. Use the documentation as reference

*This summary was auto-generated by the Multi-Agent Software Team*
"""
                        summary_filename = f"summary_{filename}"
                        summary_path = os.path.join(solutions_dir, summary_filename)
                        with open(summary_path, "w", encoding="utf-8") as f:
                            f.write(summary_content)
                        
                        return f"""✅ Solution saved successfully!
📁 Main File: {filename}
📂 Location: {os.path.abspath(solutions_dir)}
📊 Size: {len(team_output_content):,} characters

💡 Files created:
• Main solution: {filename}
• Summary: {summary_filename}"""
                        
                    except Exception as e:
                        logger.error(f"Error saving team response: {str(e)}")
                        return f"❌ Error saving file: {str(e)}"
                
                # Connect the functionality
                generate_btn.click(
                    fn=run_software_team,
                    inputs=[project_description, project_file],
                    outputs=[status_output, team_output, team_summary]
                )
                
                save_btn.click(
                    fn=save_team_response,
                    inputs=[team_output, project_description],
                    outputs=[save_status]
                )
            
            with gr.TabItem("🛠️ Evaluate RAG"):
                gr.Markdown("### RAG System Evaluation")
                gr.Markdown("*🎯 Test and evaluate your RAG system's performance using LLM-as-a-Judge metrics*")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        eval_description = gr.Markdown("""
### How RAG Evaluation Works:

1. **Generate Questions**: System samples random document chunks and creates synthetic question-answer pairs
2. **RAG Response**: Your RAG system answers the same questions  
3. **Judge Evaluation**: An LLM judge compares RAG responses against synthetic ground truth
4. **Score Calculation**: Percentage of questions where RAG outperformed the baseline

**Pass Threshold:** 60% (5 out of 8 questions must score [2])

**Scoring:**
- **[1]** - Baseline answer is preferred / RAG answer has issues
- **[2]** - RAG answer is better and doesn't contradict baseline
                        """)
                        
                        eval_mode = gr.Radio(
                            ["Basic", "RAG"], 
                            value="RAG", 
                            label="Chain to Evaluate"
                        )
                        
                        start_eval_btn = gr.Button("🎓 Start Evaluation", variant="primary", size="lg")
                        
                        eval_stats = gr.Markdown("", label="Evaluation Statistics")
                    
                    with gr.Column(scale=2):
                        eval_chatbot = gr.Chatbot(
                            [],
                            elem_id="eval_chatbot", 
                            type="messages",
                            height=600,
                            show_label=False,
                            show_copy_button=True
                        )
                
                # Evaluation function
                def run_rag_evaluation(chain_key):
                    """Run RAG evaluation similar to frontend_block.py"""
                    try:
                        # Check if docstore exists
                        if docstore is None:
                            return [{"role": "assistant", "content": "❌ No documents found. Please upload a PDF first."}], "**Status:** No documents available for evaluation"
                        
                        # Get document chunks
                        doc_chunks = list(docstore.docstore._dict.values())
                        if len(doc_chunks) < 10:
                            return [{"role": "assistant", "content": f"❌ Need at least 10 document chunks for evaluation. Found: {len(doc_chunks)}"}], "**Status:** Insufficient document chunks"
                        
                        # Initialize evaluation
                        messages = []
                        messages.append({"role": "assistant", "content": f"🎯 **Starting RAG Evaluation**\n\n**Mode:** {chain_key}\n**Documents:** {len(doc_chunks)} chunks available\n**Questions:** 8 questions will be generated\n\n---"})
                        
                        num_points = 0
                        num_questions = 8
                        
                        for i in range(num_questions):
                            # Generate synthetic QA pair
                            doc1, doc2 = random.sample(doc_chunks, 2)
                            synth_prompt = get_synth_prompt([doc1, doc2])
                            
                            # Add question generation message
                            messages.append({"role": "user", "content": f"**Question {i+1}/{num_questions}:** Generating synthetic QA pair..."})
                            
                            try:
                                synth_qa = (synth_prompt | llm).invoke({})
                                synth_pair = synth_qa.content.split("\n\n")
                                
                                if len(synth_pair) < 2:
                                    messages.append({"role": "assistant", "content": "❌ Failed to generate valid QA pair"})
                                    continue
                                
                                synth_q, synth_a = synth_pair[:2]
                                messages.append({"role": "assistant", "content": f"**Generated Question:**\n{synth_q}\n\n**Ground Truth Answer:**\n{synth_a}"})
                                
                                # Get RAG response
                                question_only = synth_q.replace("Question: ", "").strip()
                                
                                if chain_key == 'Basic':
                                    rag_response = llm.invoke(question_only).content
                                else:  # RAG mode
                                    context = retrieve_documents(question_only)
                                    context_str = docs2str(context)
                                    rag_response = generate_response(context_str, question_only)
                                
                                messages.append({"role": "user", "content": f"**RAG Answer:**\n{rag_response}"})
                                
                                # Evaluate with judge
                                eval_prompt = get_eval_prompt()
                                usr_msg = f"Question: {synth_q}\n\nAnswer 1 (Ground Truth): {synth_a}\n\nAnswer 2 (New): {rag_response}"
                                eval_response = (eval_prompt | llm).invoke(usr_msg)
                                
                                # Check for score
                                is_better = "[2]" in eval_response.content
                                if is_better:
                                    num_points += 1
                                
                                score_indicator = "✅ [2]" if is_better else "❌ [1]"
                                messages.append({"role": "assistant", "content": f"**Judge Evaluation:**\n{eval_response.content}\n\n{score_indicator} - Running Score: {num_points}/{i+1}"})
                                
                            except Exception as e:
                                messages.append({"role": "assistant", "content": f"❌ Error in question {i+1}: {str(e)}"})
                                continue
                        
                        # Final results
                        final_score = num_points / num_questions
                        if final_score >= 0.60:
                            final_msg = f"🎉 **EVALUATION PASSED!** 🎉\n\n📊 **Final Score:** {final_score:.2%} ({num_points}/{num_questions})\n✅ **Threshold:** 60% met\n\n**Congratulations!** Your RAG system performed well!"
                        else:
                            final_msg = f"❌ **Evaluation needs improvement**\n\n📊 **Final Score:** {final_score:.2%} ({num_points}/{num_questions})\n⚠️ **Threshold:** 60% required\n\n**Suggestion:** Review your document processing and retrieval strategy."
                        
                        messages.append({"role": "assistant", "content": final_msg})
                        
                        stats = f"**Evaluation Complete**\n- **Score:** {final_score:.2%}\n- **Questions:** {num_questions}\n- **Passed:** {num_points}\n- **Status:** {'✅ PASSED' if final_score >= 0.60 else '❌ NEEDS IMPROVEMENT'}"
                        
                        return messages, stats
                        
                    except Exception as e:
                        error_msg = [{"role": "assistant", "content": f"❌ **Evaluation Error:** {str(e)}"}]
                        error_stats = "**Status:** Error occurred during evaluation"
                        return error_msg, error_stats
                
                def get_synth_prompt(docs):
                    """Generate synthetic question prompt from document chunks"""
                    doc1, doc2 = docs[:2] if len(docs) >= 2 else (docs[0], docs[0])
                    
                    def format_chunk(doc):
                        return (
                            f"Paper: {doc.metadata.get('Title', 'unknown')}"
                            f"\n\nSummary: {doc.metadata.get('Summary', 'unknown')}"
                            f"\n\nPage Body: {doc.page_content}"
                        )
                    
                    sys_msg = (
                        "Use the documents provided by the user to generate an interesting question-answer pair."
                        " Try to use both documents if possible, and rely more on the document bodies than the summary. Be specific!"
                        " Use the format:\nQuestion: (good question, 1-3 sentences, detailed)\n\nAnswer: (answer derived from the documents)"
                        " DO NOT SAY: \"Here is an interesting question pair\" or similar. FOLLOW FORMAT!"
                    )
                    usr_msg = f"Document1: {format_chunk(doc1)}\n\nDocument2: {format_chunk(doc2)}"
                    return ChatPromptTemplate.from_messages([('system', sys_msg), ('user', usr_msg)])
                
                def get_eval_prompt():
                    """Get evaluation prompt for LLM judge"""
                    eval_instruction = (
                        "Evaluate the following Question-Answer pair for human preference and consistency."
                        "\nAssume the first answer is a ground truth answer and has to be correct."
                        "\nAssume the second answer may or may not be true."
                        "\n[1] The first answer is extremely preferable, or the second answer heavily deviates."
                        "\n[2] The second answer does not contradict the first and significantly improves upon it."
                        "\n\nOutput Format:"
                        "\nJustification\n[2] if 2 is strongly preferred, [1] otherwise"
                        "\n\nQuestion-Answer Pair:"
                        "\n{input}\n\n"
                    )
                    return ChatPromptTemplate.from_messages([('system', eval_instruction), ('user', '{input}')])
                
                # Connect evaluation functionality
                start_eval_btn.click(
                    fn=run_rag_evaluation,
                    inputs=[eval_mode],
                    outputs=[eval_chatbot, eval_stats]
                )
            
            with gr.TabItem("📊 Chat with Data"):
                gr.Markdown("### Upload and Chat with Your Data")
                gr.Markdown("*📈 Upload CSV/Excel files to analyze your data and chat with an AI data scientist*")
                
                # Global state for data chat
                data_analysis_agent = gr.State(None)
                data_chat_agent = gr.State(None)
                temp_files = gr.State([])
                
                with gr.Row():
                    with gr.Column(scale=1):
                        data_file_input = gr.File(
                            label="Upload Data File",
                            file_types=[".csv", ".xlsx", ".xls"],
                            type="filepath"
                        )
                        
                        user_context_input = gr.Textbox(
                            label="Context/Problem Description (Optional)",
                            placeholder="Describe what you want to analyze or the problem you're trying to solve...",
                            lines=3
                        )
                        
                        target_variable_input = gr.Textbox(
                            label="Target Variable (Optional)",
                            placeholder="Name of the column you want to predict/analyze"
                        )
                        
                        analyze_btn = gr.Button("🔍 Analyze Data", variant="primary", size="lg")
                        data_status = gr.Textbox(label="Status", lines=2)
                    
                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.TabItem("📋 Analysis Results"):
                                analysis_output = gr.Markdown(
                                    value="*Upload a CSV or Excel file to get started with data analysis.*"
                                )
                            
                            with gr.TabItem("📊 Data Preview"):
                                data_preview = gr.Markdown(
                                    value="*Data preview will appear here after upload.*"
                                )
                            
                            with gr.TabItem("📈 Visualizations"):
                                data_plots = gr.Gallery(
                                    label="Generated Plots",
                                    show_label=True,
                                    elem_id="data_plots",
                                    columns=2,
                                    rows=2,
                                    height="auto"
                                )
                
                # Chat interface for data questions
                gr.Markdown("### 💬 Ask Questions About Your Data")
                
                with gr.Row():
                    data_chatbot = gr.Chatbot(
                        value=[],
                        elem_id="data_chatbot",
                        type="messages",
                        height=300,
                        show_label=False
                    )
                
                with gr.Row():
                    data_chat_input = gr.Textbox(
                        scale=4,
                        show_label=False,
                        placeholder="Ask questions about your data...",
                        container=False,
                    )
                    data_chat_btn = gr.Button("Send", variant="primary")
                
                def analyze_uploaded_data(file, context, target_var, current_temp_files):
                    if file is None:
                        return "❌ Please upload a data file", "*Upload a CSV or Excel file to get started.*", "*Data preview will appear here.*", [], None, None, current_temp_files
                    
                    try:
                        # Validate file
                        is_valid, message = validate_file_upload(file)
                        if not is_valid:
                            return f"❌ {message}", "*Upload a CSV or Excel file to get started.*", "*Data preview will appear here.*", [], None, None, current_temp_files
                        
                        # Save file temporarily
                        temp_file_path = save_uploaded_file(file)
                        new_temp_files = current_temp_files + [temp_file_path]
                        
                        # Create data request
                        data_request = DataRequest(
                            file_path=temp_file_path,
                            user_context=context if context.strip() else None,
                            target_variable=target_var if target_var.strip() else None
                        )
                        
                        # Initialize analysis agent with Enhanced LangGraph
                        analysis_agent = EnhancedLangGraphDataAnalysisAgent(llm, embedder)
                        
                        # Perform analysis
                        analysis_result = analysis_agent.analyze_dataset(data_request)
                        
                        # Format output
                        formatted_analysis = format_analysis_output(analysis_result)
                        
                        # Get data preview
                        data_preview_text = get_sample_data_info(analysis_agent.data_processor.data)
                        
                        # Prepare visualizations for gallery
                        plot_images = []
                        for plot_base64 in analysis_result.visualizations:
                            plot_images.append(f"data:image/png;base64,{plot_base64}")
                        
                        # Create chat agent with Enhanced LangGraph
                        chat_agent = EnhancedLangGraphDataChatAgent(llm, embedder, analysis_agent.data_processor)
                        
                        return (
                            "✅ Data analysis complete!",
                            formatted_analysis,
                            data_preview_text,
                            plot_images,
                            analysis_agent,
                            chat_agent,
                            new_temp_files
                        )
                        
                    except Exception as e:
                        logger.error(f"Error in data analysis: {str(e)}")
                        return f"❌ Error: {str(e)}", "*Upload a CSV or Excel file to get started.*", "*Data preview will appear here.*", [], None, None, current_temp_files
                
                def chat_with_data(message, history, chat_agent_state):
                    if not message.strip():
                        return history, ""
                    
                    if chat_agent_state is None:
                        history.append({"role": "user", "content": message})
                        history.append({"role": "assistant", "content": "Please upload and analyze a dataset first before asking questions."})
                        return history, ""
                    
                    try:
                        # Get response from chat agent
                        response = chat_agent_state.answer_question(message)
                        
                        # Update chat history
                        history.append({"role": "user", "content": message})
                        history.append({"role": "assistant", "content": response})
                        
                        return history, ""
                        
                    except Exception as e:
                        logger.error(f"Error in data chat: {str(e)}")
                        history.append({"role": "user", "content": message})
                        history.append({"role": "assistant", "content": f"Sorry, I encountered an error: {str(e)}"})
                        return history, ""
                
                # Event handlers
                analyze_btn.click(
                    fn=analyze_uploaded_data,
                    inputs=[data_file_input, user_context_input, target_variable_input, temp_files],
                    outputs=[data_status, analysis_output, data_preview, data_plots, data_analysis_agent, data_chat_agent, temp_files]
                )
                
                data_chat_input.submit(
                    fn=chat_with_data,
                    inputs=[data_chat_input, data_chatbot, data_chat_agent],
                    outputs=[data_chatbot, data_chat_input]
                )
                
                data_chat_btn.click(
                    fn=chat_with_data,
                    inputs=[data_chat_input, data_chatbot, data_chat_agent],
                    outputs=[data_chatbot, data_chat_input]
                )
    
    return demo

# Create retrieval and RAG chains
from langchain_core.runnables import RunnableLambda

retrieval_chain = (
    {'input': RunnableLambda(lambda x: x)}
    | RunnableAssign({
        'context': RunnableLambda(lambda d: d['input'])
        | RunnableLambda(lambda x: retrieve_documents(x))
        | RunnableLambda(lambda docs: docs2str(docs))
    })
)

generator_chain = RunnableLambda(lambda d: generate_response(d['context'], d['input']))
rag_chain = retrieval_chain | RunnableLambda(lambda d: {"output": generator_chain.invoke(d)}) | RunnableLambda(output_puller)

# Mount Gradio app to FastAPI
demo = get_demo()

@app.get("/")
async def read_root():
    return {"message": "RAG PDF Server is running. Access the Gradio interface at /gradio"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/upload_assignment/")
async def upload_assignment_endpoint(submission_zip: UploadFile = File(...), rubric_file: UploadFile = File(None)):
    """Accept a zip of student code + rubric and return evaluation report JSON."""
    if not submission_zip.filename.lower().endswith('.zip'):
        return JSONResponse(
            status_code=400,
            content={"message": "Submission must be a ZIP file"}
        )
    
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, submission_zip.filename)
    
    try:
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(submission_zip.file, buffer)
        
        rubric = {}
        if rubric_file:
            rubric_path = os.path.join(temp_dir, rubric_file.filename)
            with open(rubric_path, "wb") as buffer:
                shutil.copyfileobj(rubric_file.file, buffer)
            rubric = load_rubric(rubric_path)
        
        result = grade_submission(zip_path, rubric, llm)
        shutil.rmtree(temp_dir)
        return result
    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        logger.error(f"Error processing assignment: {get_traceback(e)}")
        return JSONResponse(
            status_code=500,
            content={"message": f"Error processing assignment: {str(e)}"}
        )

# Mount Gradio at /gradio path to avoid conflicts
app = gr.mount_gradio_app(app, demo, path="/gradio")

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
