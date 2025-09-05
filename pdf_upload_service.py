from flask import Flask, request, jsonify
import os
import shutil
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
import gradio as gr
import tempfile

app = Flask(__name__)

# Initialize embeddings and store
embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")
docstore = None

# Load existing docstore if available
try:
    os.system("tar xzvf docstore_index.tgz")
    docstore = FAISS.load_local("docstore_index", embedder, allow_dangerous_deserialization=True)
    print("Loaded existing docstore")
except Exception as e:
    print(f"Could not load existing docstore: {e}")
    # Create a new one
    docstore = FAISS(
        embedding_function=embedder,
        index=None,
        docstore=None
    )

def process_pdf(pdf_path):
    # Open the PDF file
    doc = fitz.open(pdf_path)
    
    # Extract text from each page
    text_content = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text_content += page.get_text()
    
    # Create metadata
    metadata = {
        "Title": os.path.basename(pdf_path),
        "Source": "User uploaded PDF",
        "Published": "2025-08-01"  # Current date as example
    }
    
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
    
    # Add chunks to vector store
    global docstore
    docstore.add_documents(chunks)
    
    # Save updated docstore
    docstore.save_local("docstore_index")
    os.system("tar czvf docstore_index.tgz docstore_index")
    
    return {
        "status": "success",
        "message": f"PDF processed with {len(chunks)} chunks added to vector store",
        "title": metadata["Title"]
    }

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected"}), 400
    
    if file and file.filename.endswith('.pdf'):
        # Save the file temporarily
        temp_dir = tempfile.mkdtemp()
        pdf_path = os.path.join(temp_dir, file.filename)
        file.save(pdf_path)
        
        try:
            result = process_pdf(pdf_path)
            # Clean up
            shutil.rmtree(temp_dir)
            return jsonify(result), 200
        except Exception as e:
            # Clean up
            shutil.rmtree(temp_dir)
            return jsonify({"status": "error", "message": str(e)}), 500
    else:
        return jsonify({"status": "error", "message": "Only PDF files are allowed"}), 400

def pdf_uploader(file):
    if file is None:
        return "No file uploaded"
    
    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, file.name)
    
    with open(pdf_path, "wb") as f:
        f.write(file)
    
    try:
        result = process_pdf(pdf_path)
        shutil.rmtree(temp_dir)
        return f"PDF processed: {result['message']}"
    except Exception as e:
        shutil.rmtree(temp_dir)
        return f"Error processing PDF: {str(e)}"

# Gradio interface for PDF upload
def create_upload_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Upload PDF to Vector Store")
        with gr.Row():
            pdf_input = gr.File(label="Upload PDF Document")
            upload_button = gr.Button("Process PDF")
        output = gr.Textbox(label="Status")
        
        upload_button.click(
            fn=pdf_uploader,
            inputs=pdf_input,
            outputs=output
        )
    
    return demo

if __name__ == "__main__":
    # Gradio interface
    upload_interface = create_upload_interface()
    upload_interface.launch(server_name="0.0.0.0", server_port=8091)
    
    # Flask API
    app.run(host='0.0.0.0', port=8092)
