# RAG PDF Document Chatbot

> Python Version `3.12.3`

This project provides a comprehensive RAG (Retrieval Augmented Generation) chatbot system that can:
1. Process and index PDF documents
2. Answer questions about the uploaded documents
3. Evaluate the quality of the RAG system
4. **NEW:** Generate complete software solutions using a multi-agent AI team
5. **NEW:** Chat with data from CSV/Excel files with AI-powered analysis

## Features

### 💬 Chat Interface
- Chat with your uploaded documents using RAG
- Basic chat mode for general conversations
- Contextual responses based on document content

### 📄 PDF Processing
- Upload and process PDF documents
- Automatic text extraction and chunking
- Vector store creation for efficient retrieval

### 🧠 Knowledge Testing
- Generate quizzes based on uploaded documents
- Automated answer evaluation
- Progress tracking and scoring

### 🤖 **Multi-Agent Software Team**
- **8 specialized AI agents** working together
- Complete software development lifecycle coverage
- From requirements to deployment documentation
- Customizable team composition for your needs
- **Professional Mermaid diagrams** in all outputs

### 📊 **Chat with Data (CSV/Excel Analysis)**
- Upload CSV/Excel files for instant analysis
- AI-powered data insights and recommendations
- Interactive visualizations and plots
- ML model suggestions and metrics recommendations
- Chat interface to ask questions about your data

## Components

### 1. Vector Store Management (07_vectorstores.ipynb)
This notebook demonstrates how to:
- Create and manage vector stores for document retrieval
- Implement conversation memory using vector stores
- Create a RAG pipeline for document-based Q&A

### 2. RAG Evaluation (08_evaluation.ipynb)
This notebook shows how to:
- Evaluate the performance of your RAG pipeline
- Generate synthetic question-answer pairs
- Implement LLM-as-a-judge evaluation metrics

### 3. PDF Upload & RAG Server (rag_pdf_server.py)
This script provides:
- A FastAPI server with PDF upload functionality
- A Gradio interface for chatting and document upload (legacy UI)
- RAG implementation using the uploaded documents
- **NEW:** Multi-agent software team integration

### 3b. Streamlit UI (streamlit_app.py)
The project now includes a **Streamlit** alternative to the Gradio UI offering feature parity:
- Chat (Basic / RAG)
- PDF upload (single + batch) with summaries & suggested questions
- Quiz generation & answer evaluation
- Multi-Agent Software Team solution generation & saving
- RAG Evaluation (LLM-as-a-Judge) – lightweight version
- Data Analysis & Chat (CSV/Excel)

Run the Streamlit interface:
```
streamlit run streamlit_app.py --server.port 8501
```
Then open: http://localhost:8501

Legacy Gradio interface is still available via `python rag_pdf_server.py` while tests reference it. You may remove Gradio later once tests are updated.

### 4. **Multi-Agent Software Team Module**
Located in `multi_agent_software_team/`, this module provides:
- **8 AI Agents**: Product Owner, Analyst, Architect, Developer, Tester, Designer, Reviewer, Tech Writer
- **Complete Workflow**: From requirements gathering to final documentation
- **Flexible Team Composition**: Select only the agents you need
- **Contextual Intelligence**: Each agent builds upon previous work
- **Professional Output**: Industry-standard deliverables for each role
- **Mermaid Diagrams**: Beautiful technical diagrams in all relevant outputs

#### AI Team Roles:
- 📋 **Product Owner**: User stories, acceptance criteria, business requirements
- 🔍 **Analyst**: Technical specifications, functional requirements
- 🏗️ **Architect**: System design, APIs, infrastructure planning
- 💻 **Developer**: Code implementation, database design, technical setup
- 🧪 **Tester**: Test plans, quality assurance, automation scripts
- 🎨 **Designer**: System diagrams, technical visualizations (with Mermaid)
- 👀 **Reviewer**: Code review, security analysis, best practices
- 📝 **Tech Writer**: Documentation, user guides, deployment instructions

### 5. **Chat with Data Module**
Located in `chat_with_data/`, this module provides:
- **CSV/Excel File Support**: Upload and analyze tabular data
- **AI Data Scientist**: Automated insights and recommendations  
- **Interactive Visualizations**: Charts, plots, and data exploration
- **ML Model Suggestions**: Recommendations for machine learning approaches
- **Chat Interface**: Ask questions about your data in natural language
- **Context-Aware Analysis**: Provide business context for targeted insights

## Setup and Usage

### Prerequisites
Make sure you have the required packages installed:
```
pip install -r requirements.txt
```

Or manually install:
```
pip install langchain langchain-nvidia-ai-endpoints gradio rich arxiv pymupdf faiss-cpu pydantic
```

### Running the PDF Upload & RAG Server (Gradio legacy)
```
python rag_pdf_server.py
```

This starts:
- Gradio interface mounted at http://localhost:8000/gradio (see root JSON at http://localhost:8000)
- FastAPI endpoints for programmatic access

### Running the Streamlit Interface (Recommended)
```
streamlit run streamlit_app.py --server.port 8501
```
Open http://localhost:8501 for the modern interface.

### Using the Interfaces
#### Streamlit Tabs
1. 💬 Chat – conversational QA (Basic / RAG)
2. 📄 Upload PDF – single & batch document ingestion
3. 🧠 Quiz – generate and evaluate knowledge questions
4. 🤖 Software Team – multi-agent solution synthesis
5. 🛠️ Evaluate RAG – quick effectiveness check
6. 📊 Data Chat – analyze & converse with tabular data

#### Gradio Tabs (Legacy)
1. **📄 Upload PDF tab**: Upload and process documents
2. **💬 Chat tab**: Ask questions about your documents (RAG mode) or general chat (Basic mode)
3. **🧠 Test Your Understanding tab**: Take quizzes based on uploaded documents
4. **🤖 Software Team tab**: Generate complete software solutions using AI agents
5. **📊 Chat with Data tab**: Upload CSV/Excel files and analyze data with AI

#### Multi-Agent Software Team Usage:
1. Describe your software project in the text box or upload a requirements file
2. Select the AI agents you want on your team (minimum: Product Owner + Developer)
3. Click "🚀 Generate Solution" to start the collaborative process
4. Review comprehensive deliverables from each selected agent

**Example Project Description:**
```
Create a task management web application for small teams with features like:
- User authentication and roles
- Project creation and assignment
- Real-time collaboration
- Progress tracking and reporting
- Integration with email notifications
```

#### Chat with Data Usage:
1. Upload a CSV or Excel file containing your data
2. Optionally provide context about what you want to analyze
3. Click "🔍 Analyze Data" to get comprehensive insights
4. Review the generated analysis, visualizations, and ML recommendations
5. Use the chat interface to ask specific questions about your data

### API Endpoints
- `/upload_pdf/` - POST a PDF file to add it to the vector store
- `/retriever/` - POST a query to retrieve relevant document chunks
- `/generator/` - POST context and query to generate a response
- `/basic_chat/` - POST a query for regular chat without document retrieval

## Implementation Details

The system works by:
1. Processing PDFs into text chunks
2. Creating embeddings for each chunk using NVIDIA's embedding model
3. Storing embeddings in a FAISS vector store
4. Retrieving relevant chunks based on query similarity
5. Generating responses using an LLM with the retrieved context

The evaluation system assesses the quality of the RAG pipeline by:
1. Creating synthetic question-answer pairs from the documents
2. Comparing RAG-generated answers to ground truth answers
3. Using an LLM to judge the quality of the responses

## Note
Remember to download and save your vector store (docstore_index.tgz) after uploading documents to preserve your index for future use.
