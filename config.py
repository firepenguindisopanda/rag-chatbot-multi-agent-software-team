"""
Configuration and setup for the enhanced LangGraph RAG system.
"""

import os
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class SystemConfig:
    """Configuration class for the enhanced RAG system."""
    
    # Vectorstore paths (separate for different data types)
    PDF_VECTORSTORE_PATH = "docstore_index"
    PDF_VECTORSTORE_ARCHIVE = "docstore_index.tgz"
    
    DATA_VECTORSTORE_PATH = "data_vectorstore_index" 
    DATA_VECTORSTORE_ARCHIVE = "data_vectorstore_index.tgz"
    
    # API Configuration
    # Removed hardcoded key. Provide via environment, .env, or secrets manager.
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
    
    # Model Configuration
    EMBEDDING_MODEL = "nvidia/nv-embed-v1"
    LLM_MODEL = "meta/llama-3.1-8b-instruct"
    
    # LangGraph Configuration
    MAX_ITERATIONS = 3
    MAX_AGENTS_PER_TEAM = 8
    
    # Data Processing Configuration
    MAX_FILE_SIZE_MB = 50
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    
    # Chat Configuration
    MAX_CHAT_HISTORY = 10
    MAX_CONTEXT_LENGTH = 4000
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate the system configuration."""
        if not cls.NVIDIA_API_KEY:
            logger.warning("NVIDIA API key not set. Using mock LLM for testing.")
            return False
        
        # Check if required directories exist
        os.makedirs("temp", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        return True
    
    @classmethod
    def get_system_info(cls) -> dict:
        """Get system configuration information."""
        return {
            "pdf_vectorstore": cls.PDF_VECTORSTORE_PATH,
            "data_vectorstore": cls.DATA_VECTORSTORE_PATH,
            "embedding_model": cls.EMBEDDING_MODEL,
            "llm_model": cls.LLM_MODEL,
            "max_file_size": f"{cls.MAX_FILE_SIZE_MB}MB",
            "api_configured": bool(cls.NVIDIA_API_KEY)
        }

# Initialize configuration
config = SystemConfig()
config.validate_config()

logger.info("Enhanced LangGraph RAG system configuration loaded")
logger.info(f"System info: {config.get_system_info()}")
