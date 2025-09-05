import os, json, hashlib, logging
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

KB_ROOT = "knowledge_base"
RAW_DIR = os.path.join(KB_ROOT, "raw")
INDEX_DIR = os.path.join(KB_ROOT, "software_team_vectorstore")
MANIFEST_PATH = os.path.join(KB_ROOT, "manifest.json")

ROLE_PREFIX = "TEAMROLE_"  # Expected prefix in filenames

class SoftwareTeamKnowledgeBase:
    def __init__(self, embedder):
        self.embedder = embedder
        self.vectorstore = None
        self.manifest = self._load_manifest()
        self._load_index_if_exists()

    def _load_manifest(self):
        if os.path.exists(MANIFEST_PATH):
            try:
                with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed loading manifest: {e}; starting fresh")
        return {"files": {}}

    def _save_manifest(self):
        os.makedirs(KB_ROOT, exist_ok=True)
        with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.manifest, f, indent=2)

    def _hash_file(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()

    def _load_index_if_exists(self):
        if os.path.isdir(INDEX_DIR):
            try:
                self.vectorstore = FAISS.load_local(INDEX_DIR, self.embedder, allow_dangerous_deserialization=True)
                logger.info("Loaded existing software team knowledge base index")
            except Exception as e:
                logger.warning(f"Could not load existing knowledge base index: {e}")

    def _ensure_vectorstore(self):
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents([], self.embedder)

    def ingest_role_documents(self, role_name: str) -> int:
        role_key = role_name.lower()
        if not os.path.isdir(RAW_DIR):
            logger.info("No raw knowledge base directory present; skipping")
            return 0

        matching_files = [f for f in os.listdir(RAW_DIR) if f.lower().startswith(f"{ROLE_PREFIX.lower()}{role_key}")]
        if not matching_files:
            return 0

        new_docs: List[Document] = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

        for fname in matching_files:
            fpath = os.path.join(RAW_DIR, fname)
            try:
                fhash = self._hash_file(fpath)
                manifest_entry = self.manifest['files'].get(fname)
                if manifest_entry and manifest_entry.get('hash') == fhash:
                    continue  # unchanged

                # load
                docs: List[Document] = []
                if fname.lower().endswith('.pdf'):
                    loader = PyPDFLoader(fpath)
                    docs = loader.load()
                elif fname.lower().endswith(('.txt', '.md', '.markdown')):
                    loader = TextLoader(fpath, encoding='utf-8')
                    docs = loader.load()
                else:
                    logger.info(f"Skipping unsupported file type: {fname}")
                    continue

                for d in docs:
                    d.metadata.update({
                        'kb_role': role_key,
                        'source_file': fname,
                        'kb_type': 'software_team_reference'
                    })

                # chunk
                chunked = splitter.split_documents(docs)
                new_docs.extend(chunked)

                # update manifest
                self.manifest['files'][fname] = {
                    'hash': fhash,
                    'chunks': len(chunked)
                }
            except Exception as e:
                logger.error(f"Failed ingesting {fname}: {e}")

        if new_docs:
            self._ensure_vectorstore()
            self.vectorstore.add_documents(new_docs)
            self.vectorstore.save_local(INDEX_DIR)
            self._save_manifest()
            logger.info(f"Ingested {len(new_docs)} new chunks for role {role_name}")
        return len(new_docs)

    def retrieve(self, role_name: str, query: str, k: int = 4) -> List[Document]:
        if self.vectorstore is None:
            # attempt lazy ingest for that role (in case files exist but index not built yet)
            self.ingest_role_documents(role_name)
            if self.vectorstore is None:
                return []
        try:
            docs = self.vectorstore.similarity_search(query, k=k*3)
            role_key = role_name.lower()
            filtered = [d for d in docs if d.metadata.get('kb_role') == role_key][:k]
            return filtered
        except Exception as e:
            logger.error(f"Knowledge base retrieval error: {e}")
            return []

# Singleton helper (simple module-level cache)
_kb_instance: Optional[SoftwareTeamKnowledgeBase] = None

def get_kb(embedder):
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = SoftwareTeamKnowledgeBase(embedder)
    return _kb_instance
