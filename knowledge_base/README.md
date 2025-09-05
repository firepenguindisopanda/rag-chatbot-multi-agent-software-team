Knowledge Base for Multi-Agent Software Team

Structure:
- raw/: source PDFs or text files named using TEAMROLE_<ROLE>_knowledgebase*.pdf (case-insensitive).
- software_team_vectorstore/: persisted FAISS index (auto-created).
- manifest.json: file hash + ingestion metadata.

Ingestion is lazy: only runs for role on first need, then persists. Missing roles are ignored gracefully.
