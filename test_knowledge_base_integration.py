"""Smoke test for knowledge base integration without actual PDFs.
Ensures orchestrator still runs when no knowledge base files exist.
"""
from multi_agent_software_team.streamlined_orchestrator import create_streamlined_software_team

class DummyLLM:
    def __init__(self):
        # minimal fake embedder for kb manager; returns fixed-size list
        class _Embedder:
            def embed_query(self, text: str):
                return [0.1]*1536
            def embed_documents(self, texts):
                return [[0.1]*1536 for _ in texts]
        self.embedder = _Embedder()
        self._step = 0
    def invoke(self, prompt: str):
        class R: pass
        r = R()
        # drive sequence deterministically
        order = [
            ("PRODUCT OWNER", "User stories here. HANDOFF TO ANALYST"),
            ("ANALYST", "Requirements listed. HANDOFF TO ARCHITECT"),
            ("ARCHITECT", "Architecture diagram. HANDOFF TO DEVELOPER"),
            ("DEVELOPER", "Implementation details. HANDOFF TO REVIEWER"),
            ("REVIEWER", "Code review findings. HANDOFF TO TESTER"),
            ("TESTER", "Test plan. HANDOFF TO TECH_WRITER"),
            ("TECH_WRITER", "Documentation complete. FINAL ANSWER"),
        ]
        content = order[min(self._step, len(order)-1)][1]
        self._step += 1
        r.content = content
        return r

def test_kb_smoke():
    team = create_streamlined_software_team(DummyLLM())
    result = team.run_example("Build a simple API.")
    assert result['success'] is True
    assert 'FINAL ANSWER' in ' '.join(result['agent_outputs'].values())
