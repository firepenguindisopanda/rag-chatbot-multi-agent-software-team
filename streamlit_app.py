"""Streamlit version of the RAG PDF Server UI previously implemented with Gradio.

Features migrated:
# Chat (Basic / RAG)
- PDF Upload (single + batch) with summary + suggested questions
- Quiz generation & answer evaluation
- Multi-Agent Software Team generation & save
- RAG Evaluation (LLM-as-a-Judge)
- Data Analysis & Chat (CSV/Excel)

Notes:
- Keeps FastAPI back-end in rag_pdf_server.py unchanged for now.
- This Streamlit app calls functions imported from rag_pdf_server where possible to avoid duplication.
- Some UI-specific helpers replicated with minor adaptation (progress, state mgmt).
- Gradio-specific return types replaced with simple primitives.

Run:
  streamlit run streamlit_app.py --server.port 8501
"""
from __future__ import annotations
import os
import io
import time
import random
import base64
import traceback
from datetime import datetime
from typing import List, Tuple

import streamlit as st

# Import functions & globals from rag_pdf_server (we reuse logic without re-writing)
# If rag_pdf_server imports gradio, that's fine; we only call its pure functions.
from rag_pdf_server import (
    process_pdf,
    pdf_uploader_with_progress as _gradio_pdf_uploader_with_progress,  # we will wrap
    batch_pdf_processor,
    retrieve_documents,
    generate_response,
    docs2str,
    generate_quiz_questions,
    evaluate_answer,
    generate_preliminary_questions,
    run_fixed_enhanced_multi_agent_collaboration,
    extract_document_metadata,
    generate_pdf_summary,
    llm,
    docstore,
)
import rag_pdf_server  # ensure dynamic access to docstore

# Data chat dependencies
from chat_with_data import (
    DataRequest,
    validate_file_upload,
    save_uploaded_file,
    format_analysis_output,
    get_sample_data_info,
)
from chat_with_data.enhanced_langgraph_agents import (
    EnhancedLangGraphDataAnalysisAgent,
    EnhancedLangGraphDataChatAgent,
)
from chat_with_data.data_processor import DataProcessor
from chat_with_data.vectorstore_manager import DataVectorStoreManager

def pdf_uploader_streamlit(uploaded_file):
    """Wrapper around existing progress-enabled function to show Streamlit progress UI.
    We mimic pdf_uploader_with_progress stages using st.progress and then call process_pdf directly.
    """
    if uploaded_file is None:
        return ("❌ **No file uploaded**", "", "")
    try:
        progress = st.progress(0, text="Starting...")
        # Save to temp file
        with st.spinner("Reading PDF..."):
            temp_dir = st.session_state.setdefault("_tmp_pdf_dir", os.path.join(os.getcwd(), "_tmp_uploads"))
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        progress.progress(15, text="Extracting text...")
        result = process_pdf(temp_path)
        progress.progress(90, text="Finalizing...")
        # Suggested questions
        questions = generate_preliminary_questions()
        questions_md = "\n".join([f"- {q}" for q in questions]) if questions else "None generated"
        progress.progress(100, text="Done")

        status = f"✅ **PDF Processed**\n\n**File:** {uploaded_file.name}\n**Chunks:** {result.get('chunks')}\n**Text Length:** {result.get('text_length', 'n/a')}"
        summary = result.get("summary", "No summary")
        questions_block = f"### Suggested Questions\n\n{questions_md}"
        return status, questions_block, f"### Document Summary\n\n{summary}"
    except Exception as e:
        return (f"❌ Error processing PDF: {e}", "", "")

def get_docstore_stats():
    ds = getattr(rag_pdf_server, 'docstore', None)
    if not ds or not getattr(ds, 'docstore', None):
        return None
    try:
        docs = list(ds.docstore._dict.values())
        titles = list({d.metadata.get('Title', 'Unknown') for d in docs})[:5]
        return {
            'chunks': len(docs),
            'sample_titles': titles,
        }
    except Exception:
        return None

################################################################################################
# Streamlit App Layout
################################################################################################

def main():
    st.set_page_config(page_title="RAG PDF Server (Streamlit)", layout="wide")
    st.title("📚 RAG PDF Server (Streamlit Edition)")
    st.caption("Migration from Gradio to Streamlit - feature parity version")

    # Global tabs
    tabs = st.tabs([
        "💬 Chat", "📄 Upload PDF", "🧠 Quiz", "🤖 Software Team", "🛠️ Evaluate RAG", "📊 Data Chat"
    ])

    ################################################################################################
    # Chat Tab
    with tabs[0]:
        st.subheader("Chat with Documents")
        mode = st.radio("Mode", ["Basic", "RAG"], index=1, horizontal=True)
        stats = get_docstore_stats()
        if mode == "RAG":
            if not stats:
                st.warning("No documents loaded. Upload PDFs first (see '📄 Upload PDF'). RAG mode disabled.")
            else:
                st.info(f"Using {stats['chunks']} indexed chunks from: {', '.join(stats['sample_titles'])}")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []  # list of dicts {role, content}

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_query = st.chat_input("Ask a question about your documents...")
        if user_query:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)
            with st.chat_message("assistant"):
                try:
                    if mode == "Basic" or not stats:
                        resp = llm.invoke(user_query).content
                    else:
                        docs = retrieve_documents(user_query)
                        ctx = docs2str(docs)
                        resp = generate_response(ctx, user_query)
                    st.markdown(resp)
                    st.session_state.chat_history.append({"role": "assistant", "content": resp})
                except Exception as e:
                    err = f"Error: {e}"
                    st.error(err)
                    st.session_state.chat_history.append({"role": "assistant", "content": err})
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.experimental_rerun()

    ################################################################################################
    # PDF Upload Tab
    with tabs[1]:
        st.subheader("Upload & Process PDFs")
        single, batch = st.tabs(["Single PDF", "Batch Processing"])
        with single:
            up_file = st.file_uploader("Select a PDF", type=["pdf"], key="single_pdf")
            if st.button("Process PDF", type="primary"):
                status, questions, summary = pdf_uploader_streamlit(up_file)
                st.markdown(status)
                st.markdown(questions)
                st.markdown(summary)
        with batch:
            batch_files = st.file_uploader("Select Multiple PDFs", type=["pdf"], accept_multiple_files=True, key="batch_pdf")
            if st.button("Process All PDFs", key="batch_btn"):
                if not batch_files:
                    st.warning("No files uploaded")
                else:
                    statuses = []
                    total_chunks = 0
                    summaries = []
                    for f in batch_files:
                        s, q, summ = pdf_uploader_streamlit(f)
                        statuses.append(s)
                        summaries.append(summ)
                        try:
                            parts = s.split("Chunks:")
                            if len(parts) > 1:
                                total_chunks += int(parts[1].split("\n")[0].strip())
                        except:  # noqa
                            pass
                    st.markdown("\n".join(statuses))
                    st.info(f"Total Chunks Added: {total_chunks}")
                    with st.expander("Batch Summaries"):
                        for summ in summaries:
                            st.markdown(summ)

    ################################################################################################
    # Quiz Tab
    with tabs[2]:
        st.subheader("Quiz Yourself")
        q_stats = get_docstore_stats()
        if not q_stats:
            st.warning("No documents indexed. Upload PDFs first to enable quiz generation.")
        else:
            st.info(f"Quiz will be generated from {q_stats['chunks']} chunks across documents: {', '.join(q_stats['sample_titles'])}")
        num_q = st.slider("Number of Questions", 1, 10, 5, disabled=not q_stats)
        if st.button("Generate Quiz", type="primary", disabled=not q_stats):
            questions, answers, status = generate_quiz_questions(num_q)
            st.session_state.quiz = {
                "questions": questions,
                "answers": answers,
                "index": 0,
                "scores": [],
                "status": status,
            }
        if "quiz" in st.session_state and st.session_state.quiz.get("questions"):
            qd = st.session_state.quiz
            st.markdown(f"**Question {qd['index']+1} of {len(qd['questions'])}:** {qd['questions'][qd['index']]}")
            user_ans = st.text_area("Your Answer", key=f"quiz_answer_{qd['index']}")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Submit Answer"):
                    fb, score = evaluate_answer(qd['questions'][qd['index']], user_ans, qd['answers'][qd['index']])
                    st.write(fb)
                    qd['scores'].append(score)
            with col2:
                if st.button("Show Correct Answer"):
                    st.info(qd['answers'][qd['index']])
            with col3:
                if st.button("Next Question"):
                    if qd['index'] < len(qd['questions']) - 1:
                        qd['index'] += 1
                        st.experimental_rerun()
                    else:
                        avg = sum(qd['scores'])/len(qd['scores']) if qd['scores'] else 0
                        st.success(f"Quiz Complete! Avg Score: {avg:.1f}/10")

    ################################################################################################
    # Software Team Tab
    with tabs[3]:
        st.subheader("Enhanced Multi-Agent Software Team")
        # Maintain description for side-panel inserts
        if "swteam_desc" not in st.session_state:
            st.session_state.swteam_desc = ""

        main_col, side_col = st.columns([3, 1], gap="large")

        with main_col:
            desc = st.text_area(
                "Project Description",
                height=200,
                value=st.session_state.get("swteam_desc", ""),
            )
            # Keep session in sync for side-panel inserts
            st.session_state["swteam_desc"] = desc
            file = st.file_uploader("Context File (optional)", type=["txt", "md", "pdf", "docx"], key="proj_file")
            if st.button("Generate Solution", type="primary"):
                file_content = ""
                if file:
                    try:
                        file_content = file.read().decode(errors="ignore")
                    except Exception:
                        pass
                with st.spinner("Running multi-agent collaboration..."):
                    try:
                        solution = run_fixed_enhanced_multi_agent_collaboration(llm, st.session_state.swteam_desc, file_content)
                        st.session_state.solution = solution
                    except Exception as e:
                        st.error(f"Error: {e}")
            if "solution" in st.session_state:
                st.markdown("### Solution Output")
                st.markdown(st.session_state.solution)
                if st.button("Save Solution"):
                    safe_description = "".join(c for c in (st.session_state.swteam_desc or "software_project")[:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"multi_agent_solution_{safe_description}_{ts}.md"
                    os.makedirs("solutions", exist_ok=True)
                    path = os.path.join("solutions", filename)
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(st.session_state.solution)
                    st.success(f"Saved to {path}")

        with side_col:
            st.markdown("### 💡 Example Prompts")
            st.caption("Expand and copy examples below")
            
            with st.expander("🌐 Task Manager Web App"):
                st.code(
                    "Create a task management web application for small teams with user authentication, project creation, "
                    "task assignment, real-time collaboration (comments/mentions), and progress tracking dashboards. Include role-based access control and activity logs.",
                    language=None
                )
            
            with st.expander("🛒 E-commerce REST API"):
                st.code(
                    "Design a RESTful API for an e-commerce platform supporting product catalog, search, cart, checkout, payments, "
                    "order tracking, and inventory management. Provide authentication, rate limiting, and OpenAPI docs.",
                    language=None
                )
            
            with st.expander("💪 Fitness Mobile App"):
                st.code(
                    "Build a fitness tracking mobile application with workout logging, progress visualization, social sharing, goals, "
                    "and integration with wearable devices. Include reminders and offline sync.",
                    language=None
                )
            
            with st.expander("📡 IoT Ingestion Pipeline"):
                st.code(
                    "Implement an IoT data ingestion pipeline handling device telemetry at scale with ingestion, validation, time-series storage, "
                    "stream processing (anomaly detection), and a dashboard for real-time monitoring.",
                    language=None
                )
            
            with st.expander("🏢 Microservices Booking System"):
                st.code(
                    "Develop a microservices-based booking system (search, reservation, payments, notifications) with saga pattern, "
                    "idempotency, and observability (metrics, tracing, logs). Provide deployment and scaling strategy.",
                    language=None
                )

    ################################################################################################
    # RAG Evaluation Tab
    with tabs[4]:
        st.subheader("RAG Evaluation (LLM-as-a-Judge)")
        mode_eval = st.radio("Evaluation Mode", ["Basic", "RAG"], index=1, horizontal=True)
        if st.button("Start Evaluation", type="primary"):
            if docstore is None:
                st.error("No documents uploaded.")
            else:
                messages = []
                num_points = 0
                num_questions = 4  # fewer for faster demo
                progress = st.progress(0, text="Starting evaluation...")
                for i in range(num_questions):
                    progress.progress(int((i/num_questions)*100), text=f"Question {i+1}/{num_questions}")
                    try:
                        # Reuse logic similar to rag_pdf_server.run_rag_evaluation
                        docs = list(docstore.docstore._dict.values())
                        if len(docs) < 2:
                            st.error("Not enough content for evaluation")
                            break
                        doc1, doc2 = random.sample(docs, 2)
                        synth_prompt = "Generate one QA pair about these excerpts:\n" + doc1.page_content[:500] + "\n---\n" + doc2.page_content[:500]
                        synth_resp = llm.invoke(synth_prompt).content
                        parts = synth_resp.split("\n")
                        question = parts[0][:500] if parts else "What is discussed?"
                        ground = parts[1][:500] if len(parts) > 1 else "Ground truth not clear."
                        if mode_eval == "Basic":
                            rag_ans = llm.invoke(question).content
                        else:
                            ctx_docs = retrieve_documents(question)
                            ctx = docs2str(ctx_docs)
                            rag_ans = generate_response(ctx, question)
                        judge_prompt = f"Question: {question}\nAnswer 1: {ground}\nAnswer 2: {rag_ans}\nWhich is better? Return [2] if second is as good or improves without contradiction else [1]."  # simplified
                        judge = llm.invoke(judge_prompt).content
                        better = "[2]" in judge
                        if better:
                            num_points += 1
                        messages.append((question, ground, rag_ans, judge, better))
                    except Exception as e:
                        messages.append((f"Error generating question {i+1}: {e}", '', '', '', False))
                progress.progress(100, text="Complete")
                passed = num_points/num_questions >= 0.6
                st.write(f"Score: {num_points}/{num_questions} -> {'PASS' if passed else 'FAIL'}")
                with st.expander("Detailed Results"):
                    for idx, (q,g,a,j,b) in enumerate(messages,1):
                        st.markdown(f"**Q{idx}:** {q}\n\n**GT:** {g}\n\n**RAG:** {a}\n\n**Judge:** {j}\n\n{'✅' if b else '❌'}")

    ################################################################################################
    # Data Chat Tab
    with tabs[5]:
        st.subheader("Data Upload, Analysis & Chat")
        st.caption("Upload a dataset for analysis. If already analyzed this session, you can continue chatting below.")
        data_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx", "xls"], key="data_file")
        context = st.text_area("Context / Problem Description", height=100)
        target_var = st.text_input("Target Variable (optional)")
        if st.button("Analyze Data", type="primary"):
            if data_file is None:
                st.error("Upload a data file")
            else:
                # Save temp
                tmp_dir = os.path.join(os.getcwd(), "_data_tmp")
                os.makedirs(tmp_dir, exist_ok=True)
                file_path = os.path.join(tmp_dir, data_file.name)
                with open(file_path, "wb") as f:
                    f.write(data_file.getbuffer())
                req = DataRequest(file_path=file_path, user_context=context or None, target_variable=target_var or None)
                analysis_agent = EnhancedLangGraphDataAnalysisAgent(llm, None)
                try:
                    analysis_result = analysis_agent.analyze_dataset(req)
                    st.session_state.analysis_agent = analysis_agent
                    st.session_state.chat_agent = EnhancedLangGraphDataChatAgent(llm, None, analysis_agent.data_processor)
                    st.markdown(format_analysis_output(analysis_result))
                    st.markdown(get_sample_data_info(analysis_agent.data_processor.data))
                except Exception as e:
                    st.error(f"Analysis error: {e}")
        st.divider()
        if "chat_agent" in st.session_state and st.session_state.chat_agent:
            prompt = st.text_input("Ask about your data", key="data_q")
            if st.button("Send", key="data_send"):
                try:
                    resp = st.session_state.chat_agent.answer_question(prompt)
                    st.markdown(f"**You:** {prompt}\n\n**AI:** {resp}")
                except Exception as e:
                    st.error(f"Chat error: {e}")
        else:
            if "analysis_agent" in st.session_state:
                st.info("Dataset loaded. Ask questions below.")
            else:
                st.info("Analyze data first to enable chat.")

if __name__ == "__main__":
    main()
