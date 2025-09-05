# RAG System Evaluation Feature

## 📊 Overview

A new **"📊 Evaluate RAG"** tab has been added to your RAG PDF Server that implements LLM-as-a-Judge evaluation metrics to assess your RAG system's performance.

## 🎯 How It Works

The evaluation system follows these steps:

1. **Document Sampling**: Randomly selects document chunks from your uploaded PDFs
2. **Synthetic QA Generation**: Creates question-answer pairs using selected documents as ground truth
3. **RAG Response**: Your RAG system answers the same questions using its retrieval and generation pipeline
4. **Judge Evaluation**: An LLM judge compares RAG responses against the synthetic ground truth answers
5. **Scoring**: Calculates percentage of questions where RAG outperformed the baseline

## 🚀 How to Use

### Step 1: Upload Documents
- Go to the **"📄 Upload PDF"** tab
- Upload one or more PDF documents
- Ensure you have uploaded enough content (at least 10 document chunks required)

### Step 2: Start Evaluation
- Navigate to the **"📊 Evaluate RAG"** tab
- Choose evaluation mode:
  - **RAG**: Tests your full RAG pipeline (recommended)
  - **Basic**: Tests basic LLM without retrieval (baseline comparison)
- Click **"🎓 Start Evaluation"**

### Step 3: Review Results
- The system will generate **8 evaluation questions**
- Watch the real-time evaluation process in the chat interface
- Each question shows:
  - Generated synthetic question
  - Ground truth answer
  - Your RAG system's response
  - Judge's evaluation and score
- Final results show overall performance percentage

## 📈 Scoring System

**Judge Scoring:**
- **[1]** - Baseline answer is preferred OR RAG answer has significant issues
- **[2]** - RAG answer is better and doesn't contradict the baseline

**Pass Threshold:** 
- **60%** (5 out of 8 questions must score [2])
- ✅ **PASSED**: Your RAG system performs well
- ❌ **NEEDS IMPROVEMENT**: Consider reviewing document processing and retrieval strategy

## 🔧 Technical Implementation

The evaluation feature is based on:
- **LLM-as-a-Judge** methodology from academic research
- **Pairwise comparison** between synthetic and RAG-generated answers
- **Synthetic data generation** for consistent, repeatable evaluation
- **Context-aware evaluation** using document content as ground truth

## 📚 Reference Files

This implementation draws from:
- `08_evaluation.ipynb` - Core evaluation methodology
- `frontend_block.py` - Gradio interface patterns
- Academic paper: ["LLM-as-a-Judge"](https://arxiv.org/abs/2306.05685)

## 🛠️ Troubleshooting

**Common Issues:**

1. **"❌ No documents found"**
   - Solution: Upload at least one PDF in the "📄 Upload PDF" tab first

2. **"❌ Need at least 10 document chunks"**
   - Solution: Upload longer documents or multiple PDFs to reach minimum chunk requirement

3. **Evaluation takes a long time**
   - Expected: Each evaluation generates 8 questions and can take 2-5 minutes depending on LLM response times

4. **Low scores consistently**
   - Check document quality and relevance
   - Ensure PDFs contain substantial text content
   - Consider adjusting chunk size or retrieval parameters

## 🎉 Benefits

- **Objective Assessment**: Quantitative metrics for RAG performance
- **Continuous Improvement**: Regular testing helps optimize your system
- **Comparable Results**: Standardized evaluation across different configurations
- **Quality Assurance**: Identifies potential issues before production deployment

---

*This evaluation system helps ensure your RAG pipeline performs reliably and provides high-quality responses to user queries.*
