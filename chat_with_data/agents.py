from typing import List, Optional, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from .schemas import DataRequest, DataSummary, AnalysisResult, DataAnalysisType
from .data_processor import DataProcessor, DataVisualizer, MLModelRecommender
import logging
import json

logger = logging.getLogger(__name__)

class DataAnalysisAgent:
    """AI agent for data analysis and insights generation."""
    
    def __init__(self, llm):
        self.llm = llm
        self.data_processor = DataProcessor()
        self.visualizer = None
        self.recommender = None
    
    def analyze_dataset(self, data_request: DataRequest) -> AnalysisResult:
        """Perform comprehensive data analysis."""
        try:
            # Load and process data
            self.data_processor.load_data(data_request.file_path)
            self.data_processor.clean_data()
            data_summary = self.data_processor.analyze_data_structure()
            
            # Detect analysis type
            analysis_type = self.data_processor.detect_analysis_type(
                data_request.user_context, 
                data_request.target_variable
            )
            
            # Create visualizer and recommender
            self.visualizer = DataVisualizer(self.data_processor.data)
            self.recommender = MLModelRecommender(data_summary, analysis_type)
            
            # Generate visualizations
            overview_plots = self.visualizer.create_overview_plots()
            target_plots = []
            if data_request.target_variable:
                target_plots = self.visualizer.create_target_analysis(data_request.target_variable)
            
            # Get ML recommendations
            recommended_models = self.recommender.recommend_models()
            suggested_metrics = self.recommender.recommend_metrics()
            
            # Generate insights using LLM
            insights = self._generate_insights(data_summary, analysis_type, data_request)
            
            # Generate summary
            summary = self._generate_summary(data_summary, analysis_type, data_request)
            
            # Generate next steps
            next_steps = self._generate_next_steps(analysis_type, data_summary)
            
            return AnalysisResult(
                summary=summary,
                visualizations=overview_plots + target_plots,
                insights=insights,
                recommended_ml_models=recommended_models,
                suggested_metrics=suggested_metrics,
                next_steps=next_steps
            )
            
        except Exception as e:
            logger.error(f"Error in data analysis: {str(e)}")
            return AnalysisResult(
                summary=f"Error analyzing data: {str(e)}",
                visualizations=[],
                insights=[],
                recommended_ml_models=[],
                suggested_metrics=[],
                next_steps=[]
            )
    
    def _generate_insights(self, data_summary: DataSummary, analysis_type: DataAnalysisType, data_request: DataRequest) -> List[str]:
        """Generate insights using LLM."""
        
        # Prepare data context
        context = {
            "shape": data_summary.shape,
            "columns": data_summary.columns[:10],  # Limit to first 10 columns
            "data_types": len(data_summary.numerical_columns), 
            "categorical_count": len(data_summary.categorical_columns),
            "missing_values": sum(data_summary.missing_values.values()),
            "analysis_type": analysis_type.value
        }
        
        prompt = ChatPromptTemplate.from_template(
            """You are a data scientist analyzing a dataset. Based on the following information, provide 3-4 key insights about the data.

DATASET INFO:
- Shape: {shape[0]} rows, {shape[1]} columns
- Numerical columns: {data_types}
- Categorical columns: {categorical_count}
- Total missing values: {missing_values}
- Analysis type: {analysis_type}
- User context: {user_context}

Provide insights as a list of 3-4 bullet points. Each insight should be:
1. Specific and actionable
2. Based on the data characteristics
3. Relevant to the analysis type
4. Keep each insight to 1-2 sentences maximum

Focus on data quality, patterns, and recommendations for analysis approach.
"""
        )
        
        try:
            response = self.llm.invoke(prompt.format(
                shape=context["shape"],
                data_types=context["data_types"],
                categorical_count=context["categorical_count"],
                missing_values=context["missing_values"],
                analysis_type=context["analysis_type"],
                user_context=data_request.user_context or "Not specified"
            ))
            
            # Parse insights from response
            content = response.content
            insights = []
            for line in content.split('\n'):
                line = line.strip()
                if line and (line.startswith('•') or line.startswith('-') or line.startswith('*')):
                    insight = line.lstrip('•-* ').strip()
                    if insight:
                        insights.append(insight)
            
            return insights[:4]  # Limit to 4 insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return [
                f"Dataset has {data_summary.shape[0]} rows and {data_summary.shape[1]} columns",
                f"Contains {len(data_summary.numerical_columns)} numerical and {len(data_summary.categorical_columns)} categorical features",
                f"Analysis type detected: {analysis_type.value}",
                "Consider data preprocessing before modeling"
            ]
    
    def _generate_summary(self, data_summary: DataSummary, analysis_type: DataAnalysisType, data_request: DataRequest) -> str:
        """Generate concise summary of the analysis."""
        
        prompt = ChatPromptTemplate.from_template(
            """Provide a concise 2-3 sentence summary of this dataset analysis:

Dataset: {shape[0]} rows, {shape[1]} columns
Analysis type: {analysis_type}
User goal: {user_context}
Missing data: {missing_percent:.1f}%

Summary should be professional and actionable. Maximum 3 sentences.
"""
        )
        
        try:
            missing_percent = (sum(data_summary.missing_values.values()) / (data_summary.shape[0] * data_summary.shape[1])) * 100
            
            response = self.llm.invoke(prompt.format(
                shape=data_summary.shape,
                analysis_type=analysis_type.value,
                user_context=data_request.user_context or "exploratory data analysis",
                missing_percent=missing_percent
            ))
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return f"Dataset contains {data_summary.shape[0]} rows and {data_summary.shape[1]} columns. Analysis type: {analysis_type.value}. Ready for {analysis_type.value} modeling."
    
    def _generate_next_steps(self, analysis_type: DataAnalysisType, data_summary: DataSummary) -> List[str]:
        """Generate actionable next steps."""
        
        steps = []
        
        # Common preprocessing steps
        if sum(data_summary.missing_values.values()) > 0:
            steps.append("Handle missing values through imputation or removal")
        
        if len(data_summary.categorical_columns) > 0:
            steps.append("Encode categorical variables for modeling")
        
        # Analysis-specific steps
        if analysis_type == DataAnalysisType.REGRESSION:
            steps.extend([
                "Check for linear relationships between features and target",
                "Consider feature scaling for better model performance",
                "Split data into train/validation/test sets"
            ])
        elif analysis_type == DataAnalysisType.CLASSIFICATION:
            steps.extend([
                "Check class balance in target variable",
                "Consider feature selection techniques",
                "Implement cross-validation for model evaluation"
            ])
        elif analysis_type == DataAnalysisType.CLUSTERING:
            steps.extend([
                "Standardize features for distance-based algorithms",
                "Determine optimal number of clusters",
                "Validate clustering results with domain knowledge"
            ])
        elif analysis_type == DataAnalysisType.TIME_SERIES:
            steps.extend([
                "Check for trend and seasonality patterns",
                "Test for stationarity",
                "Consider lag features for modeling"
            ])
        else:  # EXPLORATORY
            steps.extend([
                "Perform correlation analysis between variables",
                "Identify potential outliers",
                "Consider dimensionality reduction for visualization"
            ])
        
        return steps[:5]  # Limit to 5 steps

class DataChatAgent:
    """AI agent for chatting about data using RAG."""
    
    def __init__(self, llm, data_processor: DataProcessor):
        self.llm = llm
        self.data_processor = data_processor
        self.conversation_history = []
    
    def answer_question(self, question: str) -> str:
        """Answer questions about the loaded dataset."""
        if self.data_processor.data is None:
            return "No dataset is currently loaded. Please upload a CSV or Excel file first."
        
        # Get data context
        data_context = self._get_data_context()
        
        prompt = ChatPromptTemplate.from_template(
            """You are a data analyst assistant. Answer the user's question about their dataset based on the following information:

DATASET CONTEXT:
{data_context}

CONVERSATION HISTORY:
{conversation_history}

USER QUESTION: {question}

Provide a clear, specific answer based on the data. If you need to perform calculations, show your reasoning. Keep responses concise but informative (max 3-4 sentences).
"""
        )
        
        try:
            # Format conversation history
            history_text = ""
            for i, msg in enumerate(self.conversation_history[-4:]):  # Last 4 messages
                history_text += f"{msg['role']}: {msg['content']}\n"
            
            response = self.llm.invoke(prompt.format(
                data_context=data_context,
                conversation_history=history_text,
                question=question
            ))
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append({"role": "assistant", "content": response.content})
            
            # Keep only last 10 messages
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return f"I encountered an error while processing your question: {str(e)}"
    
    def _get_data_context(self) -> str:
        """Get relevant context about the current dataset."""
        if self.data_processor.data is None:
            return "No data loaded"
        
        data = self.data_processor.data
        
        # Basic statistics
        context = f"Dataset shape: {data.shape[0]} rows, {data.shape[1]} columns\n"
        context += f"Columns: {', '.join(data.columns.tolist()[:10])}{'...' if len(data.columns) > 10 else ''}\n"
        
        # Numerical columns summary
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            context += f"Numerical columns: {', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''}\n"
            context += f"Sample statistics:\n{data[numeric_cols].describe().round(2).to_string()}\n"
        
        # Categorical columns
        cat_cols = data.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            context += f"Categorical columns: {', '.join(cat_cols[:5])}{'...' if len(cat_cols) > 5 else ''}\n"
        
        # Missing values
        missing = data.isnull().sum()
        if missing.sum() > 0:
            context += f"Missing values: {missing[missing > 0].to_dict()}\n"
        
        return context
