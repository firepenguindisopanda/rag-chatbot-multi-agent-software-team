"""
Enhanced LangGraph-based agents for CSV/Excel data analysis with vectorstore integration.
Provides sophisticated data analysis and chat capabilities.
"""

import json
import logging
import pandas as pd
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Union
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.tools import PythonREPLTool
from langchain.agents.agent_types import AgentType

from .data_processor import DataProcessor, DataVisualizer, MLModelRecommender
from .vectorstore_manager import DataVectorStoreManager
from .schemas import DataRequest, DataSummary, AnalysisResult, DataAnalysisType

logger = logging.getLogger(__name__)

class DataAnalysisError(Exception):
    """Custom exception for data analysis errors."""
    pass

class DataAnalysisState(TypedDict):
    """State for enhanced data analysis workflow."""
    messages: Annotated[List, add_messages]
    data_summary: Optional[DataSummary]
    analysis_type: Optional[DataAnalysisType]
    insights: List[str]
    plots: List[str]
    recommendations: Dict[str, List[str]]
    current_step: str
    error: Optional[str]
    file_path: Optional[str]
    user_context: Optional[str]
    target_variable: Optional[str]
    vectorstore_status: Optional[str]

class DataChatState(TypedDict):
    """State for enhanced data chat workflow."""
    messages: Annotated[List, add_messages]
    query: str
    response: str
    vectorstore_context: List[str]
    data_context: Optional[str]
    pandas_agent_response: Optional[str]

class EnhancedLangGraphDataAnalysisAgent:
    """Enhanced LangGraph-based agent for comprehensive data analysis with vectorstore."""
    
    def __init__(self, llm, embedder):
        self.llm = llm
        self.embedder = embedder
        self.data_processor = DataProcessor()
        self.vectorstore_manager = DataVectorStoreManager(embedder)
        self.vectorstore_manager.load_existing_vectorstore()
        self.analysis_graph = self._build_analysis_graph()
        self.pandas_agent = None
    
    def analyze_dataset(self, data_request: DataRequest) -> AnalysisResult:
        """Perform comprehensive data analysis using LangGraph workflow."""
        try:
            # Initialize state
            initial_state = DataAnalysisState(
                messages=[
                    HumanMessage(content=f"file_path:{data_request.file_path}"),
                    HumanMessage(content=f"context:{data_request.user_context or ''}"),
                    HumanMessage(content=f"target_variable:{data_request.target_variable or ''}")
                ],
                data_summary=None,
                analysis_type=None,
                insights=[],
                plots=[],
                recommendations={},
                current_step="start",
                error=None,
                file_path=data_request.file_path,
                user_context=data_request.user_context,
                target_variable=data_request.target_variable,
                vectorstore_status=None
            )
            
            # Run the workflow
            final_state = self.analysis_graph.invoke(initial_state)
            
            if final_state.get("error"):
                raise DataAnalysisError(final_state["error"])
            
            # Create pandas agent for the loaded data
            if self.data_processor.data is not None:
                self.pandas_agent = create_pandas_dataframe_agent(
                    self.llm,
                    self.data_processor.data,
                    agent_type=AgentType.OPENAI_FUNCTIONS,
                    verbose=True,
                    allow_dangerous_code=True
                )
            
            # Generate summary
            summary = self._generate_analysis_summary(final_state)
            
            return AnalysisResult(
                summary=summary,
                visualizations=final_state.get("plots", []),
                insights=final_state.get("insights", []),
                recommended_ml_models=final_state.get("recommendations", {}).get("models", []),
                suggested_metrics=final_state.get("recommendations", {}).get("metrics", []),
                next_steps=self._generate_next_steps(final_state)
            )
            
        except Exception as e:
            logger.error(f"Error in enhanced data analysis: {str(e)}")
            return AnalysisResult(
                summary=f"Error in data analysis: {str(e)}",
                visualizations=[],
                insights=[],
                recommended_ml_models=[],
                suggested_metrics=[],
                next_steps=[]
            )
    
    def _build_analysis_graph(self) -> StateGraph:
        """Build the enhanced data analysis workflow graph."""
        workflow = StateGraph(DataAnalysisState)
        
        # Add nodes
        workflow.add_node("start", self._start_analysis)
        workflow.add_node("load_data", self._load_data)
        workflow.add_node("add_to_vectorstore", self._add_to_vectorstore)
        workflow.add_node("detect_analysis", self._detect_analysis_type)
        workflow.add_node("generate_insights", self._generate_insights)
        workflow.add_node("create_plots", self._create_visualizations)
        workflow.add_node("get_recommendations", self._get_ml_recommendations)
        workflow.add_node("finalize", self._finalize_analysis)
        
        # Add edges
        workflow.set_entry_point("start")
        workflow.add_edge("start", "load_data")
        workflow.add_edge("load_data", "add_to_vectorstore")
        workflow.add_edge("add_to_vectorstore", "detect_analysis")
        workflow.add_edge("detect_analysis", "generate_insights")
        workflow.add_edge("generate_insights", "create_plots")
        workflow.add_edge("create_plots", "get_recommendations")
        workflow.add_edge("get_recommendations", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _start_analysis(self, state: DataAnalysisState) -> DataAnalysisState:
        """Start the analysis workflow."""
        state["current_step"] = "starting"
        state["messages"].append(SystemMessage(content="Starting enhanced data analysis workflow..."))
        return state
    
    def _load_data(self, state: DataAnalysisState) -> DataAnalysisState:
        """Load and process the data."""
        try:
            if not state["file_path"]:
                state["error"] = "No file path provided"
                return state
            
            # Load data
            self.data_processor.load_data(state["file_path"])
            self.data_processor.clean_data()
            state["data_summary"] = self.data_processor.analyze_data_structure()
            state["current_step"] = "data_loaded"
            
            logger.info(f"Successfully loaded data with shape: {self.data_processor.data.shape}")
            
        except Exception as e:
            state["error"] = f"Error loading data: {str(e)}"
            logger.error(f"Error loading data: {str(e)}")
        
        return state
    
    def _add_to_vectorstore(self, state: DataAnalysisState) -> DataAnalysisState:
        """Add data to the dedicated CSV/Excel vectorstore."""
        try:
            if self.data_processor.data is None:
                state["error"] = "No data to add to vectorstore"
                return state
            
            file_info = {
                "filename": state["file_path"].split("/")[-1] if state["file_path"] else "unknown",
                "user_context": state["user_context"],
                "target_variable": state["target_variable"]
            }
            
            # Add to vectorstore (separate from PDF docstore)
            result = self.vectorstore_manager.add_data_to_vectorstore(
                self.data_processor.data, 
                file_info
            )
            
            if result["status"] == "success":
                state["vectorstore_status"] = "success"
                state["current_step"] = "vectorstore_updated"
                logger.info(f"Added data to vectorstore: {result['message']}")
            else:
                # Log the error but don't fail the entire analysis
                logger.warning(f"Vectorstore update failed: {result.get('message', 'Unknown error')}")
                state["vectorstore_status"] = "failed"
                state["current_step"] = "vectorstore_skipped"
                # Continue with analysis even if vectorstore fails
            
        except Exception as e:
            # Log the error but don't fail the entire analysis
            error_msg = f"Error adding to vectorstore: {str(e)}"
            logger.warning(error_msg)
            state["vectorstore_status"] = "failed"
            state["current_step"] = "vectorstore_skipped"
            # Don't set error in state - continue with analysis
        
        return state
    
    def _detect_analysis_type(self, state: DataAnalysisState) -> DataAnalysisState:
        """Detect the appropriate analysis type."""
        try:
            analysis_type = self.data_processor.detect_analysis_type(
                state["user_context"], 
                state["target_variable"]
            )
            state["analysis_type"] = analysis_type
            state["current_step"] = "analysis_type_detected"
            
        except Exception as e:
            state["error"] = f"Error detecting analysis type: {str(e)}"
        
        return state
    
    def _generate_insights(self, state: DataAnalysisState) -> DataAnalysisState:
        """Generate AI-powered insights about the data with professional ML engineering feedback."""
        try:
            if not state["data_summary"] or not state["analysis_type"]:
                state["error"] = "Missing data summary or analysis type"
                return state
            
            # Get vectorstore context for insights
            vectorstore_context = ""
            try:
                query = f"data insights analysis {state['analysis_type'].value}"
                vectorstore_docs = self.vectorstore_manager.search_data_context(query, k=3)
                vectorstore_context = "\n".join([doc.page_content for doc in vectorstore_docs])
            except Exception as e:
                logger.warning(f"Could not retrieve vectorstore context: {e}")
            
            # Generate comprehensive professional insights
            insights = []
            
            # 1. Data preprocessing recommendations
            preprocessing_insights = self._generate_preprocessing_recommendations(state)
            insights.extend(preprocessing_insights)
            
            # 2. Feature relationship insights
            feature_insights = self._generate_feature_relationship_insights(state)
            insights.extend(feature_insights)
            
            # 3. Modeling opportunity insights
            modeling_insights = self._generate_modeling_opportunity_insights(state)
            insights.extend(modeling_insights)
            
            # 4. Additional professional insights from LLM
            llm_insights = self._generate_llm_professional_insights(state, vectorstore_context)
            insights.extend(llm_insights)
            
            # Ensure we have comprehensive insights
            state["insights"] = insights[:8] if len(insights) >= 8 else insights + self._get_fallback_insights(state)
            state["current_step"] = "insights_generated"
            
        except Exception as e:
            state["error"] = f"Error generating insights: {str(e)}"
            logger.error(f"Insights generation error: {str(e)}")
        
        return state
    
    def _generate_preprocessing_recommendations(self, state: DataAnalysisState) -> List[str]:
        """Generate comprehensive data preprocessing recommendations based on detailed data analysis."""
        recommendations = []
        data_summary = state["data_summary"]
        
        # Advanced missing values analysis with patterns
        total_missing = sum(data_summary.missing_values.values())
        missing_percentage = (total_missing / (data_summary.shape[0] * data_summary.shape[1])) * 100
        
        if missing_percentage > 20:
            recommendations.append(
                f"📊 CRITICAL MISSING DATA: {missing_percentage:.1f}% missing values detected. "
                f"RECOMMENDED STEPS: 1) Analyze missing patterns (MCAR/MAR/MNAR) using missingno library, "
                f"2) Apply advanced imputation (KNNImputer, IterativeImputer, or MICE), "
                f"3) Create missing value indicators for important features, "
                f"4) Consider domain-specific imputation strategies."
            )
        elif missing_percentage > 5:
            recommendations.append(
                f"⚠️ MODERATE MISSING DATA: {missing_percentage:.1f}% missing values. "
                f"RECOMMENDED STEPS: 1) Use median/mode imputation for stable features, "
                f"2) Apply forward-fill/backward-fill for time series, "
                f"3) Consider predictive imputation using other features, "
                f"4) Validate imputation impact on target variable correlation."
            )
        elif total_missing > 0:
            recommendations.append(
                f"✅ LOW MISSING DATA: {missing_percentage:.1f}% missing values. "
                f"RECOMMENDED STEPS: 1) Verify if missing is random or systematic, "
                f"2) Use simple imputation (mean/median/mode) or listwise deletion, "
                f"3) Document missing value treatment for reproducibility."
            )
        
        # Advanced feature scaling and transformation
        if len(data_summary.numerical_columns) > 1:
            recommendations.append(
                f"🔧 FEATURE SCALING & TRANSFORMATION: {len(data_summary.numerical_columns)} numerical features. "
                f"RECOMMENDED STEPS: 1) Check distributions (use QQ plots, Shapiro-Wilk test), "
                f"2) Apply transformations for skewed data (log, box-cox, yeo-johnson), "
                f"3) Scale features: StandardScaler for normal distributions, RobustScaler for outliers, "
                f"4) MinMaxScaler for neural networks, no scaling for tree-based models."
            )
        
        # Comprehensive categorical encoding strategy
        if len(data_summary.categorical_columns) > 0:
            recommendations.append(
                f"🏷️ CATEGORICAL ENCODING STRATEGY: {len(data_summary.categorical_columns)} categorical features. "
                f"RECOMMENDED STEPS: 1) Analyze cardinality for each feature, "
                f"2) OneHotEncoder for low cardinality (<10 unique values), "
                f"3) LabelEncoder for ordinal relationships, "
                f"4) Target/mean encoding for high cardinality (>10 values), "
                f"5) Consider embeddings for very high cardinality features."
            )
        
        # Data quality and outlier detection
        recommendations.append(
            "🔍 DATA QUALITY ASSESSMENT: "
            "RECOMMENDED STEPS: 1) Detect outliers using IQR method, Z-score (>3), or Isolation Forest, "
            "2) Validate data consistency and business logic rules, "
            "3) Check for duplicate records and handle appropriately, "
            "4) Verify data types and format consistency across features."
        )
        
        # Feature engineering opportunities
        recommendations.append(
            "⚙️ FEATURE ENGINEERING OPPORTUNITIES: "
            "RECOMMENDED STEPS: 1) Create interaction features for important variable pairs, "
            "2) Generate polynomial features for potential non-linear relationships, "
            "3) Extract temporal features from datetime columns (hour, day, month, season), "
            "4) Create aggregation features (mean, std, count) for grouped data, "
            "5) Apply domain-specific transformations based on business context."
        )
        
        return recommendations[:3]  # Return top 3 comprehensive preprocessing insights
    
    def _generate_feature_relationship_insights(self, state: DataAnalysisState) -> List[str]:
        """Generate comprehensive insights about feature relationships and pattern detection."""
        insights = []
        data_summary = state["data_summary"]
        analysis_type = state["analysis_type"]
        
        # Advanced correlation analysis
        if len(data_summary.numerical_columns) > 1:
            insights.append(
                f"🔍 ADVANCED CORRELATION ANALYSIS: {len(data_summary.numerical_columns)} numerical features detected. "
                f"RECOMMENDED STEPS: 1) Compute Pearson correlation for linear relationships, "
                f"2) Calculate Spearman correlation for monotonic relationships, "
                f"3) Apply mutual information for non-linear dependencies, "
                f"4) Remove highly correlated features (>0.95) to reduce multicollinearity, "
                f"5) Use VIF (Variance Inflation Factor) to detect multicollinearity."
            )
        
        # Feature interaction analysis
        if len(data_summary.numerical_columns) >= 2 and len(data_summary.categorical_columns) >= 1:
            insights.append(
                "🧬 FEATURE INTERACTION ANALYSIS: Mixed feature types present. "
                "RECOMMENDED STEPS: 1) Create numerical×categorical interactions using group statistics, "
                "2) Generate polynomial features for numerical pairs (degree 2-3), "
                "3) Apply statistical tests (ANOVA for num vs cat, chi-square for cat vs cat), "
                "4) Use feature selection methods (RFE, SelectKBest) to identify important interactions, "
                "5) Consider ensemble feature importance to guide interaction creation."
            )
        
        # Pattern detection strategies based on analysis type
        pattern_strategy = self._get_comprehensive_pattern_detection_strategy(analysis_type, data_summary)
        insights.append(pattern_strategy)
        
        # Dimensionality and feature selection insights
        feature_ratio = data_summary.shape[1] / data_summary.shape[0]
        if feature_ratio > 0.1:
            insights.append(
                f"📐 DIMENSIONALITY REDUCTION: High feature-to-sample ratio ({feature_ratio:.3f}). "
                f"RECOMMENDED STEPS: 1) Apply PCA for linear dimensionality reduction, "
                f"2) Use t-SNE or UMAP for non-linear visualization, "
                f"3) Implement feature selection (RFE, LASSO, SelectFromModel), "
                f"4) Apply regularization (L1/L2) to prevent overfitting, "
                f"5) Consider domain knowledge for manual feature selection."
            )
        elif len(data_summary.numerical_columns) + len(data_summary.categorical_columns) > 50:
            insights.append(
                f"🎯 FEATURE SELECTION: {data_summary.shape[1]} features detected. "
                f"RECOMMENDED STEPS: 1) Use univariate feature selection (f_classif, f_regression), "
                f"2) Apply recursive feature elimination with cross-validation, "
                f"3) Implement LASSO regularization for automatic feature selection, "
                f"4) Use tree-based feature importance (Random Forest, XGBoost), "
                f"5) Apply correlation-based feature selection to remove redundant features."
            )
        
        # Data distribution and transformation insights
        if len(data_summary.numerical_columns) > 0:
            insights.append(
                "📊 DISTRIBUTION ANALYSIS & TRANSFORMATIONS: "
                "RECOMMENDED STEPS: 1) Test normality using Shapiro-Wilk or Kolmogorov-Smirnov tests, "
                "2) Apply log transformation for right-skewed distributions, "
                "3) Use Box-Cox or Yeo-Johnson transformations for non-normal data, "
                "4) Detect and handle outliers using IQR, Z-score, or Isolation Forest, "
                "5) Create binned versions of continuous variables for non-linear models."
            )
        
        return insights[:3]  # Return top 3 comprehensive feature insights
    
    def _generate_modeling_opportunity_insights(self, state: DataAnalysisState) -> List[str]:
        """Generate modeling opportunity insights based on analysis type and data characteristics."""
        insights = []
        analysis_type = state["analysis_type"]
        data_summary = state["data_summary"]
        data_size = data_summary.shape[0]
        
        # Analysis-type specific modeling strategies
        if analysis_type == DataAnalysisType.REGRESSION:
            insights.append(
                "🎯 REGRESSION MODELING STRATEGY: "
                "RECOMMENDED STEPS: 1) Start with Linear Regression for baseline and interpretability, "
                "2) Progress to ensemble methods (Random Forest, XGBoost, LightGBM), "
                "3) Consider polynomial features for non-linear relationships, "
                "4) Apply regularization (Ridge, Lasso, ElasticNet) for feature selection, "
                "5) Use cross-validation with R², MAE, and RMSE for evaluation."
            )
        elif analysis_type == DataAnalysisType.CLASSIFICATION:
            insights.append(
                "🎯 CLASSIFICATION MODELING STRATEGY: "
                "RECOMMENDED STEPS: 1) Begin with Logistic Regression baseline, "
                "2) Progress to ensemble methods (RF, XGBoost, LightGBM), "
                "3) Check class imbalance and apply SMOTE/ADASYN if needed, "
                "4) Consider neural networks for complex patterns, "
                "5) Evaluate with accuracy, F1-score, ROC-AUC, and confusion matrix."
            )
        elif analysis_type == DataAnalysisType.CLUSTERING:
            insights.append(
                "🎯 CLUSTERING MODELING STRATEGY: "
                "RECOMMENDED STEPS: 1) Start with K-Means for spherical clusters, "
                "2) Use DBSCAN for arbitrary shapes and noise handling, "
                "3) Apply hierarchical clustering for dendrograms, "
                "4) Determine optimal clusters using elbow method and silhouette analysis, "
                "5) Validate clusters with domain knowledge and business logic."
            )
        elif analysis_type == DataAnalysisType.TIME_SERIES:
            insights.append(
                "🎯 TIME SERIES MODELING STRATEGY: "
                "RECOMMENDED STEPS: 1) Check stationarity using ADF test, "
                "2) Start with ARIMA/SARIMA for linear patterns, "
                "3) Use Prophet for seasonality and holiday effects, "
                "4) Consider LSTM/GRU for complex non-linear patterns, "
                "5) Implement walk-forward validation for temporal data."
            )
        else:  # EXPLORATORY
            insights.append(
                "🎯 EXPLORATORY MODELING OPPORTUNITIES: "
                "RECOMMENDED STEPS: 1) Apply unsupervised learning for pattern discovery, "
                "2) Use association rule mining for relationship identification, "
                "3) Consider dimensionality reduction for visualization, "
                "4) Explore supervised learning if target variable emerges, "
                "5) Apply anomaly detection for outlier identification."
            )
        
        # Data size and complexity considerations
        if data_size < 1000:
            insights.append(
                f"📊 SMALL DATASET STRATEGY: With {data_size} samples, "
                f"RECOMMENDED STEPS: 1) Use simple models to avoid overfitting, "
                f"2) Apply cross-validation with more folds (k=10), "
                f"3) Consider data augmentation if applicable, "
                f"4) Use regularization techniques heavily, "
                f"5) Focus on feature engineering over complex models."
            )
        elif data_size > 10000:
            insights.append(
                f"📊 LARGE DATASET STRATEGY: With {data_size} samples, "
                f"RECOMMENDED STEPS: 1) Consider deep learning models, "
                f"2) Use train/validation/test splits (60/20/20), "
                f"3) Apply ensemble methods for robust predictions, "
                f"4) Consider distributed computing for very large datasets, "
                f"5) Implement early stopping and learning rate scheduling."
            )
        else:
            insights.append(
                f"📊 MEDIUM DATASET STRATEGY: With {data_size} samples, "
                f"RECOMMENDED STEPS: 1) Balance between simple and complex models, "
                f"2) Use stratified k-fold cross-validation, "
                f"3) Compare multiple algorithms systematically, "
                f"4) Apply feature selection and engineering, "
                f"5) Use ensemble methods for improved performance."
            )
        
        # Business value and deployment insights
        insights.append(
            "💼 MODEL DEPLOYMENT & MONITORING STRATEGY: "
            "RECOMMENDED STEPS: 1) Plan MLOps pipeline with model versioning, "
            "2) Implement A/B testing framework for model comparison, "
            "3) Set up performance monitoring and drift detection, "
            "4) Consider model interpretability (SHAP, LIME) for stakeholder communication, "
            "5) Plan for regulatory compliance and ethical AI considerations."
        )
        
        return insights[:3]  # Return top 3 comprehensive modeling insights
    
    def _get_comprehensive_pattern_detection_strategy(self, analysis_type: DataAnalysisType, data_summary: DataSummary) -> str:
        """Get comprehensive pattern detection strategy based on analysis type and data characteristics."""
        if analysis_type == DataAnalysisType.TIME_SERIES:
            return (
                "📈 TEMPORAL PATTERN ANALYSIS: "
                "RECOMMENDED STEPS: 1) Decompose series (trend, seasonal, residual components), "
                "2) Test stationarity using ADF and KPSS tests, "
                "3) Analyze autocorrelation (ACF) and partial autocorrelation (PACF), "
                "4) Detect seasonality patterns and cyclical behaviors, "
                "5) Apply spectral analysis for frequency domain insights."
            )
        elif len(data_summary.categorical_columns) > 2:
            return (
                "🕸️ CATEGORICAL PATTERN MINING: "
                "RECOMMENDED STEPS: 1) Compute chi-square tests for independence, "
                "2) Calculate Cramér's V for association strength, "
                "3) Apply association rules mining (Apriori algorithm), "
                "4) Use contingency tables for cross-tabulation analysis, "
                "5) Identify rare category combinations and potential data quality issues."
            )
        elif len(data_summary.numerical_columns) > 3:
            return (
                "📊 MULTIVARIATE PATTERN DISCOVERY: "
                "RECOMMENDED STEPS: 1) Apply clustering analysis to identify data segments, "
                "2) Use principal component analysis for variance explanation, "
                "3) Perform anomaly detection using Isolation Forest or Local Outlier Factor, "
                "4) Create scatter plot matrices for pairwise relationships, "
                "5) Apply density-based pattern recognition techniques."
            )
        else:
            return (
                "📋 BASIC DISTRIBUTION ANALYSIS: "
                "RECOMMENDED STEPS: 1) Test normality using Shapiro-Wilk and Anderson-Darling tests, "
                "2) Identify outliers using IQR method and Z-scores, "
                "3) Analyze skewness and kurtosis for distribution shape, "
                "4) Create Q-Q plots for distribution comparison, "
                "5) Apply appropriate transformations for data normalization."
            )
    
    def _generate_llm_professional_insights(self, state: DataAnalysisState, vectorstore_context: str) -> List[str]:
        """Generate additional professional insights using LLM with enhanced prompting."""
        try:
            # Check if required data is available
            if not state.get("data_summary") or not state.get("analysis_type"):
                logger.warning("Missing data_summary or analysis_type for LLM insights")
                return self._get_fallback_insights_safe(state)
            
            prompt = ChatPromptTemplate.from_template(
                """You are a Senior Data Scientist and ML Engineer providing professional analysis insights. Based on the dataset characteristics, provide 2 strategic professional insights as a JSON list.

                DATASET PROFILE:
                - Shape: {shape[0]} rows, {shape[1]} columns  
                - Numerical features: {numerical_count}, Categorical features: {categorical_count}
                - Data completeness: {completeness:.1f}%
                - Analysis type: {analysis_type}
                - Feature-to-sample ratio: {feature_ratio:.3f}

                VECTORSTORE CONTEXT (related data patterns): 
                {vectorstore_context}

                ANALYSIS CONTEXT:
                {user_context}

                As a senior professional, focus on:
                1. **Advanced Analytical Techniques**: Specific methods for this data profile (statistical tests, advanced modeling approaches, feature engineering strategies)
                2. **Risk Assessment & Validation**: Data quality concerns, model validation strategies, potential biases, production deployment considerations
                3. **Business Value & ROI**: Actionable insights that drive business decisions, success metrics, implementation priorities
                4. **Technical Implementation**: Scalability considerations, tool recommendations, infrastructure requirements

                Provide insights that demonstrate deep expertise and practical experience. Each insight should be actionable and specific to this dataset's characteristics.

                Return ONLY a JSON array of exactly 2 professional insights.
                Format: ["🧠 ADVANCED TECHNIQUE: Specific technical insight with methodology", "🔬 STRATEGIC VALIDATION: Risk assessment or business value insight"]
                """
            )
            
            missing_percentage = (sum(state["data_summary"].missing_values.values()) / 
                                (state["data_summary"].shape[0] * state["data_summary"].shape[1])) * 100
            completeness = 100 - missing_percentage
            feature_ratio = state["data_summary"].shape[1] / state["data_summary"].shape[0]
            
            # Safely get analysis type value
            analysis_type_value = state["analysis_type"].value if state["analysis_type"] else "exploratory"
            
            response = self.llm.invoke(prompt.format(
                shape=state["data_summary"].shape,
                numerical_count=len(state["data_summary"].numerical_columns),
                categorical_count=len(state["data_summary"].categorical_columns),
                completeness=completeness,
                analysis_type=analysis_type_value,
                feature_ratio=feature_ratio,
                vectorstore_context=vectorstore_context[:1000] if vectorstore_context else "No prior context available",
                user_context=state.get("user_context", "No specific context provided")[:500]
            ))
            
            try:
                insights_data = json.loads(response.content.strip())
                if isinstance(insights_data, list) and len(insights_data) >= 2:
                    return [str(insight) for insight in insights_data[:2]]
            except json.JSONDecodeError:
                pass
            
            # Enhanced fallback professional insights based on analysis type
            return self._get_enhanced_fallback_insights(state, completeness)
            
        except Exception as e:
            logger.warning(f"LLM professional insights failed: {e}")
            return self._get_safe_fallback_insights(state)

    def _get_enhanced_fallback_insights(self, state: DataAnalysisState, completeness: float) -> List[str]:
        """Get enhanced fallback insights based on analysis type."""
        analysis_type = state.get("analysis_type")
        
        if analysis_type == DataAnalysisType.REGRESSION:
            return [
                f"🧠 ADVANCED TECHNIQUE: For regression with {state['data_summary'].shape[1]} features, implement Elastic Net regularization with alpha optimization to balance Ridge/Lasso penalties, and consider polynomial feature engineering with interaction terms for non-linear relationships.",
                f"🔬 STRATEGIC VALIDATION: With {completeness:.1f}% data completeness, implement nested cross-validation for robust model selection and use residual analysis to validate assumptions. Monitor prediction intervals for uncertainty quantification in production."
            ]
        elif analysis_type == DataAnalysisType.CLASSIFICATION:
            return [
                "🧠 ADVANCED TECHNIQUE: Implement stratified ensemble methods with probability calibration (Platt scaling) for reliable confidence scores, and use SHAP values for feature importance interpretation to meet explainability requirements.",
                "🔬 STRATEGIC VALIDATION: Apply temporal validation if data has time dependencies, and implement fairness metrics (demographic parity, equalized odds) to ensure model equity across different population groups."
            ]
        else:
            analysis_type_str = analysis_type.value if analysis_type else "data"
            return [
                f"🧠 ADVANCED TECHNIQUE: Apply multi-level feature engineering including domain-specific transformations, temporal aggregations, and interaction features to capture complex patterns in your {analysis_type_str} analysis.",
                "🔬 STRATEGIC VALIDATION: Implement comprehensive model monitoring with drift detection on both features and predictions, establishing clear KPIs and automated retraining triggers for production deployment."
            ]

    def _get_safe_fallback_insights(self, state: DataAnalysisState) -> List[str]:
        """Get safe fallback insights when there are errors."""
        analysis_type = state.get("analysis_type")
        analysis_type_str = analysis_type.value if analysis_type else "general"
        
        return [
            f"🧠 ADVANCED TECHNIQUE: Consider advanced ensemble methods and sophisticated feature engineering for robust {analysis_type_str} modeling with proper hyperparameter optimization.",
            "🔬 STRATEGIC VALIDATION: Implement comprehensive validation framework with business impact metrics, A/B testing capability, and continuous monitoring for production-ready ML solutions."
        ]

    def _get_fallback_insights_safe(self, state: DataAnalysisState) -> List[str]:
        """Generate safe fallback insights when required data is missing."""
        data_summary = state.get("data_summary")
        analysis_type = state.get("analysis_type")
        
        insights = []
        
        if data_summary:
            insights.append(f"Dataset contains {data_summary.shape[0]} rows and {data_summary.shape[1]} columns")
            insights.append(f"Dataset has {len(data_summary.numerical_columns)} numerical and {len(data_summary.categorical_columns)} categorical features")
        else:
            insights.append("Dataset structure analysis in progress")
        
        if analysis_type:
            insights.append(f"Analysis type identified as {analysis_type.value}")
        else:
            insights.append("Analysis type detection in progress")
            
        insights.append("Data preprocessing and analysis can proceed with the identified structure")
        
        return insights
    
    def _get_fallback_insights(self, state: DataAnalysisState) -> List[str]:
        """Generate fallback insights when LLM parsing fails."""
        data_summary = state.get("data_summary")
        analysis_type = state.get("analysis_type")
        
        insights = []
        
        if data_summary:
            insights.append(f"Dataset contains {data_summary.shape[0]} rows and {data_summary.shape[1]} columns")
        
        if analysis_type:
            insights.append(f"Analysis type identified as {analysis_type.value}")
        
        if data_summary:
            insights.append(f"Dataset has {len(data_summary.numerical_columns)} numerical and {len(data_summary.categorical_columns)} categorical features")
        
        insights.append("Data preprocessing and analysis can proceed with the identified structure")
        
        return insights
    
    def _create_visualizations(self, state: DataAnalysisState) -> DataAnalysisState:
        """Create comprehensive visualizations using DataVisualizer."""
        try:
            if self.data_processor.data is None:
                state["error"] = "No data available for visualization"
                return state

            # Use the working DataVisualizer class
            try:
                visualizer = DataVisualizer(self.data_processor.data)
                plots = visualizer.create_overview_plots()
                
                # Add target variable analysis if specified
                if state.get("target_variable"):
                    target_plots = visualizer.create_target_analysis(state["target_variable"])
                    plots.extend(target_plots)
                
                # Ensure we have plots as base64 strings
                if plots and all(isinstance(plot, str) for plot in plots):
                    state["plots"] = plots
                    state["current_step"] = "visualizations_created"
                    
                    # Add success message
                    state["messages"].append(SystemMessage(
                        content=f"Successfully created {len(plots)} visualizations. "
                               f"Charts include data overview, distributions, and correlations."
                    ))
                else:
                    # Fallback to basic summary if visualizer fails
                    state["plots"] = []
                    state["current_step"] = "visualizations_created"
                    logger.warning("DataVisualizer did not return proper base64 plots")
                
            except Exception as viz_error:
                logger.error(f"DataVisualizer failed: {str(viz_error)}")
                # Create minimal fallback plots
                state["plots"] = []
                state["current_step"] = "visualizations_created"
                state["messages"].append(SystemMessage(
                    content="Visualization creation encountered issues. Analysis continues without plots."
                ))
            
        except Exception as e:
            state["error"] = f"Error creating visualizations: {str(e)}"
            logger.error(f"Visualization error: {str(e)}")
        
        return state
    
    def _get_visualization_prompts(self, state: DataAnalysisState) -> List[str]:
        """Get appropriate visualization prompts based on analysis type."""
        data_info = f"Dataset shape: {state['data_summary'].shape}, numerical columns: {len(state['data_summary'].numerical_columns)}, categorical columns: {len(state['data_summary'].categorical_columns)}"
        
        base_prompts = [
            f"Create a comprehensive overview plot showing the distribution of numerical columns. {data_info}",
            f"Generate correlation heatmap for numerical features. {data_info}"
        ]
        
        if state["analysis_type"] == DataAnalysisType.REGRESSION and state.get("target_variable"):
            base_prompts.append(f"Create scatter plots showing relationship between {state['target_variable']} and other numerical features")
            
        elif state["analysis_type"] == DataAnalysisType.CLASSIFICATION and state.get("target_variable"):
            base_prompts.append(f"Create distribution plots and box plots for {state['target_variable']} by categories")
            
        elif state["analysis_type"] == DataAnalysisType.TIME_SERIES:
            base_prompts.append("Create time series plots showing trends over time")
            
        elif state["analysis_type"] == DataAnalysisType.CLUSTERING:
            base_prompts.append("Create scatter plots and distribution plots suitable for clustering analysis")
        
        # Add missing data visualization if there are missing values
        if sum(state['data_summary'].missing_values.values()) > 0:
            base_prompts.append("Create a missing data heatmap to visualize data completeness patterns")
            
        return base_prompts
    
    def _generate_plot_code(self, prompt: str, state: DataAnalysisState) -> str:
        """Generate executable Python plotting code using the LLM."""
        try:
            code_prompt = ChatPromptTemplate.from_template(
                """You are a data visualization expert. Generate clean, executable Python code to create the requested plot.

DATA CONTEXT:
- Dataset shape: {shape}
- Columns: {columns}
- Analysis type: {analysis_type}
- Target variable: {target_variable}

VISUALIZATION REQUEST:
{prompt}

REQUIREMENTS:
1. Use matplotlib/seaborn for plotting
2. Include proper titles, labels, and formatting
3. Handle missing data gracefully with dropna() or appropriate methods
4. Make plots informative and publication-ready
5. Do NOT include plt.show() - this will be added separately
6. Do NOT include figure creation - this will be handled separately
7. Return ONLY the core plotting code, no explanations
8. Use 'df' as the dataframe variable name

IMPORTANT: The dataframe is already loaded as 'df'. Do not reload data.
Focus on the core plotting logic only.

Generate the Python plotting code:
"""
            )
            
            response = self.llm.invoke(code_prompt.format(
                shape=state["data_summary"].shape,
                columns=", ".join(state["data_summary"].columns[:10]),  # Limit for prompt size
                analysis_type=state["analysis_type"].value,
                target_variable=state.get("target_variable", "None"),
                prompt=prompt
            ))
            
            # Extract code from response
            code = response.content.strip()
            
            # Clean up the code - remove markdown formatting if present
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].strip()
            
            # Remove any plt.figure() or plt.show() calls as these are handled separately
            lines = code.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if not line.startswith('plt.figure(') and not line.startswith('plt.show()') and line:
                    cleaned_lines.append(line)
            
            code = '\n'.join(cleaned_lines)
            
            # Ensure basic error handling for common issues
            if 'hist(' in code or 'plot(' in code:
                code = f"# Handle missing values\ndf_clean = df.dropna()\n\n{code}"
                
            return code
            
        except Exception as e:
            logger.error(f"Error generating plot code: {e}")
            return f"# Error generating plot: {e}\nprint('Plot generation failed for: {prompt}')"
    
    def _get_ml_recommendations(self, state: DataAnalysisState) -> DataAnalysisState:
        """Get ML model and metrics recommendations."""
        try:
            if not state["data_summary"] or not state["analysis_type"]:
                state["error"] = "Missing data summary or analysis type for recommendations"
                return state
            
            recommender = MLModelRecommender(state["data_summary"], state["analysis_type"])
            models = recommender.recommend_models()
            metrics = recommender.recommend_metrics()
            
            state["recommendations"] = {
                "models": models,
                "metrics": metrics
            }
            state["current_step"] = "recommendations_generated"
            
        except Exception as e:
            state["error"] = f"Error getting recommendations: {str(e)}"
        
        return state
    
    def _finalize_analysis(self, state: DataAnalysisState) -> DataAnalysisState:
        """Finalize the analysis workflow."""
        state["current_step"] = "completed"
        state["messages"].append(SystemMessage(content="Data analysis workflow completed successfully"))
        return state
    
    def _generate_analysis_summary(self, state: DataAnalysisState) -> str:
        """Generate a comprehensive analysis summary."""
        if not state["data_summary"] or not state["analysis_type"]:
            return "Analysis incomplete due to errors."
        
        summary = f"""
📊 **Data Analysis Summary**

**Dataset Overview:**
- **Size:** {state['data_summary'].shape[0]:,} rows × {state['data_summary'].shape[1]} columns
- **Analysis Type:** {state['analysis_type'].value.title()}
- **Data Quality:** {((state['data_summary'].shape[0] * state['data_summary'].shape[1] - sum(state['data_summary'].missing_values.values())) / (state['data_summary'].shape[0] * state['data_summary'].shape[1]) * 100):.1f}% complete

**Feature Types:**
- **Numerical:** {len(state['data_summary'].numerical_columns)} columns
- **Categorical:** {len(state['data_summary'].categorical_columns)} columns
- **DateTime:** {len(state['data_summary'].datetime_columns)} columns

**Vectorstore Status:** ✅ Data successfully added to dedicated CSV/Excel vectorstore (separate from PDF documents)

**Key Insights:**
{chr(10).join([f"• {insight}" for insight in state.get('insights', [])])}

The dataset has been processed and is ready for {state['analysis_type'].value} analysis.
"""
        return summary.strip()
    
    def _generate_next_steps(self, state: DataAnalysisState) -> List[str]:
        """Generate actionable next steps based on analysis with professional ML guidance."""
        if not state["analysis_type"]:
            return ["Complete the data analysis first"]
        
        steps = [
            "📊 Review the generated visualizations and comprehensive professional insights",
            "🔧 Implement data preprocessing recommendations from the detailed analysis"
        ]
        
        if state["analysis_type"] == DataAnalysisType.REGRESSION:
            steps.extend([
                "🔍 Apply advanced correlation analysis (Pearson, Spearman, mutual information)",
                "📈 Implement regression models with hyperparameter tuning and cross-validation",
                "🎯 Engineer features based on identified relationships and patterns",
                "📊 Validate models using R², MAE, RMSE metrics with proper train/test splits",
                "🚀 Consider ensemble methods and regularization for production deployment"
            ])
        elif state["analysis_type"] == DataAnalysisType.CLASSIFICATION:
            steps.extend([
                "⚖️ Analyze class distribution and implement SMOTE/ADASYN for imbalance",
                "🏷️ Apply comprehensive categorical encoding (OneHot, Target, Label encoding)",
                "🤖 Build classification pipeline with ensemble methods and cross-validation",
                "📊 Evaluate using accuracy, F1-score, ROC-AUC, and confusion matrix analysis",
                "🔬 Implement feature selection and model interpretability (SHAP, LIME)"
            ])
        elif state["analysis_type"] == DataAnalysisType.CLUSTERING:
            steps.extend([
                "📏 Apply feature scaling and dimensionality reduction techniques",
                "🎯 Determine optimal clusters using elbow method, silhouette, and Gap statistic",
                "🕸️ Implement multiple clustering algorithms (K-Means, DBSCAN, Hierarchical)",
                "📊 Validate clusters using internal metrics and domain knowledge",
                "🔍 Analyze cluster characteristics and create business interpretations"
            ])
        elif state["analysis_type"] == DataAnalysisType.TIME_SERIES:
            steps.extend([
                "📈 Perform comprehensive temporal analysis (stationarity, seasonality, trends)",
                "🔮 Implement forecasting models (ARIMA, Prophet, LSTM) with proper validation",
                "📊 Apply walk-forward validation and evaluate with MAPE, MAE, RMSE",
                "⏰ Monitor model performance with drift detection and retraining strategies",
                "📋 Create forecasting intervals and uncertainty quantification"
            ])
        else:  # EXPLORATORY
            steps.extend([
                "🔍 Conduct multivariate pattern analysis and feature relationship mapping",
                "💡 Apply unsupervised learning techniques for pattern discovery",
                "🎯 Identify potential supervised learning opportunities and target variables",
                "📊 Implement association rule mining and anomaly detection",
                "🔬 Create comprehensive data profiling and quality assessment reports"
            ])
        
        # Add professional deployment and monitoring steps
        steps.extend([
            "🚀 Design MLOps pipeline with model versioning, monitoring, and A/B testing",
            "📋 Generate executive summary and technical documentation for stakeholders",
            "� Implement model governance, ethics review, and regulatory compliance checks"
        ])
        
        return steps[:8]  # Return top 8 comprehensive and actionable steps

class EnhancedLangGraphDataChatAgent:
    """Enhanced LangGraph-based agent for chatting about data using vectorstore RAG."""
    
    def __init__(self, llm, embedder, data_processor: DataProcessor):
        self.llm = llm
        self.embedder = embedder
        self.data_processor = data_processor
        self.vectorstore_manager = DataVectorStoreManager(embedder)
        self.vectorstore_manager.load_existing_vectorstore()
        self.chat_graph = self._build_chat_graph()
        self.pandas_agent = None
        self._initialize_pandas_agent()
    
    def _initialize_pandas_agent(self):
        """Initialize pandas agent for direct data queries."""
        if self.data_processor.data is not None:
            try:
                self.pandas_agent = create_pandas_dataframe_agent(
                    self.llm,
                    self.data_processor.data,
                    agent_type=AgentType.OPENAI_FUNCTIONS,
                    verbose=False,
                    allow_dangerous_code=True
                )
            except Exception as e:
                logger.warning(f"Could not create pandas agent: {str(e)}")
    
    def answer_question(self, question: str) -> str:
        """Answer questions about the data using enhanced RAG with vectorstore."""
        try:
            initial_state = DataChatState(
                messages=[HumanMessage(content=question)],
                query=question,
                response="",
                vectorstore_context=[],
                data_context=None,
                pandas_agent_response=None
            )
            
            final_state = self.chat_graph.invoke(initial_state)
            return final_state.get("response", "Sorry, I couldn't process your question.")
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return f"I encountered an error while processing your question: {str(e)}"
    
    def _build_chat_graph(self) -> StateGraph:
        """Build the enhanced chat workflow graph."""
        workflow = StateGraph(DataChatState)
        
        # Add nodes
        workflow.add_node("start", self._start_chat)
        workflow.add_node("search_vectorstore", self._search_vectorstore)
        workflow.add_node("get_data_context", self._get_data_context)
        workflow.add_node("query_pandas_agent", self._query_pandas_agent)
        workflow.add_node("generate_response", self._generate_response)
        
        # Add edges
        workflow.set_entry_point("start")
        workflow.add_edge("start", "search_vectorstore")
        workflow.add_edge("search_vectorstore", "get_data_context")
        workflow.add_edge("get_data_context", "query_pandas_agent")
        workflow.add_edge("query_pandas_agent", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()
    
    def _start_chat(self, state: DataChatState) -> DataChatState:
        """Start the chat workflow."""
        return state
    
    def _search_vectorstore(self, state: DataChatState) -> DataChatState:
        """Search the CSV/Excel vectorstore for relevant context."""
        try:
            docs = self.vectorstore_manager.search_data_context(state["query"], k=5)
            state["vectorstore_context"] = [doc.page_content for doc in docs]
        except Exception as e:
            logger.warning(f"Could not search vectorstore: {str(e)}")
            state["vectorstore_context"] = []
        
        return state
    
    def _get_data_context(self, state: DataChatState) -> DataChatState:
        """Get direct data context from the loaded dataset."""
        if self.data_processor.data is None:
            state["data_context"] = "No dataset is currently loaded."
            return state
        
        try:
            data = self.data_processor.data
            context = f"""
Current Dataset Information:
- Shape: {data.shape[0]} rows, {data.shape[1]} columns
- Columns: {', '.join(data.columns.tolist())}
- Data types: {data.dtypes.to_dict()}
- Missing values: {data.isnull().sum().to_dict()}
- Sample data:
{data.head(3).to_string()}
"""
            state["data_context"] = context
        except Exception as e:
            state["data_context"] = f"Error getting data context: {str(e)}"
        
        return state
    
    def _query_pandas_agent(self, state: DataChatState) -> DataChatState:
        """Query the pandas agent for specific data operations."""
        if self.pandas_agent is None:
            state["pandas_agent_response"] = "Pandas agent not available."
            return state
        
        try:
            # Check if query requires data calculation/analysis
            calculation_keywords = ['calculate', 'compute', 'count', 'sum', 'mean', 'average', 'max', 'min', 'correlation', 'statistics']
            
            if any(keyword in state["query"].lower() for keyword in calculation_keywords):
                response = self.pandas_agent.run(state["query"])
                state["pandas_agent_response"] = response
            else:
                state["pandas_agent_response"] = "No calculation needed."
                
        except Exception as e:
            logger.warning(f"Pandas agent error: {str(e)}")
            state["pandas_agent_response"] = f"Could not execute pandas operation: {str(e)}"
        
        return state
    
    def _generate_response(self, state: DataChatState) -> DataChatState:
        """Generate the final response using all available context with professional ML guidance."""
        
        # Check if the query relates to modeling, preprocessing, or feature engineering
        professional_keywords = [
            'model', 'modeling', 'machine learning', 'ml', 'algorithm', 'predict', 'classification', 'regression',
            'clustering', 'preprocessing', 'feature', 'engineering', 'correlation', 'pattern', 'relationship',
            'analysis', 'insight', 'recommend', 'suggest', 'approach', 'strategy', 'validation', 'performance'
        ]
        
        needs_professional_guidance = any(keyword in state["query"].lower() for keyword in professional_keywords)
        
        if needs_professional_guidance:
            prompt = ChatPromptTemplate.from_template(
                """You are a senior data scientist and machine learning engineer. Answer the user's question with professional expertise and actionable recommendations.

VECTORSTORE CONTEXT (from CSV/Excel data analysis):
{vectorstore_context}

CURRENT DATA CONTEXT:
{data_context}

PANDAS AGENT RESULT:
{pandas_agent_response}

USER QUESTION: {query}

Provide a comprehensive answer that includes:
1. Direct answer to the question
2. Professional data preprocessing recommendations if relevant
3. Feature relationship and pattern analysis suggestions
4. Modeling opportunities and best practices
5. Validation and deployment considerations

Format your response with clear sections and actionable insights. Use emojis to highlight key points.
Maximum 8-10 sentences with practical guidance.
"""
            )
        else:
            prompt = ChatPromptTemplate.from_template(
                """You are a data analyst assistant. Answer the user's question about their dataset using the provided context.

VECTORSTORE CONTEXT (from CSV/Excel data analysis):
{vectorstore_context}

CURRENT DATA CONTEXT:
{data_context}

PANDAS AGENT RESULT:
{pandas_agent_response}

USER QUESTION: {query}

Provide a clear, specific answer based on the available data and context. If calculations were performed, include the results. 
Keep responses concise but informative (maximum 4-5 sentences).
"""
            )
        
        try:
            vectorstore_text = "\n".join(state["vectorstore_context"][:3])  # Limit context
            
            response = self.llm.invoke(prompt.format(
                vectorstore_context=vectorstore_text,
                data_context=state["data_context"] or "No current data context available.",
                pandas_agent_response=state["pandas_agent_response"] or "No calculation performed.",
                query=state["query"]
            ))
            
            state["response"] = response.content
            
        except Exception as e:
            state["response"] = f"Error generating response: {str(e)}"
        
        return state
