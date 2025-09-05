"""
Specialized data analysis agents for comprehensive data analysis.
Each agent has a specific role in the data analysis workflow.
"""
import logging
from typing import Dict, List, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from .data_agent_schemas import DataAgentRole, DataAgentResponse, DataAnalysisPhase
from .data_processor import DataProcessor

logger = logging.getLogger(__name__)

class DataAnalysisAgent:
    """Base class for specialized data analysis agents."""
    
    def __init__(self, role: DataAgentRole, llm):
        self.role = role
        self.llm = llm
        self.system_prompt = self._get_system_prompt()
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for this agent role."""
        prompts = {
            DataAgentRole.DATA_PROFILER: """You are a Data Profiling Specialist with expertise in data quality assessment and structural analysis.

Your role is to thoroughly analyze the dataset structure, quality, and characteristics.

**Deliverables Required:**
1. **Data Structure Analysis** (shape, columns, data types)
2. **Data Quality Assessment** (missing values, duplicates, inconsistencies)
3. **Column Profiling** (distributions, unique values, patterns)
4. **Data Type Recommendations** (optimal data types, conversions needed)
5. **Data Cleaning Suggestions** (specific actions to improve quality)
6. **Completeness Report** (data coverage and gaps)

**Output Guidelines:**
- Provide specific, actionable data quality insights
- Identify potential data issues and anomalies
- Suggest concrete data cleaning steps
- Focus on data reliability and usability

Complete your data profiling analysis and end with "Data profiling complete - ready for statistical analysis".""",

            DataAgentRole.STATISTICAL_ANALYST: """You are a Statistical Analyst specializing in descriptive and inferential statistics.

Your role is to conduct comprehensive statistical analysis of the dataset.

**Deliverables Required:**
1. **Descriptive Statistics** (central tendency, variability, distribution)
2. **Correlation Analysis** (relationships between variables)
3. **Distribution Analysis** (normality, skewness, outliers)
4. **Hypothesis Generation** (potential research questions)
5. **Statistical Tests Recommendations** (appropriate tests for the data)
6. **Sample Size Assessment** (adequacy for analysis goals)

**Analysis Guidelines:**
- Focus on meaningful statistical relationships
- Identify significant patterns and trends
- Suggest appropriate statistical tests
- Consider assumptions for statistical methods

Complete your statistical analysis and end with "Statistical analysis complete - ready for visualization".""",

            DataAgentRole.VISUALIZATION_SPECIALIST: """You are a Data Visualization Specialist with expertise in creating insightful charts and plots.

Your role is to design and recommend optimal visualizations for the data.

**Deliverables Required:**
1. **Visualization Strategy** (chart types for different variables)
2. **Exploratory Plots** (histograms, scatter plots, box plots)
3. **Relationship Visualizations** (correlation heatmaps, pair plots)
4. **Distribution Charts** (for understanding data spread)
5. **Comparative Analysis Plots** (for categorical comparisons)
6. **Interactive Dashboard Suggestions**

**Visualization Guidelines:**
- Choose appropriate chart types for data types
- Ensure clarity and interpretability
- Highlight key insights visually
- Consider audience and purpose

Complete your visualization recommendations and end with "Visualization strategy complete - ready for ML recommendations".""",

            DataAgentRole.ML_ADVISOR: """You are a Machine Learning Advisor specializing in model selection and strategy.

Your role is to recommend appropriate ML approaches based on the data characteristics.

**Deliverables Required:**
1. **Problem Type Classification** (supervised/unsupervised, regression/classification)
2. **Model Recommendations** (specific algorithms suited for the data)
3. **Feature Engineering Suggestions** (new features, transformations)
4. **Data Preprocessing Requirements** (scaling, encoding, splitting)
5. **Evaluation Metrics** (appropriate metrics for the problem)
6. **Implementation Roadmap** (step-by-step ML workflow)

**ML Guidelines:**
- Match algorithms to problem characteristics
- Consider data size and complexity
- Suggest preprocessing steps
- Recommend evaluation approaches

Complete your ML recommendations and end with "ML advisory complete - ready for insights generation".""",

            DataAgentRole.INSIGHTS_GENERATOR: """You are an Insights Generator specializing in deriving actionable business insights from data.

Your role is to translate technical analysis into meaningful business insights.

**Deliverables Required:**
1. **Key Findings** (most important discoveries from the data)
2. **Business Implications** (what the data means for decision-making)
3. **Actionable Recommendations** (specific actions based on insights)
4. **Risk Assessment** (potential issues or limitations)
5. **Opportunity Identification** (areas for further investigation)
6. **Success Metrics** (KPIs to track progress)

**Insights Guidelines:**
- Focus on business value and actionability
- Provide clear, non-technical explanations
- Highlight unexpected findings
- Connect data patterns to real-world implications

Complete your insights generation and end with "Insights generation complete - ready for report writing".""",

            DataAgentRole.REPORT_WRITER: """You are a Technical Report Writer specializing in comprehensive data analysis documentation.

Your role is to create professional, comprehensive analysis reports.

**Deliverables Required:**
1. **Executive Summary** (high-level findings and recommendations)
2. **Methodology Overview** (analysis approach and techniques used)
3. **Detailed Findings** (comprehensive results with visualizations)
4. **Technical Appendix** (statistical details and assumptions)
5. **Recommendations Section** (actionable next steps)
6. **Conclusion and Future Work** (summary and follow-up suggestions)

**Report Guidelines:**
- Write for both technical and business audiences
- Include clear visualizations and tables
- Provide comprehensive documentation
- Ensure professional presentation

Complete your comprehensive report and end with "Data analysis report complete"."""
        }
        
        return prompts.get(self.role, "You are a data analysis specialist.")
    
    def process(self, context: str, data_summary: Dict[str, Any], previous_outputs: List[str] = None) -> DataAgentResponse:
        """Process the analysis request for this agent's specialty."""
        try:
            # Build context for the agent
            full_context = self._build_agent_context(context, data_summary)
            
            # Create prompt
            prompt = ChatPromptTemplate.from_template(
                """{system_prompt}

**Data Context:**
{context}

**Previous Team Analysis:**
{previous_work}

Provide your specialized analysis focusing on your role's deliverables. Be specific, actionable, and thorough."""
            )
            
            # Generate response
            messages = prompt.format_messages(
                system_prompt=self.system_prompt,
                context=full_context,
                previous_work=self._format_previous_outputs(previous_outputs) if previous_outputs else "None"
            )
            
            response = self.llm.invoke(messages)
            
            # Extract recommendations and next steps
            recommendations = self._extract_recommendations(response.content)
            next_steps = self._extract_next_steps(response.content)
            
            return DataAgentResponse(
                role=self.role,
                phase=self._get_phase_for_role(),
                content=response.content,
                recommendations=recommendations,
                next_steps=next_steps,
                confidence=0.85,
                metadata={
                    "data_shape": data_summary.get("shape", "Unknown"),
                    "columns": len(data_summary.get("columns", [])),
                    "agent_version": "1.0"
                }
            )
            
        except Exception as e:
            logger.error(f"Error in {self.role.value} agent: {str(e)}")
            return DataAgentResponse(
                role=self.role,
                phase=self._get_phase_for_role(),
                content=f"Error in {self.role.value} analysis: {str(e)}",
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    def _build_agent_context(self, context: str, data_summary: Dict[str, Any]) -> str:
        """Build comprehensive context for the agent."""
        agent_context = f"Dataset Analysis Context:\n{context}\n\n"
        
        if data_summary:
            agent_context += "Dataset Summary:\n"
            agent_context += f"- Shape: {data_summary.get('shape', 'Unknown')}\n"
            agent_context += f"- Columns: {', '.join(data_summary.get('columns', [])[:10])}\n"
            agent_context += f"- Data Types: {data_summary.get('data_types', {})}\n"
            agent_context += f"- Missing Values: {data_summary.get('missing_summary', 'Not available')}\n\n"
        
        return agent_context
    
    def _format_previous_outputs(self, previous_outputs: List[str]) -> str:
        """Format previous agent outputs for context."""
        if not previous_outputs:
            return "No previous analysis available."
        
        formatted = "Previous Analysis Results:\n\n"
        for i, output in enumerate(previous_outputs[-3:], 1):  # Show last 3 outputs
            formatted += f"Analysis {i}:\n{output[:500]}...\n\n"
        
        return formatted
    
    def _extract_recommendations(self, content: str) -> List[str]:
        """Extract recommendations from agent output."""
        recommendations = []
        lines = content.split('\n')
        
        in_recommendations = False
        for line in lines:
            if any(keyword in line.lower() for keyword in ['recommendation', 'suggest', 'should', 'consider']):
                in_recommendations = True
            elif in_recommendations and line.strip().startswith(('-', '•', '*')):
                recommendations.append(line.strip()[1:].strip())
            elif in_recommendations and line.strip() == '':
                continue
            elif in_recommendations and not line.strip().startswith(('-', '•', '*')):
                in_recommendations = False
        
        return recommendations[:5]  # Limit to top 5
    
    def _extract_next_steps(self, content: str) -> List[str]:
        """Extract next steps from agent output."""
        next_steps = []
        lines = content.split('\n')
        
        in_next_steps = False
        for line in lines:
            if any(keyword in line.lower() for keyword in ['next step', 'follow-up', 'action item']):
                in_next_steps = True
            elif in_next_steps and line.strip().startswith(('-', '•', '*')):
                next_steps.append(line.strip()[1:].strip())
            elif in_next_steps and line.strip() == '':
                continue
            elif in_next_steps and not line.strip().startswith(('-', '•', '*')):
                in_next_steps = False
        
        return next_steps[:3]  # Limit to top 3
    
    def _get_phase_for_role(self) -> DataAnalysisPhase:
        """Get the analysis phase for this agent role."""
        phase_mapping = {
            DataAgentRole.DATA_PROFILER: DataAnalysisPhase.DATA_PROFILING,
            DataAgentRole.STATISTICAL_ANALYST: DataAnalysisPhase.STATISTICAL_ANALYSIS,
            DataAgentRole.VISUALIZATION_SPECIALIST: DataAnalysisPhase.VISUALIZATION,
            DataAgentRole.ML_ADVISOR: DataAnalysisPhase.ML_RECOMMENDATIONS,
            DataAgentRole.INSIGHTS_GENERATOR: DataAnalysisPhase.INSIGHTS_GENERATION,
            DataAgentRole.REPORT_WRITER: DataAnalysisPhase.REPORT_GENERATION
        }
        return phase_mapping.get(self.role, DataAnalysisPhase.INITIALIZATION)

def create_data_analysis_agents(llm) -> Dict[DataAgentRole, DataAnalysisAgent]:
    """Create all specialized data analysis agents."""
    return {
        role: DataAnalysisAgent(role, llm)
        for role in DataAgentRole
    }
