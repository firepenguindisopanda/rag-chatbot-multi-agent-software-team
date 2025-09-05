import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from typing import Dict, List, Tuple, Optional, Any
from .schemas import DataSummary, DataAnalysisType
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handle data loading, cleaning, and basic processing."""
    
    def __init__(self):
        self.data = None
        self.original_data = None
        self.data_summary = None
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV or Excel file."""
        try:
            file_extension = file_path.lower().split('.')[-1]
            
            if file_extension == 'csv':
                # Try different encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        data = pd.read_csv(file_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError("Unable to read CSV file with any supported encoding")
            
            elif file_extension in ['xlsx', 'xls']:
                data = pd.read_excel(file_path)
            
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            self.original_data = data.copy()
            self.data = data
            logger.info(f"Loaded data with shape: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def analyze_data_structure(self) -> DataSummary:
        """Analyze the structure and basic statistics of the data."""
        if self.data is None:
            raise ValueError("No data loaded")
        
        # Basic info
        shape = self.data.shape
        columns = self.data.columns.tolist()
        
        # Data types
        data_types = {}
        numerical_columns = []
        categorical_columns = []
        datetime_columns = []
        
        for col in self.data.columns:
            dtype = str(self.data[col].dtype)
            data_types[col] = dtype
            
            if pd.api.types.is_numeric_dtype(self.data[col]):
                numerical_columns.append(col)
            elif pd.api.types.is_datetime64_any_dtype(self.data[col]):
                datetime_columns.append(col)
            else:
                categorical_columns.append(col)
        
        # Missing values
        missing_values = self.data.isnull().sum().to_dict()
        
        self.data_summary = DataSummary(
            shape=shape,
            columns=columns,
            data_types=data_types,
            missing_values=missing_values,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            datetime_columns=datetime_columns
        )
        
        return self.data_summary
    
    def detect_analysis_type(self, user_context: Optional[str] = None, target_variable: Optional[str] = None) -> DataAnalysisType:
        """Detect the most appropriate analysis type based on data and context."""
        if self.data_summary is None:
            self.analyze_data_structure()
        
        # If user provided context, try to infer from keywords
        if user_context:
            context_lower = user_context.lower()
            if any(keyword in context_lower for keyword in ['predict', 'forecast', 'future']):
                if target_variable and target_variable in self.data_summary.numerical_columns:
                    return DataAnalysisType.REGRESSION
                elif target_variable and target_variable in self.data_summary.categorical_columns:
                    return DataAnalysisType.CLASSIFICATION
                else:
                    return DataAnalysisType.PREDICTIVE
            elif any(keyword in context_lower for keyword in ['cluster', 'group', 'segment']):
                return DataAnalysisType.CLUSTERING
            elif any(keyword in context_lower for keyword in ['time', 'trend', 'temporal']):
                return DataAnalysisType.TIME_SERIES
        
        # Auto-detect based on data characteristics
        if len(self.data_summary.datetime_columns) > 0:
            return DataAnalysisType.TIME_SERIES
        elif target_variable:
            if target_variable in self.data_summary.numerical_columns:
                return DataAnalysisType.REGRESSION
            else:
                return DataAnalysisType.CLASSIFICATION
        else:
            return DataAnalysisType.EXPLORATORY
    
    def clean_data(self) -> pd.DataFrame:
        """Basic data cleaning operations."""
        if self.data is None:
            raise ValueError("No data loaded")
        
        cleaned_data = self.data.copy()
        
        # Remove completely empty rows/columns
        cleaned_data = cleaned_data.dropna(how='all')
        cleaned_data = cleaned_data.dropna(axis=1, how='all')
        
        # Convert object columns that might be numeric
        for col in cleaned_data.select_dtypes(include=['object']).columns:
            try:
                # Try to convert to numeric, but handle exceptions explicitly
                numeric_series = pd.to_numeric(cleaned_data[col], errors='coerce')
                # Only convert if more than 50% of values are successfully converted
                if numeric_series.notna().sum() > len(cleaned_data[col]) * 0.5:
                    cleaned_data[col] = numeric_series
            except Exception:
                pass
        
        self.data = cleaned_data
        return cleaned_data

class DataVisualizer:
    """Create visualizations for data analysis."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        # Set matplotlib backend and style
        try:
            plt.style.use('default')
            sns.set_palette("husl")
            # Ensure we're using a non-interactive backend
            plt.switch_backend('Agg')
        except Exception as e:
            logger.warning(f"Could not set plotting style: {str(e)}")
            # Continue with default settings
    
    def create_overview_plots(self) -> List[str]:
        """Create overview plots for the dataset."""
        plots = []
        
        # Create individual plots to avoid subplot issues
        
        # 1. Missing values plot
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        missing_data = self.data.isnull()
        if missing_data.any().any():
            sns.heatmap(missing_data, cbar=True, ax=ax1, cmap='viridis')
            ax1.set_title('Missing Values Pattern', fontsize=14, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No Missing Values Found', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=16, fontweight='bold')
            ax1.set_title('Missing Values Pattern', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plots.append(self._fig_to_base64(fig1))
        plt.close(fig1)
        
        # 2. Data types distribution
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        dtypes_count = self.data.dtypes.value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(dtypes_count)))
        _, _, _ = ax2.pie(dtypes_count.values, labels=dtypes_count.index, 
                                          autopct='%1.1f%%', colors=colors)
        ax2.set_title('Data Types Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plots.append(self._fig_to_base64(fig2))
        plt.close(fig2)
        
        # 3. Numerical columns distribution
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Limit to first 6 columns to avoid overcrowding
            cols_to_plot = numeric_cols[:6]
            n_cols = len(cols_to_plot)
            fig3, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig3.suptitle('Numerical Columns Distribution', fontsize=16, fontweight='bold')
            axes = axes.flatten()
            
            for i, col in enumerate(cols_to_plot):
                try:
                    self.data[col].hist(bins=20, ax=axes[i], alpha=0.7, color=f'C{i}')
                    axes[i].set_title(f'{col}', fontsize=10)
                    axes[i].tick_params(axis='x', rotation=45)
                except Exception:
                    axes[i].text(0.5, 0.5, f'Error plotting {col}', ha='center', va='center', 
                               transform=axes[i].transAxes)
            
            # Hide unused subplots
            for i in range(n_cols, 6):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plots.append(self._fig_to_base64(fig3))
            plt.close(fig3)
        
        # 4. Correlation heatmap (if enough numeric columns)
        if len(numeric_cols) > 1:
            fig4, ax4 = plt.subplots(figsize=(12, 8))
            corr_matrix = self.data[numeric_cols].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Show only lower triangle
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       mask=mask, ax=ax4, fmt='.2f', square=True)
            ax4.set_title('Correlation Matrix (Numerical Features)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plots.append(self._fig_to_base64(fig4))
            plt.close(fig4)
        
        # 5. Categorical columns analysis (if any)
        cat_cols = self.data.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            # Show first 4 categorical columns
            cols_to_plot = cat_cols[:4]
            fig5, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig5.suptitle('Categorical Columns Distribution', fontsize=16, fontweight='bold')
            axes = axes.flatten()
            
            for i, col in enumerate(cols_to_plot):
                try:
                    value_counts = self.data[col].value_counts()
                    # Limit to top 10 categories to avoid overcrowding
                    if len(value_counts) > 10:
                        value_counts = value_counts.head(10)
                    value_counts.plot(kind='bar', ax=axes[i], color=f'C{i}')
                    axes[i].set_title(f'{col}', fontsize=10)
                    axes[i].tick_params(axis='x', rotation=45)
                except Exception:
                    axes[i].text(0.5, 0.5, f'Error plotting {col}', ha='center', va='center', 
                               transform=axes[i].transAxes)
            
            # Hide unused subplots
            for i in range(len(cols_to_plot), 4):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plots.append(self._fig_to_base64(fig5))
            plt.close(fig5)
        
        # Ensure we always have at least 4 plots
        if len(plots) < 4:
            # Add a summary statistics plot
            fig6, ax6 = plt.subplots(figsize=(12, 8))
            if len(numeric_cols) > 0:
                stats_data = self.data[numeric_cols].describe().T
                stats_data[['mean', 'std', 'min', 'max']].plot(kind='bar', ax=ax6)
                ax6.set_title('Summary Statistics (Numerical Features)', fontsize=14, fontweight='bold')
                ax6.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                plots.append(self._fig_to_base64(fig6))
                plt.close(fig6)
        
        return plots
    
    def create_target_analysis(self, target_column: str) -> List[str]:
        """Create plots specifically for target variable analysis."""
        plots = []
        
        if target_column not in self.data.columns:
            return plots
        
        # Create separate plots instead of subplots to avoid issues
        
        # 1. Target distribution
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        if pd.api.types.is_numeric_dtype(self.data[target_column]):
            self.data[target_column].hist(bins=30, ax=ax1, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_title(f'{target_column} Distribution', fontsize=14, fontweight='bold')
            ax1.set_xlabel(target_column)
            ax1.set_ylabel('Frequency')
        else:
            # Categorical target
            value_counts = self.data[target_column].value_counts()
            if len(value_counts) > 15:  # Limit to top 15 categories
                value_counts = value_counts.head(15)
            value_counts.plot(kind='bar', ax=ax1, color='lightcoral')
            ax1.set_title(f'{target_column} Distribution', fontsize=14, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plots.append(self._fig_to_base64(fig1))
        plt.close(fig1)
        
        # 2. Box plot or pie chart
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        if pd.api.types.is_numeric_dtype(self.data[target_column]):
            # Box plot for numerical data
            self.data.boxplot(column=target_column, ax=ax2)
            ax2.set_title(f'{target_column} Box Plot', fontsize=14, fontweight='bold')
        else:
            # Pie chart for categorical data
            value_counts = self.data[target_column].value_counts()
            if len(value_counts) > 8:  # Limit to top 8 for readability
                others_count = value_counts.iloc[8:].sum()
                value_counts = value_counts.head(8)
                if others_count > 0:
                    value_counts['Others'] = others_count
            
            ax2.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
            ax2.set_title(f'{target_column} Proportions', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plots.append(self._fig_to_base64(fig2))
        plt.close(fig2)
        
        # 3. Correlation with other features (for numerical targets)
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1 and target_column in numeric_cols:
            fig3, ax3 = plt.subplots(figsize=(12, 8))
            correlations = self.data[numeric_cols].corr()[target_column].drop(target_column)
            correlations = correlations.sort_values(key=abs, ascending=True)
            
            # Color bars based on correlation strength
            colors = ['red' if x < -0.3 else 'orange' if x < 0 else 'lightgreen' if x < 0.3 else 'green' for x in correlations]
            correlations.plot(kind='barh', ax=ax3, color=colors)
            ax3.set_title(f'Feature Correlation with {target_column}', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Correlation Coefficient')
            ax3.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            
            plt.tight_layout()
            plots.append(self._fig_to_base64(fig3))
            plt.close(fig3)
        
        # 4. Target vs categorical features (if any categorical columns exist)
        cat_cols = self.data.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0 and target_column in numeric_cols:
            fig4, ax4 = plt.subplots(figsize=(12, 8))
            
            # Choose the categorical column with reasonable number of categories
            suitable_cat_col = None
            for col in cat_cols:
                if len(self.data[col].unique()) <= 10:  # Reasonable number of categories
                    suitable_cat_col = col
                    break
            
            if suitable_cat_col:
                self.data.boxplot(column=target_column, by=suitable_cat_col, ax=ax4)
                ax4.set_title(f'{target_column} by {suitable_cat_col}', fontsize=14, fontweight='bold')
                plt.suptitle('')  # Remove the automatic title
            else:
                ax4.text(0.5, 0.5, 'No suitable categorical columns for comparison', 
                        ha='center', va='center', transform=ax4.transAxes, fontsize=12)
                ax4.set_title('Target vs Categorical Features', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plots.append(self._fig_to_base64(fig4))
            plt.close(fig4)
        
        return plots
    
    def create_feature_importance_plot(self, features: List[str], importance_scores: List[float]) -> str:
        """Create a feature importance plot."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by importance
        sorted_idx = np.argsort(importance_scores)
        sorted_features = [features[i] for i in sorted_idx]
        sorted_scores = [importance_scores[i] for i in sorted_idx]
        
        ax.barh(range(len(sorted_features)), sorted_scores)
        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features)
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance')
        
        plt.tight_layout()
        plot_base64 = self._fig_to_base64(fig)
        plt.close(fig)
        
        return plot_base64
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        try:
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            
            graphic = base64.b64encode(image_png)
            graphic = graphic.decode('utf-8')
            
            return graphic
        except Exception as e:
            logger.error(f"Error converting figure to base64: {str(e)}")
            # Return a placeholder image
            return ""

class MLModelRecommender:
    """Recommend ML models and metrics based on data characteristics."""
    
    def __init__(self, data_summary: DataSummary, analysis_type: DataAnalysisType):
        self.data_summary = data_summary
        self.analysis_type = analysis_type
    
    def recommend_models(self) -> List[str]:
        """Recommend appropriate ML models."""
        recommendations = []
        
        data_size = self.data_summary.shape[0]
        
        if self.analysis_type == DataAnalysisType.REGRESSION:
            if data_size < 1000:
                recommendations.extend([
                    "Linear Regression - Good for small datasets with linear relationships",
                    "Random Forest Regressor - Handles non-linear patterns well",
                    "Support Vector Regression - Effective for small datasets"
                ])
            else:
                recommendations.extend([
                    "Random Forest Regressor - Robust and handles mixed data types",
                    "Gradient Boosting (XGBoost/LightGBM) - Often provides best performance",
                    "Neural Networks - For complex non-linear patterns"
                ])
        
        elif self.analysis_type == DataAnalysisType.CLASSIFICATION:
            if data_size < 1000:
                recommendations.extend([
                    "Logistic Regression - Simple and interpretable",
                    "Random Forest Classifier - Good balance of performance and interpretability",
                    "Support Vector Machine - Effective for small datasets"
                ])
            else:
                recommendations.extend([
                    "Random Forest Classifier - Robust baseline model",
                    "Gradient Boosting (XGBoost/LightGBM) - Often achieves highest accuracy",
                    "Neural Networks - For complex pattern recognition"
                ])
        
        elif self.analysis_type == DataAnalysisType.CLUSTERING:
            recommendations.extend([
                "K-Means - Simple and effective for spherical clusters",
                "DBSCAN - Good for clusters of varying density",
                "Hierarchical Clustering - Provides cluster hierarchy"
            ])
        
        elif self.analysis_type == DataAnalysisType.TIME_SERIES:
            recommendations.extend([
                "ARIMA - Classical time series forecasting",
                "Prophet - Handles seasonality and trends well",
                "LSTM Neural Networks - For complex temporal patterns"
            ])
        
        return recommendations[:3]  # Limit to top 3 recommendations
    
    def recommend_metrics(self) -> List[str]:
        """Recommend appropriate evaluation metrics."""
        metrics = []
        
        if self.analysis_type == DataAnalysisType.REGRESSION:
            metrics.extend([
                "Mean Absolute Error (MAE) - Easy to interpret, robust to outliers",
                "Root Mean Square Error (RMSE) - Penalizes large errors more",
                "R² Score - Explains variance, good for model comparison"
            ])
        
        elif self.analysis_type == DataAnalysisType.CLASSIFICATION:
            metrics.extend([
                "Accuracy - Overall correctness percentage",
                "F1-Score - Balance between precision and recall",
                "ROC-AUC - Good for imbalanced datasets"
            ])
        
        elif self.analysis_type == DataAnalysisType.CLUSTERING:
            metrics.extend([
                "Silhouette Score - Measures cluster separation",
                "Calinski-Harabasz Index - Ratio of between/within cluster variance",
                "Davies-Bouldin Index - Lower values indicate better clustering"
            ])
        
        return metrics
