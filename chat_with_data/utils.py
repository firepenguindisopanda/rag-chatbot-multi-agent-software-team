import os
import tempfile
import shutil
from typing import Optional, List
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def validate_file_upload(file) -> tuple[bool, str]:
    """Validate uploaded file for data analysis."""
    if file is None:
        return False, "No file uploaded"
    
    # Get file extension
    if isinstance(file, str):
        file_path = file
    elif hasattr(file, 'name'):
        file_path = file.name
    else:
        return False, "Invalid file format"
    
    file_extension = file_path.lower().split('.')[-1]
    
    if file_extension not in ['csv', 'xlsx', 'xls']:
        return False, "Only CSV and Excel files are supported"
    
    # Check file size (limit to 50MB)
    try:
        file_size = os.path.getsize(file_path)
        if file_size > 50 * 1024 * 1024:  # 50MB
            return False, "File size too large. Maximum size is 50MB"
    except:
        pass  # Size check might fail for some file objects
    
    return True, "File validation passed"

def save_uploaded_file(file) -> str:
    """Save uploaded file to temporary location and return path."""
    try:
        temp_dir = tempfile.mkdtemp(prefix="chat_with_data_")
        
        if isinstance(file, str):
            # File is already a path, copy it
            filename = os.path.basename(file)
            temp_path = os.path.join(temp_dir, filename)
            shutil.copy2(file, temp_path)
            return temp_path
        
        elif hasattr(file, 'name'):
            # File object with name attribute
            filename = os.path.basename(file.name)
            temp_path = os.path.join(temp_dir, filename)
            shutil.copy2(file.name, temp_path)
            return temp_path
        
        else:
            raise ValueError("Invalid file object")
            
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        raise

def cleanup_temp_files(file_paths: List[str]):
    """Clean up temporary files and directories."""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                # Remove the file
                os.remove(file_path)
                
                # Remove the parent directory if it's empty and appears to be a temp dir
                parent_dir = os.path.dirname(file_path)
                if os.path.basename(parent_dir).startswith("chat_with_data_"):
                    try:
                        os.rmdir(parent_dir)
                    except OSError:
                        pass  # Directory not empty
        except Exception as e:
            logger.warning(f"Could not clean up file {file_path}: {str(e)}")

def format_analysis_output(analysis_result) -> str:
    """Format analysis result for display."""
    output = f"# 📊 Data Analysis Results\n\n"
    
    # Summary
    output += f"## Summary\n{analysis_result.summary}\n\n"
    
    # Insights
    if analysis_result.insights:
        output += "## 🔍 Key Insights\n"
        for i, insight in enumerate(analysis_result.insights, 1):
            output += f"{i}. {insight}\n"
        output += "\n"
    
    # ML Recommendations
    if analysis_result.recommended_ml_models:
        output += "## 🤖 Recommended ML Models\n"
        for i, model in enumerate(analysis_result.recommended_ml_models, 1):
            output += f"{i}. {model}\n"
        output += "\n"
    
    # Metrics
    if analysis_result.suggested_metrics:
        output += "## 📈 Suggested Evaluation Metrics\n"
        for i, metric in enumerate(analysis_result.suggested_metrics, 1):
            output += f"{i}. {metric}\n"
        output += "\n"
    
    # Next Steps
    if analysis_result.next_steps:
        output += "## 🎯 Next Steps\n"
        for i, step in enumerate(analysis_result.next_steps, 1):
            output += f"{i}. {step}\n"
        output += "\n"
    
    # Visualizations note
    if analysis_result.visualizations:
        output += f"## 📊 Visualizations\n"
        output += f"Generated {len(analysis_result.visualizations)} visualization(s) to help understand your data.\n\n"
    
    output += "*Analysis complete! You can now ask questions about your data using the chat interface below.*"
    
    return output

def get_sample_data_info(data: pd.DataFrame, n_rows: int = 5) -> str:
    """Get sample data information for display."""
    info = f"**Dataset Shape:** {data.shape[0]} rows × {data.shape[1]} columns\n\n"
    
    info += f"**Column Information:**\n"
    for col in data.columns[:10]:  # Show first 10 columns
        dtype = str(data[col].dtype)
        non_null = data[col].count()
        info += f"- `{col}`: {dtype} ({non_null} non-null)\n"
    
    if len(data.columns) > 10:
        info += f"- ... and {len(data.columns) - 10} more columns\n"
    
    info += f"\n**Sample Data (first {n_rows} rows):**\n"
    info += data.head(n_rows).to_markdown(index=False)
    
    return info

def create_data_summary_table(data_summary) -> str:
    """Create a formatted summary table of the dataset."""
    summary = "## 📋 Dataset Overview\n\n"
    
    summary += f"| Metric | Value |\n"
    summary += f"|--------|-------|\n"
    summary += f"| Rows | {data_summary.shape[0]:,} |\n"
    summary += f"| Columns | {data_summary.shape[1]} |\n"
    summary += f"| Numerical Features | {len(data_summary.numerical_columns)} |\n"
    summary += f"| Categorical Features | {len(data_summary.categorical_columns)} |\n"
    summary += f"| DateTime Features | {len(data_summary.datetime_columns)} |\n"
    summary += f"| Total Missing Values | {sum(data_summary.missing_values.values())} |\n"
    
    missing_percent = (sum(data_summary.missing_values.values()) / (data_summary.shape[0] * data_summary.shape[1])) * 100
    summary += f"| Missing Data % | {missing_percent:.2f}% |\n"
    
    return summary

def extract_column_suggestions(data: pd.DataFrame) -> dict:
    """Extract suggestions for target variables and analysis types."""
    suggestions = {
        "potential_targets": [],
        "categorical_targets": [],
        "numerical_targets": [],
        "datetime_columns": []
    }
    
    # Analyze columns
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            # Check if it looks like a target variable
            if any(keyword in col.lower() for keyword in ['target', 'label', 'class', 'outcome', 'result', 'score', 'price', 'value']):
                suggestions["potential_targets"].append(col)
            suggestions["numerical_targets"].append(col)
        
        elif pd.api.types.is_datetime64_any_dtype(data[col]):
            suggestions["datetime_columns"].append(col)
        
        else:  # Categorical
            # Check for binary classification
            unique_values = data[col].nunique()
            if unique_values <= 10:  # Reasonable number of categories
                if any(keyword in col.lower() for keyword in ['target', 'label', 'class', 'category', 'type']):
                    suggestions["potential_targets"].append(col)
                suggestions["categorical_targets"].append(col)
    
    return suggestions
